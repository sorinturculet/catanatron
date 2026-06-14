"""
V20: Mixture-of-Experts Hierarchical PPO (15M steps)

Loads the v19 trained model (30M steps, 21.2% win rate) and adds a
Mixture-of-Experts (MoE) policy head on top of the shared CNN backbone.

Architecture:
    CNN [64,128,256,512] -> 512-dim features
        ├── Gate Network: Linear(512,128) -> LeakyReLU -> Linear(128,2) -> Softmax
        ├── OWS Policy MLP [2048,1024,512,256] -> Linear(256,290)       (init from v19)
        ├── Expansion Policy MLP [2048,1024,512,256] -> Linear(256,290) (init from v19)
        └── Value MLP [2048,1024,512,256] -> Linear(256,1)              (init from v19)

    Final logits = w_ows * logits_ows + w_expansion * logits_expansion

Key differences from v19:
    - Custom MoEMaskablePolicy with 2 expert heads + learned gate
    - Both expert heads initialised from v19 (both start competent)
    - Gate starts near-uniform, learns board-conditioned strategy selection
    - CNN frozen for first 2M steps so gate can stabilise
    - LR reset: 5e-5 -> 1e-5 (same pattern that unlocked gains in v19)
    - Reward pipeline UNCHANGED: production_aware_rewards + ValueWeightedWrapper 0.05

Why NOT v14/v16 mistake:
    - v14/v16 added strategy as extra reward signals -> diluted expert bonus
    - v20 changes policy STRUCTURE only -> no extra rewards whatsoever
    - The gate learns strategy selection internally from CNN features
"""

import math
import os
import random
import glob
import re

os.environ["WANDB_DISABLE_SYMLINKS"] = "True"
from typing import Any
import atexit
import time
import multiprocessing

from functools import partial
import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from reward_functions import production_aware_rewards, mask_fn
from value_weighted_wrapper import ValueWeightedWrapper
from moe_policy import MoEMaskablePolicy, load_v19_weights_into_moe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir, experiment_prefix):
    pattern = os.path.join(checkpoint_dir, f"{experiment_prefix}_*_steps.zip")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return None, 0
    latest_checkpoint = None
    max_timesteps = 0
    for checkpoint_file in checkpoint_files:
        match = re.search(r'_(\d+)_steps\.zip$', checkpoint_file)
        if match:
            timesteps = int(match.group(1))
            if timesteps > max_timesteps:
                max_timesteps = timesteps
                latest_checkpoint = checkpoint_file
    return latest_checkpoint, max_timesteps


def learning_rate_schedule(initial_lr, final_lr):
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CNNFreezeCallback(BaseCallback):
    """Freeze CNN for the first `freeze_steps` to let gate + expert heads stabilise."""

    def __init__(self, freeze_steps: int = 2_000_000, verbose: int = 0):
        super().__init__(verbose)
        self.freeze_steps = freeze_steps
        self._frozen = False

    def _on_training_start(self) -> None:
        for param in self.model.policy.features_extractor.parameters():
            param.requires_grad = False
        self._frozen = True
        print(f"[CNNFreezeCallback] CNN frozen for first {self.freeze_steps:,} steps.")

    def _on_step(self) -> bool:
        if self._frozen and self.num_timesteps >= self.freeze_steps:
            for param in self.model.policy.features_extractor.parameters():
                param.requires_grad = True
            self._frozen = False
            print(f"[CNNFreezeCallback] CNN unfrozen at step {self.num_timesteps:,}.")
        return True


class GateLoggingCallback(BaseCallback):
    """Log MoE gate weights and entropy to WandB."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            gate_weights = getattr(self.model.policy, "_last_gate_weights", None)
            if gate_weights is not None and gate_weights.numel() > 0:
                mean_w = gate_weights.mean(dim=0)  # [num_experts]
                # Entropy: -sum(w * log(w)), higher = more balanced usage
                eps = 1e-8
                entropy = -(mean_w * (mean_w + eps).log()).sum().item()
                log_dict = {
                    "gate/ows_weight": mean_w[0].item(),
                    "gate/expansion_weight": mean_w[1].item(),
                    "gate/entropy": entropy,
                }
                wandb.log(log_dict, step=self.num_timesteps)
        return True


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper statistics to WandB."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get("infos", [])
                if infos:
                    def _mean(key):
                        vals = [i.get(key, 0) for i in infos if key in i]
                        return np.mean(vals) if vals else None

                    log_dict = {}
                    for key, wkey in [
                        ("value_bonus",            "value/bonus"),
                        ("value_bonus_scale",       "value/bonus_scale"),
                        ("value_normalized_score",  "value/normalized_score"),
                        ("value_avg_score",         "value/avg_score"),
                        ("value_best_action_rate",  "value/best_action_rate"),
                    ]:
                        v = _mean(key)
                        if v is not None:
                            log_dict[wkey] = v
                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ===== V20: MoE Hierarchical PPO (15M steps) =====
    total_timesteps = 15_000_000

    # Architecture (same backbone as v15-v19)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512   # must match v19 (CustomCNN was built with features_dim=512)
    num_experts = 2

    # LR: reset higher (same trick that worked v18->v19)
    initial_lr = 5e-5
    final_lr = 1e-5

    # Entropy coefficient (same as v15-v19)
    ent_coef = 0.02

    # ValueWeightedWrapper: constant at floor (same as v18/v19)
    bonus_scale = 0.05
    decay_start_step = 999_999_999
    decay_rate = 0.99999
    min_scale = 0.05

    # CNN frozen for first N steps
    cnn_freeze_steps = 2_000_000

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]
    reward_function = partial(production_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = production_aware_rewards.__name__
    representation = "mixed"

    gamma = 0.99
    gae_lambda = 0.95
    seed = 42
    n_envs = 16
    n_steps = 512
    batch_size = 512
    n_epochs = 4
    max_grad_norm = 0.5

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    # Experiment name
    iters = round(math.log(total_timesteps, 10))
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = (
        f"ppo_v20-MoE-{num_experts}experts-{iters}-{batch_size}-{gamma}"
        f"-{enemy_desc}-{reward_function.__name__}-{representation}"
        f"-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    )
    print(experiment_name)

    config = {
        "version": "v20_moe_hierarchical",
        "approach": "mixture_of_experts_gated_policy",
        "base_model": "model_v19 (30M steps, 21.2% win rate)",
        "num_experts": num_experts,
        "expert_names": ["OWS (cities+dev cards)", "Expansion (settlements+roads)"],
        "cnn_freeze_steps": cnn_freeze_steps,
        "total_timesteps": total_timesteps,
        "total_timesteps_including_prev": 45_000_000,
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "net_arch": net_arch,
        "features_dim": features_dim,
        "cnn_arch": cnn_arch,
        "activation_fn": activation_fn.__name__,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": [str(e) for e in enemies],
        "reward_function": reward_function.__name__,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "bonus_scale": bonus_scale,
        "decay_start_step": "never (floor)",
        "min_scale": min_scale,
    }

    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
        name="v20",
    )

    def print_name():
        print(experiment_name)
    atexit.register(print_name)

    per_env_decay_start = decay_start_step // n_envs

    def make_env(rank, seed=0):
        def _init():
            env = gym.make(
                env_name,
                config={
                    "map_type": map_type,
                    "vps_to_win": vps_to_win,
                    "enemies": enemies,
                    "reward_function": reward_function,
                    "representation": representation,
                    "normalized": True,
                },
            )
            env = ActionMasker(env, mask_fn)
            env = ValueWeightedWrapper(
                env,
                player_color=Color.BLUE,
                bonus_scale=bonus_scale,
                decay_start_step=per_env_decay_start,
                decay_rate=decay_rate,
                min_scale=min_scale,
            )
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_v19_path = os.path.join(script_dir, "model_v19")
    model_path = os.path.join(script_dir, "model_v20")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V20: MoE HIERARCHICAL)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    # Priority 1: Resume from a v20 checkpoint (interrupted run)
    if latest_checkpoint:
        print(f"\n  Found v20 checkpoint at {checkpoint_timesteps:,} steps")
        print(f"  Path: {latest_checkpoint}")
        try:
            model = MaskablePPO.load(
                latest_checkpoint,
                env,
                device=device,
                custom_objects={
                    "features_extractor_class": CustomCNN,
                    "policy_class": MoEMaskablePolicy,
                },
            )
            model.gamma = gamma
            model.gae_lambda = gae_lambda
            model.ent_coef = ent_coef
            model.learning_rate = lr_schedule
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.max_grad_norm = max_grad_norm
            model._setup_lr_schedule()
            timesteps_completed = checkpoint_timesteps
            model_loaded = True
            print(f"  Resumed v20 from step {timesteps_completed:,}")
        except Exception as e:
            print(f"  Failed to load v20 checkpoint: {e}")

    # Priority 2: Create fresh MoE model and load v19 weights
    if not model_loaded and os.path.exists(model_v19_path + ".zip"):
        print(f"\n  Creating fresh MoE model and loading v19 weights...")
        policy_kwargs: Any = dict(
            activation_fn=activation_fn,
            net_arch=net_arch[0],
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(cnn_arch=cnn_arch, features_dim=features_dim),
            num_experts=num_experts,
        )
        model = MaskablePPO(
            MoEMaskablePolicy,
            env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
            device=device,
            seed=seed,
        )
        load_v19_weights_into_moe(model, model_v19_path)
        model_loaded = True
        print(f"  MoE model created and v19 weights loaded.")

    # Priority 3: Fallback — train from scratch (should not happen)
    if not model_loaded:
        print(f"\n  WARNING: model_v19 not found. Training MoE from scratch.")
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=net_arch[0],
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(cnn_arch=cnn_arch, features_dim=features_dim),
            num_experts=num_experts,
        )
        model = MaskablePPO(
            MoEMaskablePolicy,
            env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
            device=device,
            seed=seed,
        )

    remaining_timesteps = total_timesteps - timesteps_completed

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V20: MoE HIERARCHICAL)")
    print("=" * 70)
    print(f"Approach:              MoE — {num_experts} expert heads + learned gate")
    print(f"Expert 0:              OWS  (cities + dev cards + largest army)")
    print(f"Expert 1:              Expansion (settlements + roads + longest road)")
    print(f"Base model:            model_v19 (30M steps, 21.2% win rate)")
    print(f"New steps target:      {total_timesteps:>15,}")
    print(f"Already completed:     {timesteps_completed:>15,}")
    print(f"Remaining:             {remaining_timesteps:>15,}")
    print(f"Total (all versions):  {45_000_000:>15,}")
    print(f"\nCNN frozen:            first {cnn_freeze_steps:,} steps")
    print(f"Learning rate:         {initial_lr:.2e} -> {final_lr:.2e} (reset from v19 floor)")
    print(f"Expert bonus:          {bonus_scale} constant (v15 floor)")
    print(f"Reward:                {reward_function.__name__}")
    print(f"Envs:                  {n_envs}  |  Batch: {batch_size}  |  Epochs: {n_epochs}")
    print("=" * 70 + "\n")

    if remaining_timesteps <= 0:
        print("Training already completed!")
        model.save(model_path)
        run.finish()
        return

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=checkpoint_dir,
        name_prefix=experiment_name,
    )
    cnn_freeze_callback = CNNFreezeCallback(freeze_steps=cnn_freeze_steps)
    gate_callback = GateLoggingCallback(log_freq=10_000)
    value_callback = ValueScoreLoggingCallback(log_freq=10_000)
    callback = CallbackList([checkpoint_callback, cnn_freeze_callback, gate_callback, value_callback])

    print("=" * 70)
    print("STARTING TRAINING (V20: MoE HIERARCHICAL)")
    print("=" * 70)
    print(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run:  https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print(f"\nWATCH: gate/entropy should stay > 0.3 (both experts used).")
    print(f"ABORT:  if gate/entropy < 0.1 after 3M steps -> one expert dominated.")
    print("=" * 70 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback)

    model.save(model_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (V20: MoE HIERARCHICAL)")
    print("=" * 70)
    print(f"Time:         {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
