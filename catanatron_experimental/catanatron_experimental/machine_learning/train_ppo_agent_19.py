"""
V19: Extended V18 Training (30M total steps)

Continues v18's recipe for 30M total steps (10M v15 + 10M v18 + 10M v19).
Loads from the v18 trained model.

Key changes from v18:
- Loads from model_v18 checkpoint (with model_v15 as fallback)
- LR reset to 5e-5 -> 1e-5 (v18 bottomed out at 1e-5, give more gradient signal)
- Expert bonus stays at floor 0.05 (proven stable)

Why: WandB showed v18 still improving at end of training:
- Episode reward still climbing (+8.3% in last 20%)
- Best action rate rose 46.4% -> 50.7%
- Avg score improved 0.68 -> 0.70
- But LR hit floor, limiting further gains. Reset gives fresh learning headroom.
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
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb
from wandb.integration.sb3 import WandbCallback

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
)
from catanatron.models.player import RandomPlayer
from reward_functions import production_aware_rewards, mask_fn
from value_weighted_wrapper import ValueWeightedWrapper


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


class ValueScoreLoggingCallback(BaseCallback):
    """Callback to log ValueWeightedWrapper statistics to WandB."""

    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                if infos:
                    bonuses = [
                        info.get('value_bonus', 0)
                        for info in infos
                        if 'value_bonus' in info
                    ]
                    scales = [
                        info.get('value_bonus_scale', 0)
                        for info in infos
                        if 'value_bonus_scale' in info
                    ]
                    scores = [
                        info.get('value_normalized_score', 0)
                        for info in infos
                        if 'value_normalized_score' in info
                    ]
                    avg_scores = [
                        info.get('value_avg_score', 0)
                        for info in infos
                        if 'value_avg_score' in info
                    ]
                    best_rates = [
                        info.get('value_best_action_rate', 0)
                        for info in infos
                        if 'value_best_action_rate' in info
                    ]

                    log_dict = {}
                    if bonuses:
                        log_dict['value/bonus'] = np.mean(bonuses)
                    if scales:
                        log_dict['value/bonus_scale'] = np.mean(scales)
                    if scores:
                        log_dict['value/normalized_score'] = np.mean(scores)
                    if avg_scores:
                        log_dict['value/avg_score'] = np.mean(avg_scores)
                    if best_rates:
                        log_dict['value/best_action_rate'] = np.mean(best_rates)

                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass

        return True


def main():
    # ===== V19: Extended V18 (30M total steps) =====
    total_timesteps = 10_000_000  # This run trains 10M MORE steps on top of v18's 20M

    # Network architecture (same as v15/v18)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[2048, 1024, 512, 256],
            pi=[2048, 1024, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU

    # Reset LR: v18 bottomed out at 1e-5, give fresh gradient headroom
    initial_lr = 5e-5
    final_lr = 1e-5

    # Entropy coefficient (same as v15/v18)
    ent_coef = 0.02

    # ValueWeightedWrapper: constant at floor (same as v18)
    bonus_scale = 0.05
    decay_start_step = 999_999_999
    decay_rate = 0.99999
    min_scale = 0.05

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"

    # Opponent (same as v15/v18)
    enemies = [ValueFunctionPlayer(Color.RED)]

    # Reward function (same as v15/v18)
    reward_function = partial(production_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = production_aware_rewards.__name__
    representation = "mixed"

    # Core PPO parameters (same as v15/v18)
    gamma = 0.99
    gae_lambda = 0.95
    normalized = False
    seed = 42

    # Parallelization (same as v15/v18)
    n_envs = 16
    n_steps = 512
    batch_size = 512
    n_epochs = 4
    max_grad_norm = 0.5

    assert (n_envs * n_steps) % batch_size == 0, "batch_size must divide n_envs * n_steps"

    start_time = time.time()

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    # Build Experiment Name
    iters = round(math.log(total_timesteps, 10))
    arch_str = (
        activation_fn.__name__
        + "x".join([str(i) for i in net_arch[:-1]])
        + "+"
        + "vf="
        + "x".join([str(i) for i in net_arch[-1]["vf"]])
        + "+"
        + "pi="
        + "x".join([str(i) for i in net_arch[-1]["pi"]])
    )
    if representation == "mixed":
        arch_str = "Cnn" + "x".join([str(i) for i in cnn_arch]) + "+" + arch_str
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = f"ppo_v19-ExtendedV18-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    print(experiment_name)

    # WandB config
    config = {
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
        "total_timesteps_including_prev": 30_000_000,
        "net_arch": net_arch,
        "activation_fn": activation_fn.__name__,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": [str(enemy) for enemy in enemies],
        "reward_function": reward_function.__name__,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "normalized": normalized,
        "cnn_arch": cnn_arch,
        "experiment_name": experiment_name,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "version": "v19_extended_v18",
        "approach": "extended_production_aware_value_imitation",
        "base_model": "model_v18 (20M steps)",
        # ValueWeightedWrapper params
        "bonus_scale": bonus_scale,
        "decay_start_step": "never (already at floor)",
        "min_scale": min_scale,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
        name="v19",
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Per-env decay threshold (set very high so decay never triggers)
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

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_v18_path = os.path.join(script_dir, "model_v18")
    model_v15_path = os.path.join(script_dir, "model_v15")
    model_path = os.path.join(script_dir, "model_v19")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 80)
    print("MODEL LOADING (V19: EXTENDED V18)")
    print("=" * 80)

    # Priority 1: Resume from v19 checkpoint (if we already started v19)
    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    if latest_checkpoint:
        print(f"\n  Found v19 checkpoint at {checkpoint_timesteps:,} timesteps")
        print(f"  Path: {latest_checkpoint}")
        try:
            model = MaskablePPO.load(latest_checkpoint, env, device=device)
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
            print(f"  Successfully loaded v19 checkpoint!")
            print(f"  Resuming from timestep {timesteps_completed:,}")
        except Exception as e:
            print(f"  Failed to load v19 checkpoint: {e}")

    # Priority 2: Load from v18 trained model (bootstrap)
    if not model_loaded and os.path.exists(model_v18_path + ".zip"):
        print(f"\n  Loading v18 trained model as starting point...")
        print(f"  Path: {model_v18_path}.zip")
        try:
            model = MaskablePPO.load(model_v18_path, env, device=device)
            model.gamma = gamma
            model.gae_lambda = gae_lambda
            model.ent_coef = ent_coef
            model.learning_rate = lr_schedule
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.max_grad_norm = max_grad_norm
            model._setup_lr_schedule()
            model_loaded = True
            print(f"  Successfully loaded v18 model!")
            print(f"  Training 10M more steps with lr ({initial_lr} -> {final_lr})")
            print(f"  Expert bonus constant at floor: {bonus_scale}")
        except Exception as e:
            print(f"  Failed to load v18 model: {e}")

    # Priority 3: Load from v15 trained model (fallback)
    if not model_loaded and os.path.exists(model_v15_path + ".zip"):
        print(f"\n  WARNING: v18 model not found, falling back to v15...")
        print(f"  Path: {model_v15_path}.zip")
        try:
            model = MaskablePPO.load(model_v15_path, env, device=device)
            model.gamma = gamma
            model.gae_lambda = gae_lambda
            model.ent_coef = ent_coef
            model.learning_rate = lr_schedule
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.max_grad_norm = max_grad_norm
            model._setup_lr_schedule()
            model_loaded = True
            print(f"  Successfully loaded v15 model as fallback!")
        except Exception as e:
            print(f"  Failed to load v15 model: {e}")

    # Priority 4: Fresh model (shouldn't happen)
    if not model_loaded:
        print(f"\n  WARNING: No v18 or v15 model found! Training from scratch.")
        print(f"  This is not intended for v19. Check that model_v18.zip exists.")
        print(f"\nCreating new model from scratch...")
        policy_kwargs: Any = dict(activation_fn=activation_fn, net_arch=net_arch[0])
        if representation == "mixed":
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(
                cnn_arch=cnn_arch, features_dim=512
            )
        model = MaskablePPO(
            MaskableActorCriticPolicy,
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

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION (V19: EXTENDED V18)")
    print("=" * 80)
    print(f"Approach:                Extended V18 (10M more steps)")
    print(f"Base model:              model_v18 (20M steps trained)")
    print(f"New timesteps target:    {total_timesteps:>15,}")
    print(f"Already completed:       {timesteps_completed:>15,}")
    print(f"Remaining timesteps:     {remaining_timesteps:>15,}")
    print(f"Total with v15+v18:      {30_000_000:>15,}")
    print(f"\nBase reward:             {reward_function.__name__:>15}")
    print(f"  - VP gain:             +0.1 per VP")
    print(f"  - VP loss:             -0.1 per VP")
    print(f"  - Production gain:     +0.02 per prod point")
    print(f"  - Win bonus:           +1.0")
    print(f"  - Loss penalty:        -1.0")
    print(f"\nValue-Weighted Bonus (constant at floor):")
    print(f"  - Bonus scale:         {bonus_scale:>15} (v15 floor)")
    print(f"  - Decay:               disabled (already at floor)")
    print(f"  - Min scale:           {min_scale:>15}")
    print(f"\nParallel environments:   {n_envs:>15,}")
    print(f"Steps per rollout:       {n_steps:>15,}")
    print(f"Batch size:              {batch_size:>15,}")
    print(f"Epochs per update:       {n_epochs:>15,}")
    print(f"Checkpoint frequency:    {500_000:>15,} steps")
    print(f"\nLearning rate:           {initial_lr:>15.6f} -> {final_lr:.06f}")
    print(f"  (reset higher than v18's floor for fresh gradient signal)")
    print(f"Entropy coefficient:     {ent_coef:>15.4f}")
    print(f"Gamma (discount):        {gamma:>15.4f}")
    print(f"GAE lambda:              {gae_lambda:>15.4f}")
    print("=" * 80 + "\n")

    if remaining_timesteps <= 0:
        print("Training already completed!")
        model.save(model_path)
        run.finish()
        return

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    value_callback = ValueScoreLoggingCallback(log_freq=10000)
    callback = CallbackList([checkpoint_callback, value_callback])

    print("=" * 80)
    print("STARTING TRAINING (V19: EXTENDED V18)")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run: https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("=" * 80 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback)

    # Save final model
    model.save(model_path)

    # Training completion summary
    elapsed_time = time.time() - start_time
    timesteps_per_second = remaining_timesteps / elapsed_time
    hours = elapsed_time / 3600

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED (V19: EXTENDED V18)")
    print("=" * 80)
    print(f"Total training time:     {elapsed_time:>15.2f} seconds ({hours:.2f} hours)")
    print(f"Timesteps trained:       {remaining_timesteps:>15,}")
    print(f"Throughput:              {timesteps_per_second:>15.2f} timesteps/second")
    print(f"Final model saved:       {model_path}")
    print(f"Checkpoints saved in:    {checkpoint_dir}")
    print("=" * 80 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
