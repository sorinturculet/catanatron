"""
V33: Potential-Based Reward Shaping (Ng et al. 1999) — FRESH start.

Instead of hand-crafted rewards (VP +0.10, settle +0.25, road +0.12, etc.),
use VFP's complete heuristic evaluation as a potential function:

    F(s, s') = γΦ(s') - Φ(s)

where Φ(s) = base_fn(s) from ValueFunctionPlayer.

Theory (Ng et al. 1999): PBRS preserves the optimal policy while providing
dense, informative reward signal derived from the FULL expert heuristic.

Why fresh start (not from v28):
  - v28's value head was calibrated for expansion_heavy_rewards
  - PBRS produces fundamentally different reward magnitudes
  - Loading v28 would give a broken value head → noisy PPO updates
  - PBRS is dense enough for fast learning from scratch

Config:
  - FRESH start (no checkpoint)
  - Reward: pbrs_rewards (PBRS from VFP heuristic)
  - Expert guidance: ValueWeightedWrapper (0.2 → 0.05, action-level)
  - 4-stage LR: 3e-4→3e-5 / 1e-4→1e-5 / 3e-5→5e-6 / 1e-5→2e-6
  - 65M steps, 32 envs, gamma=0.997
"""

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

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from reward_functions import pbrs_rewards, mask_fn
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


def make_absolute_lr_schedule(total_timesteps, offset, stages):
    """
    Multi-stage LR schedule immune to SB3's progress_remaining reset bug.
    Uses absolute step counting to determine correct LR stage.
    """
    remaining = total_timesteps - offset

    def schedule(progress_remaining):
        steps_in_this_run = (1.0 - progress_remaining) * remaining
        abs_step = offset + steps_in_this_run
        abs_frac = abs_step / total_timesteps

        for frac_start, frac_end, lr_start, lr_end in stages:
            if abs_frac < frac_end:
                local_frac = (abs_frac - frac_start) / (frac_end - frac_start)
                local_frac = max(0.0, min(1.0, local_frac))
                return lr_start + (lr_end - lr_start) * local_frac

        return stages[-1][3]

    return schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper metrics + actual LR + LR stage to WandB."""

    def __init__(self, total_timesteps, offset, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_timesteps = total_timesteps
        self.offset = offset

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                abs_step = self.offset + self.num_timesteps
                log_dict = {"train/absolute_step": abs_step}

                if infos:
                    for key, wkey in [
                        ("value_bonus",           "value/bonus"),
                        ("value_bonus_scale",      "value/bonus_scale"),
                        ("value_normalized_score", "value/normalized_score"),
                        ("value_avg_score",        "value/avg_score"),
                        ("value_best_action_rate", "value/best_action_rate"),
                    ]:
                        vals = [i.get(key, 0) for i in infos if key in i]
                        if vals:
                            log_dict[wkey] = np.mean(vals)

                # Log actual LR
                lr = self.model.lr_schedule(self.model._current_progress_remaining)
                log_dict["train/learning_rate_actual"] = lr

                # Log LR stage
                frac = abs_step / self.total_timesteps
                if frac < 0.25:
                    log_dict["train/lr_stage"] = 1
                elif frac < 0.50:
                    log_dict["train/lr_stage"] = 2
                elif frac < 0.75:
                    log_dict["train/lr_stage"] = 3
                else:
                    log_dict["train/lr_stage"] = 4

                wandb.log(log_dict, step=abs_step)
            except Exception:
                pass
        return True


def main():
    total_timesteps = 65_000_000

    # 4-stage LR schedule (same as v28)
    lr_stages = [
        (0.00, 0.25, 3e-4, 3e-5),   # Stage 1 ( 0–16.25M): exploration
        (0.25, 0.50, 1e-4, 1e-5),   # Stage 2 (16.25–32.5M): refine
        (0.50, 0.75, 3e-5, 5e-6),   # Stage 3 (32.5–48.75M): fine-tune
        (0.75, 1.00, 1e-5, 2e-6),   # Stage 4 (48.75–65M): final polish
    ]

    # Architecture — same as v28
    gamma      = 0.997
    gae_lambda = 0.95
    ent_coef   = 0.02
    n_envs     = 32
    n_steps    = 512
    batch_size = 1024
    n_epochs   = 4
    max_grad_norm = 0.5
    seed       = 42

    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    # Expert bonus: 0.2 → 0.05 floor at 5M (same as v28)
    bonus_scale      = 0.2
    decay_start_step = 5_000_000
    decay_rate       = 0.99999
    min_scale        = 0.05

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"
    enemies      = [ValueFunctionPlayer(Color.RED)]

    reward_function = partial(pbrs_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = pbrs_rewards.__name__

    representation = "mixed"

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    experiment_name = "ppo_v33-PBRS-from-v28"

    config = {
        "version": "v33_pbrs_fresh",
        "approach": "Potential-Based Reward Shaping (Ng et al. 1999) — fresh start",
        "base_model": "none (fresh)",
        "total_timesteps": total_timesteps,
        "lr_stage1": "3e-4 → 3e-5  (0-16.25M, exploration)",
        "lr_stage2": "1e-4 → 1e-5  (16.25-32.5M, refine)",
        "lr_stage3": "3e-5 → 5e-6  (32.5-48.75M, fine-tune)",
        "lr_stage4": "1e-5 → 2e-6  (48.75-65M, final polish)",
        "reward_function": "pbrs_rewards",
        "reward_notes": "F(s,s') = γΦ(s') - Φ(s), Φ = VFP base_fn heuristic, normalized by 3e15",
        "theory": "Ng et al. 1999 — policy invariance under potential-based shaping",
        "bonus_scale": bonus_scale,
        "decay_start_step": decay_start_step,
        "min_scale": min_scale,
        "ent_coef": ent_coef,
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_envs": n_envs,
        "batch_size": batch_size,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v33")

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

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(script_dir, "model_v33")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V33: PBRS — FRESH START)")
    print("=" * 70)

    # Check for v33 checkpoint (resume interrupted training)
    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    if latest_checkpoint:
        print(f"\n  Found v33 checkpoint at {checkpoint_timesteps:,} steps")
        try:
            lr_schedule = make_absolute_lr_schedule(
                total_timesteps, checkpoint_timesteps, lr_stages
            )
            model = MaskablePPO.load(
                latest_checkpoint, env, device=device,
                custom_objects={"features_extractor_class": CustomCNN},
            )
            model.gamma         = gamma
            model.gae_lambda    = gae_lambda
            model.ent_coef      = ent_coef
            model.learning_rate = lr_schedule
            model.batch_size    = batch_size
            model.n_epochs      = n_epochs
            model.max_grad_norm = max_grad_norm
            model._setup_lr_schedule()
            timesteps_completed = checkpoint_timesteps
            model_loaded = True
            print(f"  Resuming v33 from {timesteps_completed:,} steps")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")

    if not model_loaded:
        print(f"\n  Starting fresh with PBRS rewards")
        lr_schedule = make_absolute_lr_schedule(total_timesteps, 0, lr_stages)
        policy_kwargs: Any = dict(
            activation_fn=activation_fn,
            net_arch=net_arch[0],
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(cnn_arch=cnn_arch, features_dim=features_dim),
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

    # Verify LR schedule
    test_lr_start = lr_schedule(1.0)
    test_lr_end   = lr_schedule(0.0)
    print(f"\n  LR verification:")
    print(f"    schedule(1.0) = {test_lr_start:.2e}")
    print(f"    schedule(0.0) = {test_lr_end:.2e}")

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V33: PBRS)")
    print("=" * 70)
    print(f"Total timesteps:      {total_timesteps:>15,}")
    print(f"Already completed:    {timesteps_completed:>15,}")
    print(f"Remaining:            {remaining_timesteps:>15,}")
    print(f"\nReward: POTENTIAL-BASED REWARD SHAPING (Ng et al. 1999)")
    print(f"  Φ(s) = VFP base_fn heuristic (13+ features)")
    print(f"  F(s,s') = γΦ(s') - Φ(s), normalized by 3e15")
    print(f"  Terminal: Win +1.0, Loss -1.0")
    print(f"\nExpert action guidance (ValueWeightedWrapper):")
    print(f"  bonus: {bonus_scale} → {min_scale} floor at {decay_start_step/1e6:.0f}M")
    print(f"\nEnvs / Steps / Batch: {n_envs} / {n_steps} / {batch_size}")
    print(f"Gamma: {gamma}  |  Ent coef: {ent_coef}")
    print("=" * 70 + "\n")

    if remaining_timesteps <= 0:
        print("Training already completed!")
        model.save(model_path)
        run.finish()
        return

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    value_callback = ValueScoreLoggingCallback(
        total_timesteps=total_timesteps,
        offset=timesteps_completed,
        log_freq=10_000,
    )
    callback = CallbackList([checkpoint_callback, value_callback])

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run:  https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("=" * 70 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback)

    model.save(model_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED (V33 — PBRS)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
