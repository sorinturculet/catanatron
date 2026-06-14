"""
V37: Diverse heuristic opponents + production-aware reward from v36.

Replaces failed NN self-play (6 FPS) with diverse VFP opponent pool (650 FPS).
  - Loads model_v36.zip (68.8% vs VFP)
  - SGDR warm restart: 2e-5 -> 5e-6 over 50M steps
  - 5 VFP variants (default, expansion, devcard, contender, noisy) sampled
    per episode — forces generalization instead of single-opponent overfitting
  - production_aware_rewards (VP delta + production delta + win/loss)

Paper backing:
  - Bansal et al. 2018: opponent diversity prevents strategy collapse
  - Parker-Holder et al. 2020: diverse opponents improve sample efficiency
  - Ng et al. 1999: potential-based reward shaping with production signal
"""

import os
import math
import random
import glob
import re

os.environ["WANDB_DISABLE_SYMLINKS"] = "True"
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from reward_functions import production_aware_rewards, mask_fn
from self_play_pool import DiverseVFPOpponent


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


def make_cosine_lr_schedule(lr_start, lr_end, total_timesteps, offset):
    remaining = total_timesteps - offset

    def schedule(progress_remaining):
        steps_in_this_run = (1.0 - progress_remaining) * remaining
        abs_step = offset + steps_in_this_run
        t = abs_step / total_timesteps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        return lr_end + (lr_start - lr_end) * cosine_factor

    return schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log LR and training progress to WandB."""

    def __init__(self, total_timesteps, offset, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_timesteps = total_timesteps
        self.offset = offset

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                abs_step = self.offset + self.num_timesteps
                log_dict = {"train/absolute_step": abs_step}

                lr = self.model.lr_schedule(self.model._current_progress_remaining)
                log_dict["train/learning_rate_actual"] = lr

                wandb.log(log_dict, step=abs_step)
            except Exception:
                pass
        return True


def main():
    total_timesteps = 50_000_000

    # SGDR warm restart: 2e-5 -> 5e-6 (conservative for new opponent mix)
    lr_start = 2e-5
    lr_end   = 5e-6

    gamma      = 0.997
    gae_lambda = 0.95
    ent_coef   = 0.03
    n_envs     = 32
    n_steps    = 512
    batch_size = 1024
    n_epochs   = 4
    max_grad_norm = 0.5
    seed       = 42

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"

    reward_function = partial(production_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = production_aware_rewards.__name__
    representation = "mixed"

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    experiment_name = "ppo_v37-diverse-vfp"

    config = {
        "version": "v37_diverse_vfp",
        "approach": "Diverse VFP opponents + production-aware reward from v36",
        "base_model": "model_v36.zip (68.8% vs VFP)",
        "total_timesteps": total_timesteps,
        "lr_schedule": f"cosine restart {lr_start} -> {lr_end} over {total_timesteps/1e6:.0f}M",
        "reward_function": reward_function.__name__,
        "opponent_pool": "5 VFP variants (default, expansion, devcard, contender, noisy)",
        "ent_coef": ent_coef,
        "gamma": gamma,
        "n_envs": n_envs,
        "batch_size": batch_size,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v37")

    def print_name():
        print(experiment_name)
    atexit.register(print_name)

    def make_env(rank, seed=0):
        def _init():
            opponent = DiverseVFPOpponent(Color.RED)
            env = gym.make(
                env_name,
                config={
                    "map_type": map_type,
                    "vps_to_win": vps_to_win,
                    "enemies": [opponent],
                    "reward_function": reward_function,
                    "representation": representation,
                    "normalized": True,
                },
            )
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(script_dir, "model_v37")
    v36_path       = os.path.join(script_dir, "model_v36.zip")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V37: diverse VFP from V36)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    def _apply_hparams(m, lr_sched):
        m.gamma         = gamma
        m.gae_lambda    = gae_lambda
        m.ent_coef      = ent_coef
        m.learning_rate = lr_sched
        m.n_steps       = n_steps
        m.batch_size    = batch_size
        m.n_epochs      = n_epochs
        m.max_grad_norm = max_grad_norm
        m.tensorboard_log = "./logs/mppo_tensorboard/v37"
        m._setup_lr_schedule()
        m.rollout_buffer = MaskableDictRolloutBuffer(
            n_steps,
            m.observation_space,
            m.action_space,
            device=m.device,
            gamma=m.gamma,
            gae_lambda=m.gae_lambda,
            n_envs=n_envs,
        )

    if latest_checkpoint:
        print(f"\n  Found v37 checkpoint at {checkpoint_timesteps:,} steps")
        try:
            lr_schedule = make_cosine_lr_schedule(
                lr_start, lr_end, total_timesteps, checkpoint_timesteps
            )
            model = MaskablePPO.load(
                latest_checkpoint, env, device=device,
                custom_objects={"features_extractor_class": CustomCNN},
            )
            _apply_hparams(model, lr_schedule)
            timesteps_completed = checkpoint_timesteps
            model_loaded = True
            print(f"  Resuming from {timesteps_completed:,} steps")
        except Exception as e:
            print(f"  Failed: {e}")

    if not model_loaded and os.path.exists(v36_path):
        print(f"\n  Loading v36 model from {v36_path}")
        lr_schedule = make_cosine_lr_schedule(lr_start, lr_end, total_timesteps, 0)
        model = MaskablePPO.load(
            v36_path, env, device=device,
            custom_objects={"features_extractor_class": CustomCNN},
        )
        _apply_hparams(model, lr_schedule)
        model_loaded = True
        print(f"  V36 model loaded — diverse VFP training")

    if not model_loaded:
        print(f"\n  ERROR: model_v36.zip not found at {v36_path}")
        return

    remaining_timesteps = total_timesteps - timesteps_completed

    print(f"\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V37: diverse VFP)")
    print("=" * 70)
    print(f"Total timesteps:      {total_timesteps:>15,}")
    print(f"Already completed:    {timesteps_completed:>15,}")
    print(f"Remaining:            {remaining_timesteps:>15,}")
    print(f"\nLR schedule: cosine {lr_start} -> {lr_end} (warm restart)")
    print(f"Reward: {reward_function.__name__}")
    print(f"  VP delta +0.10, Production delta +0.02, Win +1.0, Loss -1.0")
    print(f"\nOpponent pool: 5 VFP variants (sampled per episode)")
    print(f"  default | expansion | devcard | contender | noisy(eps=0.15)")
    print(f"\nEnvs / Steps / Batch: {n_envs} / {n_steps} / {batch_size}")
    print(f"Gamma: {gamma}  |  Ent coef: {ent_coef}")
    print("=" * 70 + "\n")

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
    print("STARTING V37 DIVERSE VFP TRAINING")
    print("=" * 70)
    print(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run:  https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("=" * 70 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback, tb_log_name="v37")
    model.save(model_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED (V37 — diverse VFP)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
