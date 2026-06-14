"""
V38: PPO fine-tuning from Behavioral Cloning (AlphaBeta teacher, 100K games).

Identical to train_ppo_agent_34.py except the base BC model is model_bc_ab.zip
(trained on AlphaBeta games) instead of model_bc.zip (trained on VFP games).
"""

import os
import math
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import wandb

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from reward_functions import dense_vp_rewards, mask_fn


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
    """Cosine decay LR schedule with absolute step counting (handles checkpoint resume)."""
    remaining = total_timesteps - offset

    def schedule(progress_remaining):
        steps_in_this_run = (1.0 - progress_remaining) * remaining
        abs_step = offset + steps_in_this_run
        t = abs_step / total_timesteps  # 0.0 → 1.0 over full training
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        return lr_end + (lr_start - lr_end) * cosine_factor

    return schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper metrics + actual LR to WandB."""

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

                lr = self.model.lr_schedule(self.model._current_progress_remaining)
                log_dict["train/learning_rate_actual"] = lr

                frac = abs_step / self.total_timesteps
                if frac < 0.30:
                    log_dict["train/lr_stage"] = 1
                elif frac < 0.60:
                    log_dict["train/lr_stage"] = 2
                else:
                    log_dict["train/lr_stage"] = 3

                wandb.log(log_dict, step=abs_step)
            except Exception:
                pass
        return True


def main():
    total_timesteps = 80_000_000

    # Cosine LR: 5e-5 → 2e-6 over 80M steps (SGDR, Loshchilov & Hutter 2016)
    lr_start = 5e-5
    lr_end   = 2e-6

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
    enemies      = [ValueFunctionPlayer(Color.RED)]

    reward_function = partial(dense_vp_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = dense_vp_rewards.__name__
    representation = "mixed"

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    experiment_name = "ppo_v38-BC-AB-finetune"

    config = {
        "version": "v38_bc_ab_finetune",
        "approach": "Behavioral Cloning from AlphaBeta (100K games) → PPO fine-tune",
        "base_model": "model_bc_ab.zip (BC on 100K AlphaBeta games)",
        "total_timesteps": total_timesteps,
        "lr_schedule": f"cosine {lr_start} → {lr_end} over {total_timesteps/1e6:.0f}M steps",
        "reward_function": reward_function.__name__,
        "expert_guidance": "None — BC pretrained, PPO explores freely",
        "ent_coef": ent_coef,
        "gamma": gamma,
        "n_envs": n_envs,
        "batch_size": batch_size,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v38")

    def print_name():
        print(experiment_name)
    atexit.register(print_name)

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
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(script_dir, "model_v38")
    bc_path        = os.path.join(script_dir, "model_bc_ab.zip")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V38: BC-AlphaBeta → PPO)")
    print("=" * 70)

    # Check for v38 checkpoint first (resume)
    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    if latest_checkpoint:
        print(f"\n  Found v38 checkpoint at {checkpoint_timesteps:,} steps")
        try:
            lr_schedule = make_cosine_lr_schedule(
                lr_start, lr_end, total_timesteps, checkpoint_timesteps
            )
            model = MaskablePPO.load(
                latest_checkpoint, env, device=device,
                custom_objects={"features_extractor_class": CustomCNN},
            )
            model.gamma         = gamma
            model.gae_lambda    = gae_lambda
            model.ent_coef      = ent_coef
            model.learning_rate = lr_schedule
            model.n_steps       = n_steps
            model.batch_size    = batch_size
            model.n_epochs      = n_epochs
            model.max_grad_norm = max_grad_norm
            model.tensorboard_log = "./logs/mppo_tensorboard/v38"
            model._setup_lr_schedule()
            # Recreate rollout buffer with correct dimensions
            model.rollout_buffer = MaskableDictRolloutBuffer(
                n_steps,
                model.observation_space,
                model.action_space,
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=n_envs,
            )
            timesteps_completed = checkpoint_timesteps
            model_loaded = True
            print(f"  Resuming from {timesteps_completed:,} steps")
        except Exception as e:
            print(f"  Failed: {e}")

    if not model_loaded and os.path.exists(bc_path):
        print(f"\n  Loading BC-AB model from {bc_path}")
        lr_schedule = make_cosine_lr_schedule(lr_start, lr_end, total_timesteps, 0)
        model = MaskablePPO.load(
            bc_path, env, device=device,
            custom_objects={"features_extractor_class": CustomCNN},
        )
        model.gamma         = gamma
        model.gae_lambda    = gae_lambda
        model.ent_coef      = ent_coef
        model.learning_rate = lr_schedule
        model.n_steps       = n_steps
        model.batch_size    = batch_size
        model.n_epochs      = n_epochs
        model.max_grad_norm = max_grad_norm
        model.tensorboard_log = "./logs/mppo_tensorboard/v38"
        model._setup_lr_schedule()
        # Recreate rollout buffer with correct n_steps/n_envs
        # (BC model was saved with default n_steps=2048 + 1 env)
        model.rollout_buffer = MaskableDictRolloutBuffer(
            n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=n_envs,
        )
        model_loaded = True
        print(f"  BC-AB model loaded — fine-tuning with PPO")

    if not model_loaded:
        print(f"\n  ERROR: model_bc_ab.zip not found at {bc_path}")
        print(f"  Run collect_and_train_bc_alphabeta.py first!")
        return

    remaining_timesteps = total_timesteps - timesteps_completed

    print(f"\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V38: BC-AlphaBeta → PPO)")
    print("=" * 70)
    print(f"Total timesteps:      {total_timesteps:>15,}")
    print(f"Already completed:    {timesteps_completed:>15,}")
    print(f"Remaining:            {remaining_timesteps:>15,}")
    print(f"\nReward: {reward_function.__name__}")
    print(f"  VP delta +0.10, Win +1.0, Loss -1.0 (minimal — let agent find its own way)")
    print(f"\nExpert guidance: NONE (BC pretrained — PPO explores freely)")
    print(f"Envs / Steps / Batch: {n_envs} / {n_steps} / {batch_size}")
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
    print("STARTING PPO FINE-TUNING")
    print("=" * 70)
    print(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run:  https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("=" * 70 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback, tb_log_name="v38")
    model.save(model_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED (V38 — BC-AlphaBeta → PPO)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
