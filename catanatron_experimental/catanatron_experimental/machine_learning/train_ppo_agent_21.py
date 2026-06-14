"""
V21: Expansion-Aware Fresh Run (30M steps)

Key differences from v19/v15:
- Fresh model from scratch (no checkpoint loading)
- New reward: expansion_aware_rewards
    - Brick+Wood production weighted 2× vs ore/wheat/sheep
    - Targets v19's failure mode: 2.07 dev-VP cards, only 1.00 extra settlements
- Expert bonus: 0.2 decaying to 0.05 at 15M steps (same schedule as v15)
    - Higher initial bonus guides the agent toward balanced play faster
    - Floor at 0.05 prevents full graduation (lesson from v17)
- LR: 1e-4 → 1e-5 (full fresh range; v19 was resumed at 5e-5)
- 30M total steps (same as v19 duration but fresh start)

Hypothesis: v19 converged to a pure OWS/dev-card strategy because:
  1. production_aware_rewards weights all production equally → OWS is cheaper/faster
  2. It started from v15 which was already OWS-biased
Starting fresh with doubled brick/wood reward should discover a more balanced
strategy with more settlements (board presence) and more balanced VP sources.
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
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from reward_functions import expansion_aware_rewards, mask_fn
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
    """Log ValueWeightedWrapper statistics to WandB."""

    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                if infos:
                    log_dict = {}
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
                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass
        return True


def main():
    # ===== V21: Expansion-Aware Fresh Run =====
    total_timesteps = 30_000_000

    # Architecture — same as v15 (what works, don't change it)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    # LR: full fresh range (v19 was resumed at 5e-5; this gets the full 1e-4 start)
    initial_lr = 1e-4
    final_lr   = 1e-5

    ent_coef = 0.02

    # Expert bonus: 0.2 → 0.05 at 15M steps (same v15 schedule, keeps floor)
    bonus_scale       = 0.2
    decay_start_step  = 15_000_000
    decay_rate        = 0.99999
    min_scale         = 0.05

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"
    enemies      = [ValueFunctionPlayer(Color.RED)]

    reward_function = partial(expansion_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = expansion_aware_rewards.__name__
    representation = "mixed"

    gamma        = 0.99
    gae_lambda   = 0.95
    normalized   = False
    seed         = 42

    n_envs      = 16
    n_steps     = 512
    batch_size  = 512
    n_epochs    = 4
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

    iters = round(math.log(total_timesteps, 10))
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = (
        f"ppo_v21-ExpansionAware-{iters}-{batch_size}-{gamma}"
        f"-{enemy_desc}-{reward_function.__name__}-{representation}"
        f"-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    )
    print(experiment_name)

    config = {
        "version": "v21_expansion_aware_fresh",
        "approach": "expansion_aware_value_imitation",
        "total_timesteps": total_timesteps,
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
        "activation_fn": activation_fn.__name__,
        "reward_function": reward_function.__name__,
        "reward_notes": "brick+wood prod 2x vs ore/wheat/sheep (0.04 vs 0.02)",
        "bonus_scale": bonus_scale,
        "decay_start_step": decay_start_step,
        "decay_rate": decay_rate,
        "min_scale": min_scale,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": [str(e) for e in enemies],
        "fresh_start": True,
        "motivation": "v19 had 2.07 dev-VP cards, only 1.00 extra settlements; fix via 2x brick/wood",
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v21")

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

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(script_dir, "model_v21")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V21: EXPANSION-AWARE FRESH RUN)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    # Priority 1: Resume interrupted run from checkpoint
    if latest_checkpoint:
        print(f"\n  Found checkpoint at {checkpoint_timesteps:,} steps")
        print(f"  Path: {latest_checkpoint}")
        try:
            model = MaskablePPO.load(
                latest_checkpoint, env,
                device=device,
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
            print(f"  Resuming from {timesteps_completed:,} steps")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
    else:
        print("\n  No checkpoint found — starting fresh (as intended)")

    # Priority 2: Fresh model from scratch
    if not model_loaded:
        print(f"\n  Creating fresh model...")
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

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V21: EXPANSION-AWARE FRESH RUN)")
    print("=" * 70)
    print(f"Total timesteps:         {total_timesteps:>15,}")
    print(f"Already completed:       {timesteps_completed:>15,}")
    print(f"Remaining:               {remaining_timesteps:>15,}")
    print(f"\nReward: {reward_function.__name__}")
    print(f"  VP delta:              +0.10 per VP")
    print(f"  OWS prod delta:        +0.02 per point  (ore/wheat/sheep)")
    print(f"  BW  prod delta:        +0.04 per point  (brick/wood — 2x)")
    print(f"  Win/Loss:              +1.0 / -1.0")
    print(f"\nExpert bonus:            {bonus_scale} → {min_scale} at {decay_start_step:,} steps")
    print(f"Per-env decay start:     {per_env_decay_start:,} steps")
    print(f"Learning rate:           {initial_lr:.1e} → {final_lr:.1e}")
    print(f"Entropy coeff:           {ent_coef}")
    print(f"Envs / Steps / Batch:    {n_envs} / {n_steps} / {batch_size}")
    print(f"Fresh start:             YES (no v19/v15 checkpoint)")
    print("=" * 70 + "\n")

    if remaining_timesteps <= 0:
        print("Training already completed!")
        model.save(model_path)
        run.finish()
        return

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    value_callback = ValueScoreLoggingCallback(log_freq=10_000)
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
    print("TRAINING COMPLETED (V21)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
