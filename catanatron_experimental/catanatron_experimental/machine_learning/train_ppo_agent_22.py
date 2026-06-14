"""
V22: Fresh 30M Run with 3-Stage LR Reset Schedule

Key insight from v21 failure: 30M fresh steps without LR resets only reaches ~10%
because the model can't escape early local optima. The v15→v18→v19 progression
worked because each LR reset unlocked new learning after the previous run plateaued.

V22 replicates the proven v15/v18/v19 LR schedule in a single fresh 30M run:
  Stage 1 (0–10M):   3e-4 → 3e-5  (mirrors v15)
  Stage 2 (10–20M):  1e-4 → 1e-5  (mirrors v18 LR reset)
  Stage 3 (20–30M):  5e-5 → 1e-5  (mirrors v19 LR reset)

Other params:
  - expansion_aware_rewards (brick+wood 2× production weight vs v15/v18/v19)
  - Expert bonus: 0.2 → 0.05 floor at 5M steps (same as v15)
  - Same architecture as v15/v19 (CNN [64,128,256,512] + MLP [2048,1024,512,256])
  - Fresh from scratch (no checkpoint loading from previous versions)
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


def three_stage_lr_schedule(total_timesteps):
    """
    Piecewise LR schedule replicating the v15→v18→v19 pattern in one run.

    progress_remaining = 1.0 at step 0, 0.0 at step total_timesteps.

    Stage 1 (steps 0    → T/3):   progress 1.000 → 0.667   LR 3e-4 → 3e-5
    Stage 2 (steps T/3  → 2T/3):  progress 0.667 → 0.333   LR 1e-4 → 1e-5  (reset up)
    Stage 3 (steps 2T/3 → T):     progress 0.333 → 0.000   LR 5e-5 → 1e-5  (reset up)
    """
    p1 = 2 / 3   # progress at end of stage 1 (step 10M)
    p2 = 1 / 3   # progress at end of stage 2 (step 20M)

    def lr_schedule(progress_remaining):
        if progress_remaining > p1:
            # Stage 1: 0–10M
            frac = (progress_remaining - p1) / (1.0 - p1)   # 1.0→0.0
            return 3e-5 + (3e-4 - 3e-5) * frac
        elif progress_remaining > p2:
            # Stage 2: 10–20M  (LR resets to 1e-4)
            frac = (progress_remaining - p2) / (p1 - p2)    # 1.0→0.0
            return 1e-5 + (1e-4 - 1e-5) * frac
        else:
            # Stage 3: 20–30M  (LR resets to 5e-5)
            frac = progress_remaining / p2                   # 1.0→0.0
            return 1e-5 + (5e-5 - 1e-5) * frac

    return lr_schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper + LR statistics to WandB."""

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

                    # Log current LR and which stage we're in
                    current_lr = self.model.lr_schedule(
                        1.0 - self.num_timesteps / self.model._total_timesteps
                    )
                    log_dict["train/learning_rate_actual"] = current_lr
                    if self.num_timesteps < 10_000_000:
                        log_dict["train/lr_stage"] = 1
                    elif self.num_timesteps < 20_000_000:
                        log_dict["train/lr_stage"] = 2
                    else:
                        log_dict["train/lr_stage"] = 3

                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass
        return True


def main():
    # ===== V22: Fresh 30M with 3-Stage LR Reset =====
    total_timesteps = 30_000_000

    # Architecture — same as v15/v19 (proven, don't change)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    ent_coef = 0.02

    # Expert bonus: 0.2 → 0.05 floor at 5M steps (same schedule as v15)
    bonus_scale      = 0.2
    decay_start_step = 5_000_000
    decay_rate       = 0.99999
    min_scale        = 0.05

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"
    enemies      = [ValueFunctionPlayer(Color.RED)]

    reward_function = partial(expansion_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = expansion_aware_rewards.__name__
    representation = "mixed"

    gamma      = 0.99
    gae_lambda = 0.95
    seed       = 42

    n_envs        = 16
    n_steps       = 512
    batch_size    = 512
    n_epochs      = 4
    max_grad_norm = 0.5

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    lr_schedule = three_stage_lr_schedule(total_timesteps)

    iters = round(math.log(total_timesteps, 10))
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = (
        f"ppo_v22-FreshMultiStage-{iters}-{batch_size}-{gamma}"
        f"-{enemy_desc}-{reward_function.__name__}-{representation}"
        f"-3stageLR-{vps_to_win}vp-{map_type}map"
    )
    print(experiment_name)

    config = {
        "version": "v22_fresh_multistage_lr",
        "approach": "fresh_3stage_lr_expansion_rewards",
        "total_timesteps": total_timesteps,
        "lr_stage1": "3e-4 → 3e-5  (0–10M, mirrors v15)",
        "lr_stage2": "1e-4 → 1e-5  (10–20M, mirrors v18 reset)",
        "lr_stage3": "5e-5 → 1e-5  (20–30M, mirrors v19 reset)",
        "reward_function": reward_function.__name__,
        "reward_notes": "brick+wood 2× vs ore/wheat/sheep (0.04 vs 0.02)",
        "bonus_scale": bonus_scale,
        "decay_start_step": decay_start_step,
        "decay_rate": decay_rate,
        "min_scale": min_scale,
        "ent_coef": ent_coef,
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
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
        "fresh_start": True,
        "motivation": "v21 (fresh 30M flat LR) only reached 10.6%; multi-stage LR replicates v15→v18→v19 gains in one run",
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v22")

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
    model_path     = os.path.join(script_dir, "model_v22")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V22: FRESH MULTI-STAGE LR)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    # Resume interrupted run from checkpoint (same experiment only)
    if latest_checkpoint:
        print(f"\n  Found checkpoint at {checkpoint_timesteps:,} steps")
        try:
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
            print(f"  Resuming from {timesteps_completed:,} steps")
            # Report which stage we're resuming in
            if timesteps_completed < 10_000_000:
                print(f"  Resuming in Stage 1 (3e-4→3e-5)")
            elif timesteps_completed < 20_000_000:
                print(f"  Resuming in Stage 2 (1e-4→1e-5)")
            else:
                print(f"  Resuming in Stage 3 (5e-5→1e-5)")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n  No checkpoint — starting fresh (as intended)")

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
    print("TRAINING CONFIGURATION (V22: FRESH MULTI-STAGE LR)")
    print("=" * 70)
    print(f"Total timesteps:     {total_timesteps:>15,}  (FRESH — no v19 base)")
    print(f"Already completed:   {timesteps_completed:>15,}")
    print(f"Remaining:           {remaining_timesteps:>15,}")
    print(f"\nLR Schedule (3-stage piecewise):")
    print(f"  Stage 1  0–10M:    3e-4 → 3e-5  (mirrors v15)")
    print(f"  Stage 2  10–20M:   1e-4 → 1e-5  (mirrors v18 reset)")
    print(f"  Stage 3  20–30M:   5e-5 → 1e-5  (mirrors v19 reset)")
    print(f"\nReward: {reward_function.__name__}")
    print(f"  VP delta:          +0.10 per VP")
    print(f"  OWS prod:          +0.02/pt  (ore/wheat/sheep)")
    print(f"  BW  prod:          +0.04/pt  (brick/wood — 2×)")
    print(f"  Win/Loss:          +1.0 / -1.0")
    print(f"\nExpert bonus:        {bonus_scale} → {min_scale} floor at {decay_start_step/1e6:.0f}M steps")
    print(f"Envs / Steps / Batch: {n_envs} / {n_steps} / {batch_size}")
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
    print("TRAINING COMPLETED (V22)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
