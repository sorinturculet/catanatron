"""
V24: Fresh 35M — Heuristic Initial Placement + Road/Settlement Bonus

Changes from v23:
  1. InitialPlacementWrapper: crafted heuristic overrides initial settlement+road
     placement based on production value, resource diversity, port proximity,
     and complementarity between 1st/2nd settlement.
  2. road_settlement_aware_rewards: adds +0.04 per road built, bumps settlement
     bonus to +0.10 (was +0.08 in v23).
  3. Same 4-stage LR schedule as v23 (proven).
  4. Fresh start, 35M steps.

Motivation: v23 (19.0%) barely builds roads (AVG ROAD 0.02) and only places
~1 new settlement per game. The heuristic ensures strong initial positions,
and road bonus incentivizes expansion chains toward new settlement spots.
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
from reward_functions import road_settlement_aware_rewards, mask_fn
from value_weighted_wrapper import ValueWeightedWrapper
from initial_placement_wrapper import InitialPlacementWrapper


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


def four_stage_lr_schedule(total_timesteps):
    """
    4-stage piecewise LR (same as v23).

    Stage 1 ( 0–10M): 3e-4 → 3e-5
    Stage 2 (10–20M): 1e-4 → 1e-5  (reset)
    Stage 3 (20–30M): 5e-5 → 1e-5  (reset)
    Stage 4 (30–35M): 3e-5 → 1e-5  (fine-tune)
    """
    p1 = 25 / 35
    p2 = 15 / 35
    p3 =  5 / 35

    def lr_schedule(progress_remaining):
        if progress_remaining > p1:
            frac = (progress_remaining - p1) / (1.0 - p1)
            return 3e-5 + (3e-4 - 3e-5) * frac
        elif progress_remaining > p2:
            frac = (progress_remaining - p2) / (p1 - p2)
            return 1e-5 + (1e-4 - 1e-5) * frac
        elif progress_remaining > p3:
            frac = (progress_remaining - p3) / (p2 - p3)
            return 1e-5 + (5e-5 - 1e-5) * frac
        else:
            frac = progress_remaining / p3
            return 1e-5 + (3e-5 - 1e-5) * frac

    return lr_schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper + LR stage to WandB."""

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

                    if self.num_timesteps < 10_000_000:
                        log_dict["train/lr_stage"] = 1
                    elif self.num_timesteps < 20_000_000:
                        log_dict["train/lr_stage"] = 2
                    elif self.num_timesteps < 30_000_000:
                        log_dict["train/lr_stage"] = 3
                    else:
                        log_dict["train/lr_stage"] = 4

                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass
        return True


def main():
    # ===== V24: Fresh 35M with Heuristic Placement + Road/Settlement Bonus =====
    total_timesteps = 35_000_000

    # Architecture — same as v15/v19/v22/v23 (proven, don't change)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    ent_coef = 0.02

    # Expert bonus: 0.2 → 0.05 floor at 5M (same schedule as v15/v22/v23)
    bonus_scale      = 0.2
    decay_start_step = 5_000_000
    decay_rate       = 0.99999
    min_scale        = 0.05

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"
    enemies      = [ValueFunctionPlayer(Color.RED)]

    reward_function = partial(road_settlement_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = road_settlement_aware_rewards.__name__
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

    lr_schedule = four_stage_lr_schedule(total_timesteps)

    iters = round(math.log(total_timesteps, 10))
    enemy_desc = "".join(e.__class__.__name__ for e in enemies)
    experiment_name = (
        f"ppo_v24-HeuristicPlacement-{iters}-{batch_size}-{gamma}"
        f"-{enemy_desc}-{reward_function.__name__}-{representation}"
        f"-4stageLR-{vps_to_win}vp-{map_type}map"
    )
    print(experiment_name)

    config = {
        "version": "v24_heuristic_placement",
        "approach": "fresh_4stage_lr_heuristic_placement_road_settle",
        "total_timesteps": total_timesteps,
        "lr_stage1": "3e-4 → 3e-5  (0–10M)",
        "lr_stage2": "1e-4 → 1e-5  (10–20M, reset)",
        "lr_stage3": "5e-5 → 1e-5  (20–30M, reset)",
        "lr_stage4": "3e-5 → 1e-5  (30–35M, fine-tune)",
        "reward_function": reward_function.__name__,
        "reward_notes": "VP +0.10, OWS +0.02, BW +0.04, settle +0.10, road +0.04",
        "initial_placement": "heuristic (production + diversity + port + complementarity)",
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
        "fresh_start": True,
        "motivation": "v23 AVG ROAD 0.02, ~1 new settle/game; heuristic placement + road bonus",
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v24")

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
            env = InitialPlacementWrapper(env, player_color=Color.BLUE)
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
    model_path     = os.path.join(script_dir, "model_v24")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V24: HEURISTIC PLACEMENT + ROAD/SETTLE BONUS)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

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
            stage = (1 if timesteps_completed < 10_000_000 else
                     2 if timesteps_completed < 20_000_000 else
                     3 if timesteps_completed < 30_000_000 else 4)
            print(f"  Resuming from {timesteps_completed:,} steps (Stage {stage})")
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
    print("TRAINING CONFIGURATION (V24: HEURISTIC PLACEMENT + ROAD/SETTLE)")
    print("=" * 70)
    print(f"Total timesteps:      {total_timesteps:>15,}  (FRESH — no prior checkpoint)")
    print(f"Already completed:    {timesteps_completed:>15,}")
    print(f"Remaining:            {remaining_timesteps:>15,}")
    print(f"\nLR Schedule (4-stage piecewise):")
    print(f"  Stage 1   0–10M:    3e-4 → 3e-5  (mirrors v15)")
    print(f"  Stage 2  10–20M:    1e-4 → 1e-5  (mirrors v18 reset)")
    print(f"  Stage 3  20–30M:    5e-5 → 1e-5  (mirrors v19 reset)")
    print(f"  Stage 4  30–35M:    3e-5 → 1e-5  (fine-tuning)")
    print(f"\nReward: {reward_function.__name__}")
    print(f"  VP delta:           +0.10 per VP")
    print(f"  OWS prod:           +0.02/pt  (ore/wheat/sheep)")
    print(f"  BW  prod:           +0.04/pt  (brick/wood — 2×)")
    print(f"  Settlement placed:  +0.10     (bumped from v23's 0.08)")
    print(f"  Road built:         +0.04     (NEW)")
    print(f"  Win/Loss:           +1.0 / -1.0")
    print(f"\nInitial Placement:    HEURISTIC (production + diversity + port + complementarity)")
    print(f"Expert bonus:         {bonus_scale} → {min_scale} floor at {decay_start_step/1e6:.0f}M steps")
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
    print("TRAINING COMPLETED (V24)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
