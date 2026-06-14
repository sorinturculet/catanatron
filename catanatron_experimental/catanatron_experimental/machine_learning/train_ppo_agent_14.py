"""
V14: Hierarchical Strategy-Conditioned PPO

Agent learns to adapt strategy based on starting position:
- City Engine (ore + wheat) → Cities
- Expansion (wood + brick) → Settlements + Longest Road
- Dev Rush (sheep + wheat + ore) → Largest Army + VP cards

Strategy weights computed at episode start, added to observation,
and used to weight strategy-specific reward bonuses.
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
from reward_functions import dense_vp_rewards, mask_fn
from strategy_wrapper import StrategyWeightedWrapper


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


class StrategyLoggingCallback(BaseCallback):
    """Callback to log strategy metrics to WandB."""

    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                if infos:
                    # Strategy weights
                    city_weights = [
                        info.get('strategy_city_weight', 0)
                        for info in infos
                        if 'strategy_city_weight' in info
                    ]
                    expansion_weights = [
                        info.get('strategy_expansion_weight', 0)
                        for info in infos
                        if 'strategy_expansion_weight' in info
                    ]
                    dev_weights = [
                        info.get('strategy_dev_weight', 0)
                        for info in infos
                        if 'strategy_dev_weight' in info
                    ]

                    # Strategy rewards
                    city_rewards = [
                        info.get('strategy_city_reward', 0)
                        for info in infos
                        if 'strategy_city_reward' in info
                    ]
                    expansion_rewards = [
                        info.get('strategy_expansion_reward', 0)
                        for info in infos
                        if 'strategy_expansion_reward' in info
                    ]
                    dev_rewards = [
                        info.get('strategy_dev_reward', 0)
                        for info in infos
                        if 'strategy_dev_reward' in info
                    ]

                    # Strategy bonus
                    bonuses = [
                        info.get('strategy_bonus', 0)
                        for info in infos
                        if 'strategy_bonus' in info
                    ]

                    log_dict = {}
                    if city_weights:
                        log_dict['strategy/city_weight'] = np.mean(city_weights)
                    if expansion_weights:
                        log_dict['strategy/expansion_weight'] = np.mean(expansion_weights)
                    if dev_weights:
                        log_dict['strategy/dev_weight'] = np.mean(dev_weights)
                    if city_rewards:
                        log_dict['strategy/city_reward'] = np.mean(city_rewards)
                    if expansion_rewards:
                        log_dict['strategy/expansion_reward'] = np.mean(expansion_rewards)
                    if dev_rewards:
                        log_dict['strategy/dev_reward'] = np.mean(dev_rewards)
                    if bonuses:
                        log_dict['strategy/bonus'] = np.mean(bonuses)

                    if log_dict:
                        wandb.log(log_dict, step=self.num_timesteps)
            except Exception:
                pass

        return True


def main():
    # ===== V14: Hierarchical Strategy-Conditioned PPO =====
    total_timesteps = 10_000_000

    # Network architecture (same as v13)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[2048, 1024, 512, 256],
            pi=[2048, 1024, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU

    # Learning rate (same as v13)
    initial_lr = 3e-4
    final_lr = 3e-5

    # Entropy coefficient (same as v13)
    ent_coef = 0.02

    # Strategy wrapper parameters
    strategy_bonus_scale = 0.3  # Multiplier for strategy-specific rewards
    recompute_weights_every_n = 0  # Only compute at episode start

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"

    # Opponent
    enemies = [ValueFunctionPlayer(Color.RED)]

    # Reward function (same as v13)
    reward_function = partial(dense_vp_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = dense_vp_rewards.__name__
    representation = "mixed"

    # Core PPO parameters (same as v13)
    gamma = 0.99
    gae_lambda = 0.95
    normalized = False
    seed = 42

    # Parallelization (same as v13)
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
    experiment_name = f"ppo_v14-HierarchicalStrategy-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    print(experiment_name)

    # WandB config
    config = {
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
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
        "version": "v14_hierarchical_strategy",
        "approach": "strategy_conditioned_ppo",
        # Strategy wrapper params
        "strategy_bonus_scale": strategy_bonus_scale,
        "recompute_weights_every_n": recompute_weights_every_n,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

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
            env = StrategyWeightedWrapper(
                env,
                player_color=Color.BLUE,
                strategy_bonus_scale=strategy_bonus_scale,
                recompute_weights_every_n=recompute_weights_every_n,
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
    model_path = os.path.join(script_dir, "model_v14")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 80)
    print("MODEL LOADING (V14: HIERARCHICAL STRATEGY)")
    print("=" * 80)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    # Priority 1: Resume from checkpoint
    if latest_checkpoint:
        print(f"\n✓ Found checkpoint at {checkpoint_timesteps:,} timesteps")
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
            print(f"✓ Successfully loaded checkpoint!")
            print(f"  Resuming training from timestep {timesteps_completed:,}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
    else:
        print(f"\n✗ No checkpoint found")

    # Priority 2: Fresh model
    if not model_loaded:
        print(f"\nCreating new model from scratch...")
        print(f"  Architecture: {net_arch}")
        print(f"  CNN Architecture: {cnn_arch}")
        print(f"  Learning rate: {initial_lr} → {final_lr}")
        print(f"  Strategy bonus scale: {strategy_bonus_scale}")
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
    print("TRAINING CONFIGURATION (V14: HIERARCHICAL STRATEGY)")
    print("=" * 80)
    print(f"Approach:                Hierarchical Strategy-Conditioned PPO")
    print(f"Total timesteps target:  {total_timesteps:>15,}")
    print(f"Already completed:       {timesteps_completed:>15,}")
    print(f"Remaining timesteps:     {remaining_timesteps:>15,}")
    print(f"Progress:                {(timesteps_completed / total_timesteps * 100):>14.1f}%")
    print(f"\nBase reward:             {reward_function.__name__:>15}")
    print(f"  - VP gain:             +0.1 per VP")
    print(f"  - VP loss:             -0.1 per VP")
    print(f"  - Win bonus:           +1.0")
    print(f"  - Loss penalty:        -1.0")
    print(f"\nStrategy Conditioning:")
    print(f"  - Bonus scale:         {strategy_bonus_scale:>15}")
    print(f"  - Recompute every:     {'episode start only' if recompute_weights_every_n == 0 else f'{recompute_weights_every_n} steps':>15}")
    print(f"  - City reward:         +0.5 per city")
    print(f"  - Expansion reward:    +0.3/settle, +0.05/road, +0.5 longest")
    print(f"  - Dev reward:          +0.15/knight, +0.5 largest army")
    print(f"\nParallel environments:   {n_envs:>15,}")
    print(f"Steps per rollout:       {n_steps:>15,}")
    print(f"Batch size:              {batch_size:>15,}")
    print(f"Epochs per update:       {n_epochs:>15,}")
    print(f"Checkpoint frequency:    {500_000:>15,} steps")
    print(f"\nLearning rate:           {initial_lr:>15.6f} → {final_lr:.6f}")
    print(f"Entropy coefficient:     {ent_coef:>15.4f}")
    print(f"Gamma (discount):        {gamma:>15.4f}")
    print(f"GAE lambda:              {gae_lambda:>15.4f}")
    print("=" * 80 + "\n")

    if remaining_timesteps <= 0:
        print("✓ Training already completed!")
        model.save(model_path)
        run.finish()
        return

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    strategy_callback = StrategyLoggingCallback(log_freq=10000)
    callback = CallbackList([checkpoint_callback, strategy_callback])

    print("=" * 80)
    print("STARTING TRAINING (V14: HIERARCHICAL STRATEGY)")
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
    print("TRAINING COMPLETED (V14: HIERARCHICAL STRATEGY)")
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
