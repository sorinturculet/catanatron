"""
Train PPO Agent v11: Production-Aware Features + Reward Shaping

Key changes from previous versions:
- Enables production features in observation space (were commented out)
- Adds buildable_node_values() - shows production value for each valid settlement spot
- Adds resource_strategy_features() - shows city potential, expansion potential, etc.
- Uses production_aware_rewards() - rewards VP delta + production delta

The agent can now SEE where good spots are, not just get rewarded after building.
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


def find_latest_checkpoint(checkpoint_dir, experiment_prefix):
    """
    Find the latest checkpoint file for a given experiment.
    Returns (checkpoint_path, num_timesteps) or (None, 0) if no checkpoint found.
    """
    pattern = os.path.join(checkpoint_dir, f"{experiment_prefix}_*_steps.zip")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None, 0

    # Extract timesteps from filenames and find the latest
    latest_checkpoint = None
    max_timesteps = 0

    for checkpoint_file in checkpoint_files:
        # Extract timesteps from filename like "experiment_name_50000_steps.zip"
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


def main():
    # ===== V11: Production-Aware PPO Training =====
    # Key insight: v7 used production_aware_rewards but agent couldn't SEE
    # production values (they were commented out in observation space).
    # v11 enables production features + adds buildable node values.

    # Train for 10M timesteps (same as v8 total)
    total_timesteps = 10_000_000
    #total_timesteps = 100_000
    # Same network architecture as v8 (proven best)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[2048, 1024, 512, 256],
            pi=[2048, 1024, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU

    # Learning rate (same as v8)
    initial_lr = 3e-4
    final_lr = 3e-5

    # Entropy coefficient (same as v8)
    ent_coef = 0.02

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]

    # Use production_aware_rewards (VP delta + production delta)
    reward_function = partial(production_aware_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = production_aware_rewards.__name__
    representation = "mixed"

    # Core PPO parameters (same as v8)
    gamma = 0.99
    gae_lambda = 0.95
    normalized = False
    selfplay = False
    seed = 42

    # Parallelization (same as v8)
    n_envs = 16
    n_steps = 512
    batch_size = 512
    n_epochs = 4
    max_grad_norm = 0.5

    # Validate batch configuration
    assert (
        n_envs * n_steps
    ) % batch_size == 0, "batch_size must divide n_envs * n_steps"

    start_time = time.time()

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    # Create learning rate schedule
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
    experiment_name = f"ppo_v11-{selfplay}-{normalized}-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
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
        "selfplay": selfplay,
        "experiment_name": experiment_name,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "version": "v11_production_aware",
        "new_features": [
            "build_production_features(True)",
            "build_production_features(False)",
            "buildable_node_values",
            "resource_strategy_features",
        ],
        "key_insight": "v7 reward worked but agent couldn't SEE production - now it can",
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Define the environment creation function
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

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    # Print the observation space to verify its type
    print("Observation Space:", env.observation_space)

    # CRITICAL: Verify feature count is correct for v11
    if hasattr(env.observation_space, 'spaces') and 'numeric' in env.observation_space.spaces:
        numeric_shape = env.observation_space.spaces['numeric'].shape[0]
        print(f"\n*** NUMERIC FEATURE COUNT: {numeric_shape} ***")
        if numeric_shape < 100:
            print("!!! WARNING: Expected 137 features for v11, got only", numeric_shape)
            print("!!! features.py may not have v11 changes (buildable_node_values, resource_strategy_features)")
            print("!!! Run: grep -c 'buildable_node_values' catanatron_gym/features.py (should be 2)")
        else:
            print("✓ Feature count looks correct for v11 (expected ~137)")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_v11")
    checkpoint_dir = "./logs/"

    print("\n" + "="*80)
    print("MODEL LOADING (V11: PRODUCTION-AWARE)")
    print("="*80)
    print("Key v11 changes:")
    print("  - Production features ENABLED (were commented out)")
    print("  - buildable_node_values: shows production value per buildable spot")
    print("  - resource_strategy_features: city/expansion potential")
    print("  - production_aware_rewards: VP delta + production delta")
    print("="*80)

    # Try to find and load the latest checkpoint
    print(f"Searching for checkpoints in: {checkpoint_dir}")
    print(f"Experiment name: {experiment_name}")
    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    # Priority 1: Load from checkpoint (if exists)
    if latest_checkpoint:
        print(f"\n✓ Found checkpoint at {checkpoint_timesteps:,} timesteps")
        print(f"  Path: {latest_checkpoint}")
        try:
            model = MaskablePPO.load(latest_checkpoint, env, device=device)
            # Override the training configuration
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
            print("  Will try alternative loading methods...")
    else:
        print(f"\n✗ No checkpoint found matching pattern: {experiment_name}_*_steps.zip")

    # Priority 2: Load from model_v11 (if checkpoint failed)
    if not model_loaded:
        print(f"\nTrying to load base model: {model_path}")
        if os.path.exists(model_path + ".zip"):
            try:
                model = MaskablePPO.load(model_path, env, device=device)
                # Override the training configuration
                model.gamma = gamma
                model.gae_lambda = gae_lambda
                model.ent_coef = ent_coef
                model.learning_rate = lr_schedule
                model.batch_size = batch_size
                model.n_epochs = n_epochs
                model.max_grad_norm = max_grad_norm
                model._setup_lr_schedule()
                model_loaded = True
                print(f"✓ Successfully loaded base model: model_v11")
                print(f"  Starting training from timestep 0")
            except Exception as e:
                print(f"✗ Failed to load base model: {e}")
        else:
            print(f"✗ Base model not found at: {model_path}.zip")

    # Priority 3: Create new model (if nothing else worked)
    if not model_loaded:
        print(f"\nCreating new model from scratch...")
        print(f"  Architecture: {net_arch}")
        print(f"  CNN Architecture: {cnn_arch}")
        print(f"  Learning rate: {initial_lr} → {final_lr}")
        print(f"  Reward function: {reward_function.__name__} (VP + PRODUCTION DELTA)")
        policy_kwargs: Any = dict(activation_fn=activation_fn, net_arch=net_arch[0])
        if representation == "mixed":
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(
                cnn_arch=cnn_arch, features_dim=512  # Adjust as needed
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

    # Calculate remaining timesteps
    remaining_timesteps = total_timesteps - timesteps_completed

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION (V11: PRODUCTION-AWARE)")
    print("="*80)
    print(f"Version:                 v11 (production-aware features)")
    print(f"Total timesteps target:  {total_timesteps:>15,}")
    print(f"Already completed:       {timesteps_completed:>15,}")
    print(f"Remaining timesteps:     {remaining_timesteps:>15,}")
    print(f"Progress:                {(timesteps_completed/total_timesteps*100):>14.1f}%")
    print(f"\nReward function:         {reward_function.__name__:>15}")
    print(f"  - VP delta:            +0.1 per VP")
    print(f"  - Production delta:    +0.02 per production point")
    print(f"  - Win bonus:           +1.0")
    print(f"  - Loss penalty:        -1.0")
    print(f"\nNew observation features:")
    print(f"  - EFFECTIVE_P*_<RESOURCE>_PRODUCTION (per resource)")
    print(f"  - TOTAL_P*_<RESOURCE>_PRODUCTION (ignores robber)")
    print(f"  - NODE*_BUILDABLE_VALUE (production per buildable spot)")
    print(f"  - P*_CITY_POTENTIAL, P*_EXPANSION_POTENTIAL, etc.")
    print(f"\nParallel environments:   {n_envs:>15,}")
    print(f"Steps per rollout:       {n_steps:>15,}")
    print(f"Batch size:              {batch_size:>15,}")
    print(f"Epochs per update:       {n_epochs:>15,}")
    print(f"Checkpoint frequency:    {500_000:>15,} steps")
    print(f"\nLearning rate:           {initial_lr:>15.6f} → {final_lr:.6f}")
    print(f"Entropy coefficient:     {ent_coef:>15.4f}")
    print(f"Gamma (discount):        {gamma:>15.4f}")
    print(f"GAE lambda:              {gae_lambda:>15.4f}")
    print("="*80 + "\n")

    if remaining_timesteps <= 0:
        print("✓ Training already completed!")
        model.save(model_path)
        run.finish()
        return

    # Save checkpoints every 500K steps
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = CallbackList([checkpoint_callback])  # , wandb_callback])

    print("="*80)
    print("STARTING TRAINING (V11: PRODUCTION-AWARE)")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run: https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("="*80 + "\n")

    model.learn(total_timesteps=remaining_timesteps, callback=callback)

    # Save final model
    model.save(model_path)

    # Training completion summary
    elapsed_time = time.time() - start_time
    timesteps_per_second = remaining_timesteps / elapsed_time
    hours = elapsed_time / 3600

    print("\n" + "="*80)
    print("TRAINING COMPLETED (V11: PRODUCTION-AWARE)")
    print("="*80)
    print(f"Total training time:     {elapsed_time:>15.2f} seconds ({hours:.2f} hours)")
    print(f"Timesteps trained:       {remaining_timesteps:>15,}")
    print(f"Throughput:              {timesteps_per_second:>15.2f} timesteps/second")
    print(f"Final model saved:       {model_path}")
    print(f"Checkpoints saved in:    {checkpoint_dir}")
    print(f"\n🎯 Run evaluation with: python evaluate_ppo.py --model model_v11")
    print("="*80 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
