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
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from reward_functions import dense_vp_rewards, mask_fn

LOAD = False


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
    # ===== Improved Hyperparameters (same as v2/v3) =====
    total_timesteps = 10_000_000
    #total_timesteps = 100_000
    # Simplified network architecture - less prone to overfitting
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[2048, 1024, 512, 256],
            pi=[2048, 1024, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU

    # Improved learning rate - standard PPO range
    initial_lr = 3e-4
    final_lr = 3e-5

    # Higher entropy for better exploration
    ent_coef = 0.02

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"

    # V4: Curriculum learning - will be set dynamically per stage
    enemies = [RandomPlayer(Color.RED)]  # Start with RandomPlayer

    # Use dense VP-based reward function
    reward_function = partial(dense_vp_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = dense_vp_rewards.__name__
    representation = "mixed"

    # Core PPO parameters
    gamma = 0.99
    gae_lambda = 0.95
    normalized = False
    selfplay = False
    seed = 42

    # Parallelization
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

    # ===== V4B: MIXED OPPONENT SETUP =====
    # Mixed opponent pool with weighted sampling (no curriculum)
    opponent_pool = [
        (ValueFunctionPlayer(Color.RED), "ValueFunctionPlayer", 0.50),  # 50%
        (WeightedRandomPlayer(Color.RED), "WeightedRandomPlayer", 0.30),  # 30%
        (VictoryPointPlayer(Color.RED), "VictoryPointPlayer", 0.20),     # 20%
    ]

    # Extract weights for sampling
    opponent_weights = [weight for _, _, weight in opponent_pool]

    # Print mixed opponent schedule
    print("\n" + "="*80)
    print("MIXED OPPONENT TRAINING SETUP")
    print("="*80)
    print("Random opponent sampling per episode:")
    for opponent, name, weight in opponent_pool:
        pct = weight * 100
        print(f"  {name}: {pct:.0f}% of episodes")
    print("="*80 + "\n")

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

    experiment_name = f"ppo_v4b_mixed-{selfplay}-{normalized}-{iters}-{batch_size}-{gamma}-MIXED_OPPONENTS-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
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
        "version": "v4b_mixed_opponents",
        "opponent_pool": [name for _, name, _ in opponent_pool],
        "opponent_weights": opponent_weights,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    # Define the environment creation function with random opponent sampling
    def make_env(rank, seed=0):
        def _init():
            # Randomly sample opponent based on weights
            opponent_idx = random.choices(range(len(opponent_pool)), weights=opponent_weights, k=1)[0]
            selected_opponent, opponent_name, _ = opponent_pool[opponent_idx]

            env = gym.make(
                env_name,
                config={
                    "map_type": map_type,
                    "vps_to_win": vps_to_win,
                    "enemies": [selected_opponent],
                    "reward_function": reward_function,
                    "representation": representation,
                    "normalized": True,
                },
            )
            env = ActionMasker(env, mask_fn)
            return env

        return _init

    # Create the vectorized environment with mixed opponent sampling
    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    # Print the observation space to verify its type
    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_v4_v2")
    checkpoint_dir = "./logs/"

    print("\n" + "="*80)
    print("MODEL LOADING")
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

    # Priority 2: Load from model_v4_curriculum (if checkpoint failed)
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
                print(f"✓ Successfully loaded base model: model_v4_curriculum")
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
        print(f"  Reward function: {reward_function.__name__} (DENSE VP-BASED)")
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

    # Calculate remaining timesteps
    remaining_timesteps = total_timesteps - timesteps_completed

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Total timesteps target:  {total_timesteps:>15,}")
    print(f"Already completed:       {timesteps_completed:>15,}")
    print(f"Remaining timesteps:     {remaining_timesteps:>15,}")
    print(f"Progress:                {(timesteps_completed/total_timesteps*100):>14.1f}%")
    print(f"\nReward function:         {reward_function.__name__:>15}")
    print(f"  - VP gain:             +0.1 per VP")
    print(f"  - VP loss:             -0.1 per VP")
    print(f"  - Win bonus:           +1.0")
    print(f"  - Loss penalty:        -1.0")
    print(f"\nParallel environments:   {n_envs:>15,}")
    print(f"Steps per rollout:       {n_steps:>15,}")
    print(f"Batch size:              {batch_size:>15,}")
    print(f"Epochs per update:       {n_epochs:>15,}")
    print(f"Checkpoint frequency:    {50_000:>15,} steps")
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

    # Save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = checkpoint_callback  # No curriculum callback needed for mixed opponent training

    print("="*80)
    print("STARTING MIXED OPPONENT TRAINING")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run: https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print(f"LR Schedule: {initial_lr:.6f} → {final_lr:.6f} (smooth decay)")
    print("="*80 + "\n")

    # Train for all remaining timesteps in ONE call - LR schedule will work correctly
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback,
        reset_num_timesteps=(timesteps_completed == 0)  # Only reset if starting fresh
    )

    # Save final model
    model.save(model_path)

    # Training completion summary
    elapsed_time = time.time() - start_time
    timesteps_per_second = remaining_timesteps / elapsed_time
    hours = elapsed_time / 3600

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total training time:     {elapsed_time:>15.2f} seconds ({hours:.2f} hours)")
    print(f"Timesteps trained:       {remaining_timesteps:>15,}")
    print(f"Throughput:              {timesteps_per_second:>15.2f} timesteps/second")
    print(f"Final model saved:       {model_path}")
    print(f"Checkpoints saved in:    {checkpoint_dir}")
    print("="*80 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
