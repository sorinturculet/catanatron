"""
V12: Production Features + Two-Stage Self-Play (Unified)

Combines:
- V11's production features (buildable_node_values, resource_strategy_features)
- V11's production_aware_rewards
- V8B's self-play curriculum (bootstrap then self-play)

All in ONE script - no manual intervention needed.

Stage 1 (5M steps): Train vs ValueFunctionPlayer to learn fundamentals + production
Stage 2 (5M steps): Self-play to push beyond the ValueFunction ceiling
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
from catanatron_experimental.machine_learning.players.ppo import PPOPlayer
from catanatron.models.player import RandomPlayer
from reward_functions import production_aware_rewards, dense_vp_rewards, mask_fn


def find_latest_checkpoint(checkpoint_dir, experiment_prefix):
    """Find the latest checkpoint file for a given experiment."""
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


class SelfPlayOpponentPool:
    """Manages opponent pool for self-play stage."""

    def __init__(self, checkpoint_dir="./logs", pool_size=5, experiment_name="ppo_v12"):
        self.checkpoint_dir = checkpoint_dir
        self.pool_size = pool_size
        self.experiment_name = experiment_name
        self.opponent_checkpoints = []

    def load_existing_checkpoints(self):
        """Load existing checkpoints from directory."""
        pattern = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_*_steps.zip")
        checkpoint_files = sorted(glob.glob(pattern))
        self.opponent_checkpoints = checkpoint_files[-self.pool_size:]

        if self.opponent_checkpoints:
            print(f"   Loaded {len(self.opponent_checkpoints)} existing checkpoints into pool")

    def add_checkpoint(self, checkpoint_path):
        """Add checkpoint to pool."""
        self.opponent_checkpoints.append(checkpoint_path)
        if len(self.opponent_checkpoints) > self.pool_size:
            self.opponent_checkpoints.pop(0)

    def sample_opponent(self, fallback_path=None):
        """Sample opponent with preference for recent checkpoints."""
        if not self.opponent_checkpoints:
            if fallback_path and os.path.exists(fallback_path):
                return PPOPlayer(Color.RED, fallback_path)
            raise ValueError("No checkpoints available for self-play!")

        if len(self.opponent_checkpoints) == 1:
            checkpoint = self.opponent_checkpoints[0]
        else:
            n = len(self.opponent_checkpoints)
            weights = [0.4 / (n - 1) for _ in range(n - 1)] + [0.6]
            checkpoint = random.choices(self.opponent_checkpoints, weights=weights)[0]

        return PPOPlayer(Color.RED, checkpoint)


def create_env(env_name, map_type, vps_to_win, enemies, reward_function, representation, rank, seed=0):
    """Create a single environment instance."""
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


def main():
    # ===== V12: Production Features + Two-Stage Self-Play =====

    # Stage configuration
    stage1_timesteps = 5_000_000  # vs ValueFunctionPlayer
    stage2_timesteps = 5_000_000  # Self-play
    total_timesteps = stage1_timesteps + stage2_timesteps

    # Network architecture (same as v3/v8/v11)
    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU

    # Learning rate
    initial_lr = 3e-4
    final_lr = 3e-5

    # PPO hyperparameters
    ent_coef = 0.02
    gamma = 0.99
    gae_lambda = 0.95
    n_envs = 16
    n_steps = 512
    batch_size = 512
    n_epochs = 4
    max_grad_norm = 0.5
    seed = 42

    # Environment config
    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    representation = "mixed"

    # Reward functions - DIFFERENT for each stage!
    # Stage 1: production_aware_rewards (learn WHERE to build vs expert)
    # Stage 2: dense_vp_rewards (learn HOW to win in self-play)
    reward_function_stage1 = partial(production_aware_rewards, vps_to_win=vps_to_win)
    reward_function_stage1.__name__ = "production_aware_rewards"

    reward_function_stage2 = partial(dense_vp_rewards, vps_to_win=vps_to_win)
    reward_function_stage2.__name__ = "dense_vp_rewards"

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_v12")
    stage1_checkpoint_path = os.path.join(script_dir, "model_v12_stage1")
    checkpoint_dir = "./logs/"

    # Initialize opponent pool for Stage 2
    opponent_pool = SelfPlayOpponentPool(
        checkpoint_dir=checkpoint_dir,
        pool_size=5,
        experiment_name="ppo_v12"
    )

    start_time = time.time()

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    # Create learning rate schedule
    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    # Experiment name
    experiment_name = "ppo_v12-production_selfplay"

    # WandB config
    config = {
        "version": "v12_production_selfplay",
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
        "stage1_timesteps": stage1_timesteps,
        "stage2_timesteps": stage2_timesteps,
        "net_arch": net_arch,
        "cnn_arch": cnn_arch,
        "activation_fn": activation_fn.__name__,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "reward_function_stage1": reward_function_stage1.__name__,
        "reward_function_stage2": reward_function_stage2.__name__,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "key_insight": "V11 production features + V8B self-play curriculum",
        "new_features": [
            "build_production_features(True)",
            "build_production_features(False)",
            "buildable_node_values",
            "resource_strategy_features"
        ],
    }

    run = wandb.init(
        project="catanatron",
        name="v12",
        config=config,
        sync_tensorboard=True,
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("V12: PRODUCTION FEATURES + TWO-STAGE SELF-PLAY")
    print("="*80)
    print(f"Device: {device}")
    print(f"Stage 1: {stage1_timesteps:,} steps vs ValueFunctionPlayer")
    print(f"Stage 2: {stage2_timesteps:,} steps self-play")
    print(f"Total:   {total_timesteps:,} steps")
    print(f"\nKey features:")
    print(f"  - Stage 1: production_aware_rewards (learn WHERE to build)")
    print(f"  - Stage 2: dense_vp_rewards (learn HOW to win)")
    print(f"  - buildable_node_values (WHERE to settle)")
    print(f"  - resource_strategy_features (city/expansion potential)")
    print("="*80 + "\n")

    # ========== STAGE 1: Train vs ValueFunctionPlayer ==========
    print("="*80)
    print("STAGE 1: TRAINING VS VALUEFUNCTIONPLAYER")
    print(f"Reward function: {reward_function_stage1.__name__}")
    print("="*80)

    # Check if Stage 1 is already done
    stage1_done = os.path.exists(stage1_checkpoint_path + ".zip")

    if stage1_done:
        print(f"Stage 1 checkpoint found: {stage1_checkpoint_path}.zip")
        print("Skipping Stage 1, loading checkpoint...")

        # Create dummy env to load model
        enemies_stage1 = [ValueFunctionPlayer(Color.RED)]
        env = SubprocVecEnv([create_env(env_name, map_type, vps_to_win, enemies_stage1,
                                        reward_function_stage1, representation, i, seed)
                            for i in range(n_envs)])
        env = VecMonitor(env)

        model = MaskablePPO.load(stage1_checkpoint_path, env, device=device)
        env.close()
        print("Stage 1 model loaded successfully!")
    else:
        print("Starting Stage 1 training from scratch...")

        # Stage 1 opponent: ValueFunctionPlayer
        enemies_stage1 = [ValueFunctionPlayer(Color.RED)]

        # Create environment
        env = SubprocVecEnv([create_env(env_name, map_type, vps_to_win, enemies_stage1,
                                        reward_function_stage1, representation, i, seed)
                            for i in range(n_envs)])
        env = VecMonitor(env)

        print("Observation Space:", env.observation_space)

        # Verify feature count
        if hasattr(env.observation_space, 'spaces') and 'numeric' in env.observation_space.spaces:
            numeric_shape = env.observation_space.spaces['numeric'].shape[0]
            print(f"\n*** NUMERIC FEATURE COUNT: {numeric_shape} ***")
            if numeric_shape < 100:
                print("!!! WARNING: Expected ~137 features for v12, got only", numeric_shape)
            else:
                print("Feature count looks correct for v12 (expected ~137)")

        # Check for existing checkpoint
        latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
            checkpoint_dir, experiment_name + "_stage1"
        )

        if latest_checkpoint and checkpoint_timesteps < stage1_timesteps:
            print(f"\nResuming from checkpoint at {checkpoint_timesteps:,} steps")
            model = MaskablePPO.load(latest_checkpoint, env, device=device)
            model.learning_rate = lr_schedule
            model._setup_lr_schedule()
            timesteps_done = checkpoint_timesteps
        else:
            print("\nCreating new model...")
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch[0])
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(cnn_arch=cnn_arch, features_dim=512)

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
            timesteps_done = 0

        remaining = stage1_timesteps - timesteps_done

        if remaining > 0:
            print(f"\nTraining Stage 1: {remaining:,} steps remaining...")

            checkpoint_callback = CheckpointCallback(
                save_freq=500_000,
                save_path=checkpoint_dir,
                name_prefix=experiment_name + "_stage1"
            )

            model.learn(total_timesteps=remaining, callback=checkpoint_callback)

            # Save Stage 1 checkpoint
            model.save(stage1_checkpoint_path)
            print(f"\nStage 1 complete! Saved to: {stage1_checkpoint_path}.zip")
        else:
            print("Stage 1 already completed in previous run.")

        env.close()

    # ========== STAGE 2: Self-Play Training ==========
    print("\n" + "="*80)
    print("STAGE 2: SELF-PLAY TRAINING")
    print(f"Reward function: {reward_function_stage2.__name__} (switched from production_aware!)")
    print("="*80)

    # Load existing checkpoints into pool
    opponent_pool.load_existing_checkpoints()

    # Add Stage 1 model to pool as starting point
    if os.path.exists(stage1_checkpoint_path + ".zip"):
        opponent_pool.add_checkpoint(stage1_checkpoint_path + ".zip")

    # Check for Stage 2 progress
    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name + "_stage2"
    )

    if latest_checkpoint:
        stage2_done = checkpoint_timesteps
        print(f"Resuming Stage 2 from {stage2_done:,} steps")
    else:
        stage2_done = 0
        print("Starting Stage 2 from scratch")

    remaining_stage2 = stage2_timesteps - stage2_done

    if remaining_stage2 > 0:
        # Sample opponent from pool
        print(f"\nSampling opponent from pool...")
        opponent = opponent_pool.sample_opponent(fallback_path=stage1_checkpoint_path + ".zip")
        enemies_stage2 = [opponent]

        # Create self-play environment - use dense_vp_rewards for self-play!
        env = SubprocVecEnv([create_env(env_name, map_type, vps_to_win, enemies_stage2,
                                        reward_function_stage2, representation, i, seed)
                            for i in range(n_envs)])
        env = VecMonitor(env)

        # Load model for Stage 2
        if latest_checkpoint:
            print(f"Loading Stage 2 checkpoint: {latest_checkpoint}")
            model = MaskablePPO.load(latest_checkpoint, env, device=device)
            model.learning_rate = lr_schedule
            model._setup_lr_schedule()
        else:
            print(f"Loading Stage 1 model for Stage 2 training...")
            model = MaskablePPO.load(stage1_checkpoint_path, env, device=device)
            model.learning_rate = lr_schedule
            model._setup_lr_schedule()

        print(f"\nTraining Stage 2: {remaining_stage2:,} steps...")

        checkpoint_callback = CheckpointCallback(
            save_freq=500_000,
            save_path=checkpoint_dir,
            name_prefix=experiment_name + "_stage2"
        )

        model.learn(total_timesteps=remaining_stage2, callback=checkpoint_callback)

        env.close()
    else:
        print("Stage 2 already completed!")
        # Load final model
        latest_checkpoint, _ = find_latest_checkpoint(checkpoint_dir, experiment_name + "_stage2")
        if latest_checkpoint:
            # Create dummy env to load
            enemies_dummy = [ValueFunctionPlayer(Color.RED)]
            env = SubprocVecEnv([create_env(env_name, map_type, vps_to_win, enemies_dummy,
                                            reward_function_stage1, representation, i, seed)
                                for i in range(n_envs)])
            env = VecMonitor(env)
            model = MaskablePPO.load(latest_checkpoint, env, device=device)
            env.close()

    # Save final model
    model.save(model_path)

    # Training complete
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600

    print("\n" + "="*80)
    print("V12 TRAINING COMPLETED")
    print("="*80)
    print(f"Total training time:     {elapsed_time:>15.2f} seconds ({hours:.2f} hours)")
    print(f"Final model saved:       {model_path}.zip")
    print(f"\nRun evaluation with:")
    print(f"  python evaluate_ppo.py --model model_v12")
    print("="*80 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
