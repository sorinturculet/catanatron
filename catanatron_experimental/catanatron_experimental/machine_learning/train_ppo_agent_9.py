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
from catanatron_experimental.machine_learning.players.ppo import PPOPlayer
from catanatron.models.player import RandomPlayer
from reward_functions import dense_vp_rewards, mask_fn
from self_play_utils import SelfPlayOpponentManager


def find_latest_cycle_checkpoint(checkpoint_dir, prefix="ppo_v9_cycle"):
    """
    Find the latest cycle checkpoint.
    Returns (checkpoint_path, cycle_number) or (None, -1) if no checkpoint found.
    """
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None, -1

    latest_checkpoint = None
    max_cycle = -1

    for checkpoint_file in checkpoint_files:
        match = re.search(r'cycle_(\d+)_(\d+)_steps\.zip$', checkpoint_file)
        if match:
            cycle = int(match.group(1))
            if cycle > max_cycle:
                max_cycle = cycle
                latest_checkpoint = checkpoint_file

    return latest_checkpoint, max_cycle


def global_lr_schedule(cycle, num_cycles, initial_lr, final_lr):
    """
    Create a learning rate schedule that accounts for global progress.
    Each cycle's LR schedule maps to its portion of the global training.
    """
    def lr_schedule(progress_remaining):
        # progress_remaining goes 1.0 → 0.0 within this cycle
        # Map to global progress across all cycles
        cycle_frac = 1.0 / num_cycles
        global_start = 1.0 - (cycle * cycle_frac)
        global_end = 1.0 - ((cycle + 1) * cycle_frac)
        global_progress = global_start + (global_end - global_start) * (1.0 - progress_remaining)
        return final_lr + (initial_lr - final_lr) * global_progress

    return lr_schedule


def main():
    # ===== V9: Pure Self-Play From Scratch =====
    # True self-play: random init model trains against itself
    # Opponent updated every 500K steps (20 cycles of 500K = 10M total)
    # Single job, single sbatch submission

    total_timesteps = 10_000_000
    steps_per_cycle = 500_000
    num_cycles = total_timesteps // steps_per_cycle  # 20 cycles

    # Same network architecture as v3/v8
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[2048, 1024, 512, 256],
            pi=[2048, 1024, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU

    # Learning rate (same as v3)
    initial_lr = 3e-4
    final_lr = 3e-5

    # Entropy coefficient (same as v3)
    ent_coef = 0.02

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"

    # Use dense VP-based reward function (same as v3/v8)
    reward_function = partial(dense_vp_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = dense_vp_rewards.__name__
    representation = "mixed"

    # Core PPO parameters (same as v3)
    gamma = 0.99
    gae_lambda = 0.95
    selfplay = True
    seed = 42

    # Parallelization (same as v3)
    n_envs = 16
    n_steps = 512
    batch_size = 512
    n_epochs = 4
    max_grad_norm = 0.5

    # Validate batch configuration
    assert (n_envs * n_steps) % batch_size == 0, "batch_size must divide n_envs * n_steps"

    start_time = time.time()

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_v9")
    checkpoint_dir = "./logs/"

    # Opponent manager for self-play pool
    opponent_manager = SelfPlayOpponentManager(
        checkpoint_dir=checkpoint_dir,
        pool_size=5,
        experiment_name="ppo_v9_cycle"
    )

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
    enemy_desc = "CyclicSelfPlay"
    experiment_name = f"ppo_v9-{selfplay}-False-{iters}-{batch_size}-{gamma}-{enemy_desc}-{reward_function.__name__}-{representation}-{arch_str}-{initial_lr}lr-{vps_to_win}vp-{map_type}map"
    print(experiment_name)

    # WandB config
    config = {
        "initial_learning_rate": initial_lr,
        "final_learning_rate": final_lr,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
        "steps_per_cycle": steps_per_cycle,
        "num_cycles": num_cycles,
        "net_arch": net_arch,
        "activation_fn": activation_fn.__name__,
        "vps_to_win": vps_to_win,
        "map_type": map_type,
        "enemies": ["Self (cyclic update every 500K steps)"],
        "reward_function": reward_function.__name__,
        "representation": representation,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "cnn_arch": cnn_arch,
        "selfplay": selfplay,
        "experiment_name": experiment_name,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "version": "v9_cyclic_selfplay",
        "approach": "random_init_cyclic_selfplay_10M",
        "opponent_pool_size": opponent_manager.pool_size,
    }
    run = wandb.init(
        project="catanatron",
        config=config,
        sync_tensorboard=True,
    )

    def print_name():
        print(experiment_name)

    atexit.register(print_name)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Check for existing checkpoints to resume =====
    latest_checkpoint, last_cycle = find_latest_cycle_checkpoint(checkpoint_dir)
    start_cycle = 0
    model = None

    if latest_checkpoint and last_cycle >= 0:
        print(f"\n✓ Found checkpoint from cycle {last_cycle + 1}/{num_cycles}")
        print(f"  Path: {latest_checkpoint}")
        try:
            # We need a temporary env to load the model
            # Use RandomPlayer as placeholder (will be replaced in cycle loop)
            tmp_enemies = [RandomPlayer(Color.RED)]

            def make_tmp_env(rank, seed=0):
                def _init():
                    env = gym.make(
                        env_name,
                        config={
                            "map_type": map_type,
                            "vps_to_win": vps_to_win,
                            "enemies": tmp_enemies,
                            "reward_function": reward_function,
                            "representation": representation,
                            "normalized": True,
                        },
                    )
                    env = ActionMasker(env, mask_fn)
                    return env
                return _init

            tmp_env = SubprocVecEnv([make_tmp_env(i, seed) for i in range(n_envs)])
            tmp_env = VecMonitor(tmp_env)

            model = MaskablePPO.load(latest_checkpoint, tmp_env, device=device)
            model.gamma = gamma
            model.gae_lambda = gae_lambda
            model.ent_coef = ent_coef
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.max_grad_norm = max_grad_norm
            start_cycle = last_cycle + 1
            print(f"✓ Resuming from cycle {start_cycle + 1}/{num_cycles}")
            print(f"  Timesteps completed: {start_cycle * steps_per_cycle:,}")

            # Close temporary env (will create proper one in loop)
            tmp_env.close()

            # Load existing checkpoints into opponent pool
            opponent_manager.load_existing_checkpoints()
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            model = None
            start_cycle = 0

    # ===== Create new model if needed =====
    if model is None:
        print("\n" + "="*80)
        print("CREATING NEW MODEL FROM SCRATCH (RANDOM INIT)")
        print("="*80)
        print(f"  Architecture: {net_arch}")
        print(f"  CNN Architecture: {cnn_arch}")
        print(f"  Learning rate: {initial_lr} → {final_lr}")
        print(f"  Approach: Cyclic self-play ({num_cycles} cycles of {steps_per_cycle:,} steps)")

        # Create a temporary env with RandomPlayer to initialize the model
        init_enemies = [RandomPlayer(Color.RED)]

        def make_init_env(rank, seed=0):
            def _init():
                env = gym.make(
                    env_name,
                    config={
                        "map_type": map_type,
                        "vps_to_win": vps_to_win,
                        "enemies": init_enemies,
                        "reward_function": reward_function,
                        "representation": representation,
                        "normalized": True,
                    },
                )
                env = ActionMasker(env, mask_fn)
                return env
            return _init

        init_env = SubprocVecEnv([make_init_env(i, seed) for i in range(n_envs)])
        init_env = VecMonitor(init_env)

        lr_schedule = global_lr_schedule(0, num_cycles, initial_lr, final_lr)
        policy_kwargs: Any = dict(activation_fn=activation_fn, net_arch=net_arch[0])
        if representation == "mixed":
            policy_kwargs["features_extractor_class"] = CustomCNN
            policy_kwargs["features_extractor_kwargs"] = dict(
                cnn_arch=cnn_arch, features_dim=512
            )
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            init_env,
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

        # Save the initial random model as the first opponent
        init_opponent_path = os.path.join(checkpoint_dir, "ppo_v9_cycle_0_0_steps.zip")
        model.save(init_opponent_path)
        opponent_manager.update_opponent_pool(init_opponent_path)
        print(f"  Saved initial random model as first opponent: {init_opponent_path}")

        init_env.close()

    # ===== TRAINING CONFIGURATION =====
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION (CYCLIC SELF-PLAY)")
    print("="*80)
    print(f"Approach:                Cyclic self-play (random init)")
    print(f"Total timesteps:         {total_timesteps:>15,}")
    print(f"Steps per cycle:         {steps_per_cycle:>15,}")
    print(f"Number of cycles:        {num_cycles:>15,}")
    print(f"Starting from cycle:     {start_cycle + 1:>15,}")
    print(f"Remaining cycles:        {num_cycles - start_cycle:>15,}")
    print(f"\nReward function:         {reward_function.__name__:>15}")
    print(f"  - VP gain:             +0.1 per VP")
    print(f"  - VP loss:             -0.1 per VP")
    print(f"  - Win bonus:           +1.0")
    print(f"  - Loss penalty:        -1.0")
    print(f"\nParallel environments:   {n_envs:>15,}")
    print(f"Steps per rollout:       {n_steps:>15,}")
    print(f"Batch size:              {batch_size:>15,}")
    print(f"Epochs per update:       {n_epochs:>15,}")
    print(f"\nLearning rate:           {initial_lr:>15.6f} → {final_lr:.6f}")
    print(f"Entropy coefficient:     {ent_coef:>15.4f}")
    print(f"Gamma (discount):        {gamma:>15.4f}")
    print(f"GAE lambda:              {gae_lambda:>15.4f}")
    print(f"\nOpponent pool size:      {opponent_manager.pool_size:>15,}")
    print(f"Opponent update:         Every {steps_per_cycle:,} steps (each cycle)")
    print("="*80 + "\n")

    if start_cycle >= num_cycles:
        print("✓ Training already completed!")
        model.save(model_path)
        run.finish()
        return

    print("="*80)
    print("STARTING CYCLIC SELF-PLAY TRAINING")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"WandB Run: https://wandb.ai/sorinturculet-babes-bolyai-university/catanatron/runs/{run.id}")
    print("="*80 + "\n")

    # ===== MAIN TRAINING LOOP: 20 CYCLES =====
    for cycle in range(start_cycle, num_cycles):
        cycle_start_time = time.time()
        total_steps_so_far = cycle * steps_per_cycle

        # --- 1. Select opponent for this cycle ---
        if len(opponent_manager.opponent_checkpoints) > 0:
            opponent = opponent_manager.sample_opponent()
        else:
            # Fallback: use RandomPlayer for cycle 0 if no checkpoints exist
            opponent = RandomPlayer(Color.RED)

        opponent_name = getattr(opponent, 'model_path', 'RandomPlayer') if hasattr(opponent, 'model_path') else type(opponent).__name__

        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle + 1}/{num_cycles} ({total_steps_so_far:,} → {total_steps_so_far + steps_per_cycle:,} steps)")
        print(f"  Opponent: {opponent_name}")
        print(f"  Pool size: {len(opponent_manager.opponent_checkpoints)}")
        print(f"{'='*60}")

        # --- 2. Create environment with current opponent ---
        cycle_enemies = [opponent]

        def make_cycle_env(rank, seed=0, enemies=cycle_enemies):
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

        env = SubprocVecEnv([make_cycle_env(i, seed) for i in range(n_envs)])
        env = VecMonitor(env)
        model.set_env(env)

        # --- 3. Set learning rate for this cycle (global schedule) ---
        lr_schedule = global_lr_schedule(cycle, num_cycles, initial_lr, final_lr)
        model.learning_rate = lr_schedule
        model._setup_lr_schedule()

        # --- 4. Train for 500K steps ---
        checkpoint_prefix = f"ppo_v9_cycle_{cycle + 1}"
        checkpoint_callback = CheckpointCallback(
            save_freq=steps_per_cycle,  # Save once at end of cycle
            save_path=checkpoint_dir,
            name_prefix=checkpoint_prefix,
        )

        model.learn(
            total_timesteps=steps_per_cycle,
            callback=CallbackList([checkpoint_callback]),
            reset_num_timesteps=True,  # Reset per cycle for clean callback triggers
        )

        # --- 5. Save cycle checkpoint ---
        cycle_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"ppo_v9_cycle_{cycle + 1}_{(cycle + 1) * steps_per_cycle}_steps.zip"
        )
        model.save(cycle_checkpoint_path)

        # --- 6. Update opponent pool ---
        opponent_manager.update_opponent_pool(cycle_checkpoint_path)

        # --- 7. Close environment for this cycle ---
        env.close()

        cycle_elapsed = time.time() - cycle_start_time
        print(f"  Cycle {cycle + 1} completed in {cycle_elapsed:.1f}s")
        print(f"  Checkpoint saved: {cycle_checkpoint_path}")
        print(f"  {opponent_manager.get_pool_info()}")

    # ===== Save final model =====
    model.save(model_path)

    # Training completion summary
    elapsed_time = time.time() - start_time
    total_trained = (num_cycles - start_cycle) * steps_per_cycle
    timesteps_per_second = total_trained / elapsed_time if elapsed_time > 0 else 0
    hours = elapsed_time / 3600

    print("\n" + "="*80)
    print("TRAINING COMPLETED (CYCLIC SELF-PLAY)")
    print("="*80)
    print(f"Total training time:     {elapsed_time:>15.2f} seconds ({hours:.2f} hours)")
    print(f"Cycles completed:        {num_cycles - start_cycle:>15,} / {num_cycles}")
    print(f"Timesteps trained:       {total_trained:>15,}")
    print(f"Throughput:              {timesteps_per_second:>15.2f} timesteps/second")
    print(f"Final model saved:       {model_path}")
    print(f"Checkpoints saved in:    {checkpoint_dir}")
    print(f"\n{opponent_manager.get_pool_info()}")
    print("="*80 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
