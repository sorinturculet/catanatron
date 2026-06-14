"""
Ablation study: v1 baseline WITHOUT expert guidance, 30M steps.

Identical to train_ablation_with_expert.py except no ValueWeightedWrapper.
Purpose: demonstrate the impact of expert-guided learning on training.
"""

import os
import random

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

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from reward_functions import partial_rewards, mask_fn


def learning_rate_schedule(initial_lr, final_lr):
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule


def main():
    # === v1 hyperparams exactly ===
    total_timesteps = 30_000_000
    cnn_arch = [64, 128, 256, 512]
    net_arch = [
        dict(
            vf=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256],
            pi=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256],
        )
    ]
    activation_fn = th.nn.LeakyReLU
    initial_lr = 1e-4
    final_lr = 1e-6
    ent_coef = 0.01
    gamma = 0.99
    seed = 42

    n_envs = 8
    n_steps = 256
    batch_size = n_envs * n_steps
    n_epochs = 10

    vps_to_win = 10
    env_name = "catanatron_gym:catanatron-v1"
    map_type = "BASE"
    enemies = [ValueFunctionPlayer(Color.RED)]
    reward_function = partial(partial_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = partial_rewards.__name__
    representation = "mixed"

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    experiment_name = "ablation_no_expert_30M"

    config = {
        "version": "ablation_no_expert",
        "approach": "v1 baseline WITHOUT expert guidance, 30M steps",
        "total_timesteps": total_timesteps,
        "lr": f"{initial_lr} → {final_lr} linear",
        "reward_function": reward_function.__name__,
        "expert_guidance": False,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "n_envs": n_envs,
        "batch_size": batch_size,
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="ablation-no-expert")

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
            # NO ValueWeightedWrapper — pure RL
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_ablation_no_expert")

    print("\n" + "=" * 70)
    print("ABLATION: NO EXPERT GUIDANCE (v1 baseline, 30M steps)")
    print("=" * 70)

    policy_kwargs: Any = dict(
        activation_fn=activation_fn,
        net_arch=net_arch[0],
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(cnn_arch=cnn_arch, features_dim=512),
    )
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log="./logs/mppo_tensorboard/" + experiment_name,
        device=device,
        seed=seed,
    )

    print(f"Total timesteps:      {total_timesteps:>12,}")
    print(f"LR: {initial_lr} → {final_lr} (linear)")
    print(f"Reward: {reward_function.__name__}")
    print(f"Expert guidance: OFF")
    print("=" * 70 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    callback = CallbackList([checkpoint_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/3600:.2f}h)")
    print(f"Model saved: {model_path}")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
