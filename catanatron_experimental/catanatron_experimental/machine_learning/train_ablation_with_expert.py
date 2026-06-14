"""
Ablation study: v1 baseline WITH expert guidance, 30M steps.

Identical to train_ablation_no_expert.py except adds ValueWeightedWrapper.
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
    BaseCallback,
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
from value_weighted_wrapper import ValueWeightedWrapper


def learning_rate_schedule(initial_lr, final_lr):
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule


class ValueScoreLoggingCallback(BaseCallback):
    """Log ValueWeightedWrapper metrics to WandB."""

    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                if infos:
                    import numpy as np
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

    # Expert guidance params (same as v13 which first introduced it)
    bonus_scale = 0.2
    decay_start_step = 5_000_000
    decay_rate = 0.99999
    min_scale = 0.05

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    lr_schedule = learning_rate_schedule(initial_lr, final_lr)

    experiment_name = "ablation_with_expert_30M"

    config = {
        "version": "ablation_with_expert",
        "approach": "v1 baseline WITH expert guidance, 30M steps",
        "total_timesteps": total_timesteps,
        "lr": f"{initial_lr} → {final_lr} linear",
        "reward_function": reward_function.__name__,
        "expert_guidance": True,
        "bonus_scale": bonus_scale,
        "decay_start_step": decay_start_step,
        "decay_rate": decay_rate,
        "min_scale": min_scale,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "n_envs": n_envs,
        "batch_size": batch_size,
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="ablation-with-expert")

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
            # WITH ValueWeightedWrapper — expert-guided RL
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_ablation_with_expert")

    print("\n" + "=" * 70)
    print("ABLATION: WITH EXPERT GUIDANCE (v1 + ValueWeightedWrapper, 30M steps)")
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
    print(f"Expert guidance: ON (bonus {bonus_scale} → {min_scale} floor at {decay_start_step/1e6:.0f}M)")
    print("=" * 70 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, save_path="./logs/", name_prefix=experiment_name
    )
    value_callback = ValueScoreLoggingCallback(log_freq=10_000)
    callback = CallbackList([checkpoint_callback, value_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/3600:.2f}h)")
    print(f"Model saved: {model_path}")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
