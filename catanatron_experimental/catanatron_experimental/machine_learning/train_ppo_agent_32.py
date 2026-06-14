"""
V32: Extend v28 (34%) with LR reset + self-play league

Phase 1 (0-15M):  Train vs VFP only, LR 5e-5→1e-5 — squeeze more from v28
Phase 2 (15-40M): 50% VFP + 50% frozen self (updated every 5M), LR 1e-5→2e-6 — generalize

Loads v28 model. Same architecture, same rewards. The self-play opponents are
frozen copies of the agent saved at phase 2 checkpoints.

Self-play implementation: each env randomly picks VFP or a frozen PPO copy
as the opponent at each game reset. The frozen copies are stored on disk
and loaded by PPOPlayer instances.
"""

import os
import random
import glob
import re
import shutil

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
from catanatron_experimental.machine_learning.players.ppo import PPOPlayer
from reward_functions import expansion_heavy_rewards, mask_fn
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


def make_absolute_lr_schedule(total_timesteps, offset, stages):
    """Multi-stage LR schedule immune to progress_remaining reset bug."""
    remaining = total_timesteps - offset

    def schedule(progress_remaining):
        steps_in_this_run = (1.0 - progress_remaining) * remaining
        abs_step = offset + steps_in_this_run
        abs_frac = abs_step / total_timesteps

        for frac_start, frac_end, lr_start, lr_end in stages:
            if abs_frac < frac_end:
                local_frac = (abs_frac - frac_start) / (frac_end - frac_start)
                local_frac = max(0.0, min(1.0, local_frac))
                return lr_start + (lr_end - lr_start) * local_frac

        return stages[-1][3]

    return schedule


class MixedOpponentWrapper(gym.Wrapper):
    """
    Wrapper that randomly picks opponent at each reset:
    - VFP (ValueFunctionPlayer) with probability `vfp_prob`
    - Frozen PPO copy with probability `1 - vfp_prob`

    The frozen PPO model path is stored in a shared file that gets
    updated by the SelfPlayCallback.
    """

    def __init__(self, env, vfp_player, self_play_model_path, vfp_prob=0.5):
        super().__init__(env)
        self.vfp_player = vfp_player
        self.self_play_model_path = self_play_model_path
        self.vfp_prob = vfp_prob
        self.ppo_opponent = None
        self.current_opponent = "vfp"
        self._games_played = 0
        self._vfp_games = 0
        self._selfplay_games = 0

    def _load_ppo_opponent(self):
        """Load or reload the frozen PPO opponent."""
        try:
            if os.path.exists(self.self_play_model_path):
                self.ppo_opponent = PPOPlayer(Color.RED, model_path=self.self_play_model_path)
                return True
        except Exception:
            pass
        return False

    def reset(self, **kwargs):
        self._games_played += 1

        # Decide opponent for this game
        use_vfp = random.random() < self.vfp_prob

        # If no self-play model available yet, always use VFP
        if not use_vfp and self.ppo_opponent is None:
            if not self._load_ppo_opponent():
                use_vfp = True

        if use_vfp:
            self.env.unwrapped.enemies = [self.vfp_player]
            self.current_opponent = "vfp"
            self._vfp_games += 1
        else:
            # Reload periodically to pick up new frozen checkpoints
            if self._selfplay_games % 50 == 0:
                self._load_ppo_opponent()
            if self.ppo_opponent is not None:
                self.env.unwrapped.enemies = [self.ppo_opponent]
                self.current_opponent = "selfplay"
                self._selfplay_games += 1
            else:
                self.env.unwrapped.enemies = [self.vfp_player]
                self.current_opponent = "vfp"
                self._vfp_games += 1

        # Rebuild players list
        self.env.unwrapped.players = [self.env.unwrapped.p0] + self.env.unwrapped.enemies

        return self.env.reset(**kwargs)


class SelfPlayCallback(BaseCallback):
    """
    Callback that:
    1. Saves frozen copies of the agent for self-play opponents
    2. Logs metrics + LR + phase info to WandB
    3. Switches from Phase 1 (VFP only) to Phase 2 (mixed) at boundary
    """

    def __init__(self, total_timesteps, offset, self_play_model_path,
                 phase2_start_step, freeze_interval, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.offset = offset
        self.self_play_model_path = self_play_model_path
        self.phase2_start_step = phase2_start_step
        self.freeze_interval = freeze_interval
        self.log_freq = log_freq
        self.last_freeze_step = 0
        self.phase2_active = False
        self.freeze_count = 0

    def _on_step(self):
        abs_step = self.offset + self.num_timesteps

        # Phase 2 activation: save first frozen copy and enable self-play
        if not self.phase2_active and abs_step >= self.phase2_start_step:
            self.phase2_active = True
            self._freeze_model(abs_step)
            # Update VFP probability in envs to enable self-play
            try:
                for env_idx in range(self.model.env.num_envs):
                    self.model.env.envs[env_idx].vfp_prob = 0.5
            except Exception:
                pass
            print(f"\n  === PHASE 2 ACTIVE at {abs_step/1e6:.1f}M — self-play enabled (50/50) ===\n")

        # Periodically freeze model during phase 2
        if self.phase2_active and (abs_step - self.last_freeze_step) >= self.freeze_interval:
            self._freeze_model(abs_step)

        # Logging
        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                log_dict = {
                    "train/absolute_step": abs_step,
                    "train/phase": 2 if self.phase2_active else 1,
                    "train/freeze_count": self.freeze_count,
                }

                if infos:
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

                lr = self.model.lr_schedule(self.model._current_progress_remaining)
                log_dict["train/learning_rate_actual"] = lr

                wandb.log(log_dict, step=abs_step)
            except Exception:
                pass
        return True

    def _freeze_model(self, abs_step):
        """Save current model as frozen opponent for self-play."""
        try:
            self.model.save(self.self_play_model_path)
            self.last_freeze_step = abs_step
            self.freeze_count += 1
            print(f"  Froze model #{self.freeze_count} at {abs_step/1e6:.1f}M → {self.self_play_model_path}")
        except Exception as e:
            print(f"  Failed to freeze model: {e}")


def main():
    total_timesteps = 40_000_000

    # Phase 1: 0-15M vs VFP, LR reset from 5e-5
    # Phase 2: 15-40M mixed (50% VFP + 50% self), lower LR
    p1_frac = 15_000_000 / total_timesteps  # 0.375

    lr_stages = [
        (0.00,   p1_frac, 5e-5, 1e-5),    # Phase 1: squeeze more from v28
        (p1_frac, 1.00,   1e-5, 2e-6),     # Phase 2: refine with self-play
    ]

    # Architecture — same as v28
    gamma      = 0.997
    gae_lambda = 0.95
    ent_coef   = 0.02
    n_envs     = 32
    n_steps    = 512
    batch_size = 1024
    n_epochs   = 4
    max_grad_norm = 0.5
    seed       = 42

    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    # Expert bonus: constant at floor (v28 already decayed)
    bonus_scale      = 0.05
    decay_start_step = 0
    decay_rate       = 1.0
    min_scale        = 0.05

    # Self-play config
    phase2_start_step = 15_000_000
    freeze_interval   = 5_000_000  # freeze every 5M steps

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"

    reward_function = partial(expansion_heavy_rewards, vps_to_win=vps_to_win)
    reward_function.__name__ = expansion_heavy_rewards.__name__
    representation = "mixed"

    assert (n_envs * n_steps) % batch_size == 0

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    experiment_name = "ppo_v32-ExtendV28-SelfPlay-40M"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    self_play_model_path = os.path.join(script_dir, "model_v32_frozen_opponent.zip")

    config = {
        "version": "v32_extend_v28_selfplay",
        "approach": "Extend v28 (34%): Phase 1 vs VFP (LR reset), Phase 2 mixed self-play",
        "base_model": "model_v28 (65M steps, 34% win rate)",
        "total_timesteps": total_timesteps,
        "phase1": "0-15M: vs VFP only, LR 5e-5→1e-5, expert 0.05",
        "phase2": "15-40M: 50% VFP + 50% frozen self (updated every 5M), LR 1e-5→2e-6",
        "reward_function": reward_function.__name__,
        "net_arch": str(net_arch),
        "gamma": gamma,
        "n_envs": n_envs,
        "batch_size": batch_size,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v32")

    def print_name():
        print(experiment_name)
    atexit.register(print_name)

    per_env_decay_start = decay_start_step // n_envs

    def make_env(rank, seed=0):
        def _init():
            vfp = ValueFunctionPlayer(Color.RED)
            env = gym.make(
                env_name,
                config={
                    "map_type": map_type,
                    "vps_to_win": vps_to_win,
                    "enemies": [vfp],
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
            # Wrap with mixed opponent — starts with vfp_prob=1.0 (Phase 1)
            # SelfPlayCallback will change to 0.5 at Phase 2
            env = MixedOpponentWrapper(
                env,
                vfp_player=vfp,
                self_play_model_path=self_play_model_path,
                vfp_prob=1.0,  # Phase 1: VFP only
            )
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_v28_path = os.path.join(script_dir, "model_v28.zip")
    model_path     = os.path.join(script_dir, "model_v32")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V32: EXTEND V28 + SELF-PLAY)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    if latest_checkpoint:
        print(f"\n  Found v32 checkpoint at {checkpoint_timesteps:,} steps")
        try:
            lr_schedule = make_absolute_lr_schedule(
                total_timesteps, checkpoint_timesteps, lr_stages
            )
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
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print(f"\n  Loading v28 model from: {model_v28_path}")

    if not model_loaded:
        lr_schedule = make_absolute_lr_schedule(total_timesteps, 0, lr_stages)
        model = MaskablePPO.load(
            model_v28_path, env, device=device,
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
        print("  Loaded v28 successfully")

    remaining_timesteps = total_timesteps - timesteps_completed

    test_lr_start = lr_schedule(1.0)
    test_lr_end   = lr_schedule(0.0)
    print(f"\n  LR verification:")
    print(f"    schedule(1.0) = {test_lr_start:.2e}")
    print(f"    schedule(0.0) = {test_lr_end:.2e}")

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V32: EXTEND V28 + SELF-PLAY)")
    print("=" * 70)
    print(f"Base model:           v28 (65M steps, 34% win rate)")
    print(f"Total timesteps:      {total_timesteps:>12,}")
    print(f"Already completed:    {timesteps_completed:>12,}")
    print(f"Remaining:            {remaining_timesteps:>12,}")
    print(f"\n  Phase 1 (0–15M):  vs VFP only, LR 5e-5→1e-5")
    print(f"  Phase 2 (15–40M): 50% VFP + 50% frozen self, LR 1e-5→2e-6")
    print(f"    Freeze interval: every {freeze_interval/1e6:.0f}M steps")
    print(f"\nReward: {reward_function.__name__}")
    print(f"Expert bonus: constant {bonus_scale} (floor)")
    print(f"Gamma: {gamma}")
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
    selfplay_callback = SelfPlayCallback(
        total_timesteps=total_timesteps,
        offset=timesteps_completed,
        self_play_model_path=self_play_model_path,
        phase2_start_step=phase2_start_step,
        freeze_interval=freeze_interval,
        log_freq=10_000,
    )
    callback = CallbackList([checkpoint_callback, selfplay_callback])

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
    print("TRAINING COMPLETED (V32)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
