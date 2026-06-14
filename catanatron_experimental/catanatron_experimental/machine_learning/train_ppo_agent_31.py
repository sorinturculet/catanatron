"""
V31: 3-phase curriculum — imitate, explore, refine

Phase 1 (0-20M):  IMITATE — strong expert bonus 0.5→0.2, learn to play like VFP
Phase 2 (20-40M): EXPLORE — expert fades 0.2→0.05, log LR stays high, ent=0.03
Phase 3 (40-65M): REFINE  — expert off 0.05→0.0, low LR, ent=0.01, fly solo

Same v28 architecture and rewards (34% best). Key insight: catanrl got 70% by
imitating first then RL. We simulate this with strong expert bonus in Phase 1.
"""

import os
import random
import glob
import re
import math

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
    """
    Multi-stage LR schedule with per-stage linear or log interpolation.
    stages: list of (frac_start, frac_end, lr_start, lr_end, mode)
    mode: 'linear' or 'log'
    """
    remaining = total_timesteps - offset

    def schedule(progress_remaining):
        steps_in_this_run = (1.0 - progress_remaining) * remaining
        abs_step = offset + steps_in_this_run
        abs_frac = abs_step / total_timesteps

        for frac_start, frac_end, lr_start, lr_end, mode in stages:
            if abs_frac < frac_end:
                local_frac = (abs_frac - frac_start) / (frac_end - frac_start)
                local_frac = max(0.0, min(1.0, local_frac))
                if mode == 'log':
                    log_lr = math.log(lr_start) + (math.log(lr_end) - math.log(lr_start)) * local_frac
                    return math.exp(log_lr)
                else:
                    return lr_start + (lr_end - lr_start) * local_frac

        return stages[-1][3]

    return schedule


class PhaseAwareCallback(BaseCallback):
    """
    Callback that:
    1. Changes ent_coef at phase boundaries
    2. Logs metrics + actual LR + phase info to WandB
    """

    def __init__(self, total_timesteps, offset, phase_boundaries, ent_coefs, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_timesteps = total_timesteps
        self.offset = offset
        self.phase_boundaries = phase_boundaries  # [20M, 40M]
        self.ent_coefs = ent_coefs  # [0.02, 0.03, 0.01]
        self.current_phase = 0

    def _on_step(self):
        abs_step = self.offset + self.num_timesteps

        # Check phase transitions
        new_phase = 0
        for i, boundary in enumerate(self.phase_boundaries):
            if abs_step >= boundary:
                new_phase = i + 1

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            new_ent = self.ent_coefs[new_phase]
            self.model.ent_coef = new_ent
            print(f"\n  === PHASE {new_phase + 1} at step {abs_step/1e6:.1f}M — ent_coef → {new_ent} ===\n")

        if self.num_timesteps % self.log_freq == 0:
            try:
                infos = self.locals.get('infos', [])
                log_dict = {
                    "train/absolute_step": abs_step,
                    "train/phase": self.current_phase + 1,
                    "train/ent_coef": self.model.ent_coef,
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

                frac = abs_step / self.total_timesteps
                if frac < 20/65:
                    log_dict["train/lr_stage"] = 1
                elif frac < 40/65:
                    log_dict["train/lr_stage"] = 2
                else:
                    log_dict["train/lr_stage"] = 3

                wandb.log(log_dict, step=abs_step)
            except Exception:
                pass
        return True


def main():
    total_timesteps = 65_000_000
    n_envs = 16

    # 3-phase LR schedule: linear, LOG, linear
    # fractions: Phase 1 = 0-20M (0-0.3077), Phase 2 = 20-40M (0.3077-0.6154), Phase 3 = 40-65M (0.6154-1.0)
    p1_end = 20_000_000 / total_timesteps  # 0.3077
    p2_end = 40_000_000 / total_timesteps  # 0.6154

    lr_stages = [
        (0.00,  p1_end, 3e-4, 3e-5, 'linear'),  # Phase 1: imitate
        (p1_end, p2_end, 1e-4, 1e-5, 'log'),     # Phase 2: explore (LOG — stays high longer)
        (p2_end, 1.00,  3e-5, 2e-6, 'linear'),   # Phase 3: refine
    ]

    # Architecture — same as v28 (34% best)
    gamma      = 0.997
    gae_lambda = 0.95
    ent_coef   = 0.02   # Phase 1 entropy (changes via callback)
    n_steps    = 512
    batch_size = 512
    n_epochs   = 4
    max_grad_norm = 0.5
    seed       = 42

    cnn_arch = [64, 128, 256, 512]
    net_arch = [dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])]
    activation_fn = th.nn.LeakyReLU
    features_dim = 512

    # Expert bonus phases (per-env steps)
    # Phase 1: 0.5 → 0.2 over 20M/16 = 1.25M per-env steps
    # Phase 2: 0.2 → 0.05 over 20M/16 = 1.25M per-env steps
    # Phase 3: 0.05 → 0.0 over 25M/16 = 1.5625M per-env steps
    per_env_total = total_timesteps // n_envs
    per_env_p1 = 20_000_000 // n_envs
    per_env_p2 = 40_000_000 // n_envs

    bonus_phases = [
        (0, 0.5),
        (per_env_p1, 0.2),
        (per_env_p2, 0.05),
        (per_env_total, 0.0),
    ]

    # Entropy phases
    phase_boundaries = [20_000_000, 40_000_000]
    ent_coefs = [0.02, 0.03, 0.01]  # Phase 1, 2, 3

    vps_to_win   = 10
    env_name     = "catanatron_gym:catanatron-v1"
    map_type     = "BASE"
    enemies      = [ValueFunctionPlayer(Color.RED)]

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

    experiment_name = "ppo_v31-3Phase-Curriculum-65M"

    config = {
        "version": "v31_3phase_curriculum",
        "approach": "3-phase: imitate(0.5→0.2) → explore(log LR, ent=0.03) → refine(no expert, ent=0.01)",
        "total_timesteps": total_timesteps,
        "phase1": "0-20M: expert 0.5→0.2, LR 3e-4→3e-5 linear, ent=0.02",
        "phase2": "20-40M: expert 0.2→0.05, LR 1e-4→1e-5 LOG, ent=0.03",
        "phase3": "40-65M: expert 0.05→0.0, LR 3e-5→2e-6 linear, ent=0.01",
        "reward_function": reward_function.__name__,
        "reward_notes": "VP +0.10, OWS +0.02, BW +0.04, settle +0.25, road +0.12, city +0.08, win/loss +/-1.0",
        "net_arch": str(net_arch),
        "cnn_arch": cnn_arch,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_envs": n_envs,
        "batch_size": batch_size,
        "fresh_start": True,
    }
    run = wandb.init(project="catanatron", config=config, sync_tensorboard=True, name="v31")

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
            env = ValueWeightedWrapper(
                env,
                player_color=Color.BLUE,
                bonus_scale=0.5,
                bonus_phases=bonus_phases,
            )
            return env
        return _init

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    print("Observation Space:", env.observation_space)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(script_dir, "model_v31")
    checkpoint_dir = "./logs/"

    print("\n" + "=" * 70)
    print("MODEL LOADING (V31: 3-PHASE CURRICULUM)")
    print("=" * 70)

    latest_checkpoint, checkpoint_timesteps = find_latest_checkpoint(
        checkpoint_dir, experiment_name
    )

    model_loaded = False
    timesteps_completed = 0

    if latest_checkpoint:
        print(f"\n  Found checkpoint at {checkpoint_timesteps:,} steps")
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
            model.learning_rate = lr_schedule
            model.batch_size    = batch_size
            model.n_epochs      = n_epochs
            model.max_grad_norm = max_grad_norm
            model._setup_lr_schedule()
            timesteps_completed = checkpoint_timesteps
            model_loaded = True

            # Set correct entropy for current phase
            phase = 0
            for i, boundary in enumerate(phase_boundaries):
                if timesteps_completed >= boundary:
                    phase = i + 1
            model.ent_coef = ent_coefs[phase]

            print(f"  Resuming from {timesteps_completed:,} steps (Phase {phase + 1})")
            test_lr = lr_schedule(1.0)
            print(f"  LR at resume: {test_lr:.2e}, ent_coef: {model.ent_coef}")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n  No checkpoint — starting fresh (as intended)")

    if not model_loaded:
        lr_schedule = make_absolute_lr_schedule(total_timesteps, 0, lr_stages)
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

    test_lr_start = lr_schedule(1.0)
    test_lr_end   = lr_schedule(0.0)
    print(f"\n  LR verification:")
    print(f"    schedule(1.0) = {test_lr_start:.2e}")
    print(f"    schedule(0.0) = {test_lr_end:.2e}")

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (V31: 3-PHASE CURRICULUM)")
    print("=" * 70)
    print(f"Total timesteps:      {total_timesteps:>15,}  (FRESH)")
    print(f"Already completed:    {timesteps_completed:>15,}")
    print(f"Remaining:            {remaining_timesteps:>15,}")
    print(f"\n  Phase 1 (IMITATE)  0–20M:  expert 0.5→0.2, LR 3e-4→3e-5 linear, ent=0.02")
    print(f"  Phase 2 (EXPLORE) 20–40M:  expert 0.2→0.05, LR 1e-4→1e-5 LOG,    ent=0.03")
    print(f"  Phase 3 (REFINE)  40–65M:  expert 0.05→0.0, LR 3e-5→2e-6 linear, ent=0.01")
    print(f"\nReward: {reward_function.__name__}")
    print(f"  VP +0.10, OWS +0.02, BW +0.04, settle +0.25, road +0.12, city +0.08")
    print(f"  Gamma: {gamma}")
    print(f"\nNetwork: pi/vf = [2048, 1024, 512, 256] (same as v28)")
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
    phase_callback = PhaseAwareCallback(
        total_timesteps=total_timesteps,
        offset=timesteps_completed,
        phase_boundaries=phase_boundaries,
        ent_coefs=ent_coefs,
        log_freq=10_000,
    )
    callback = CallbackList([checkpoint_callback, phase_callback])

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
    print("TRAINING COMPLETED (V31)")
    print("=" * 70)
    print(f"Total time:   {elapsed:.2f}s  ({elapsed/3600:.2f}h)")
    print(f"Throughput:   {remaining_timesteps/elapsed:.1f} steps/s")
    print(f"Model saved:  {model_path}")
    print("=" * 70 + "\n")

    run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
