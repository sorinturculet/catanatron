"""
Behavioral Cloning: collect expert data from VFP + train policy network.

Phase 1 of the BC → PPO pipeline.
  1. Play N games using VFP's decisions in the gym env
  2. Save (observation, expert_action, mask, value_target) chunks to disk
  3. Train a MaskablePPO policy with cross-entropy loss on expert actions
     + MSE loss on value head using VFP's state evaluation
  4. Save as model_bc.zip (compatible with MaskablePPO.load)

Memory-safe: saves chunks every 1000 games, streams during training.
"""

import os
import time
import random
import glob
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
    base_fn,
)
from catanatron_gym.envs.catanatron_env import to_action_space
from reward_functions import mask_fn


# ─── Config ───
NUM_GAMES = 100_000      # ~15M decision points
BC_EPOCHS = 10
BC_BATCH_SIZE = 256
BC_LR = 1e-3
VALUE_LOSS_WEIGHT = 0.5
SEED = 42
CHUNK_SIZE = 5000        # save to disk every 5000 games (~15GB per chunk in RAM)

# Architecture (must match v28 exactly)
CNN_ARCH = [64, 128, 256, 512]
NET_ARCH = dict(vf=[2048, 1024, 512, 256], pi=[2048, 1024, 512, 256])
FEATURES_DIM = 512


def collect_expert_data(num_games, data_dir):
    """Play games using VFP's decisions, save chunks to disk."""
    print(f"\n{'='*60}")
    print(f"COLLECTING EXPERT DATA ({num_games} games)")
    print(f"Saving chunks to {data_dir}")
    print(f"{'='*60}\n")

    os.makedirs(data_dir, exist_ok=True)

    env = gym.make(
        "catanatron_gym:catanatron-v1",
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "enemies": [ValueFunctionPlayer(Color.RED)],
            "representation": "mixed",
            "normalized": True,
        },
    )
    env = ActionMasker(env, mask_fn)
    value_fn = base_fn()

    # Resume from existing chunks if any
    existing = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    chunk_idx = len(existing)
    games_already = chunk_idx * CHUNK_SIZE
    if chunk_idx > 0:
        print(f"  Found {chunk_idx} existing chunks ({games_already} games) — collecting remaining {num_games - games_already} games")

    boards, numerics, actions, masks, values = [], [], [], [], []
    total_decisions = 0
    wins = 0

    t0 = time.time()
    for game_i in range(games_already, num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            game = env.unwrapped.game
            playable_actions = game.state.playable_actions

            if len(playable_actions) <= 1:
                action_idx = to_action_space(playable_actions[0]) if playable_actions else 0
                obs, _, terminated, truncated, _ = env.step(action_idx)
                done = terminated or truncated
                continue

            # VFP picks the best action (1-ply lookahead)
            best_value = float("-inf")
            best_action = None
            for pa in playable_actions:
                game_copy = game.copy()
                game_copy.execute(pa)
                v = value_fn(game_copy, Color.BLUE)
                if v > best_value:
                    best_value = v
                    best_action = pa

            expert_action_idx = to_action_space(best_action)

            # Action mask
            valid = env.unwrapped.get_valid_actions()
            action_mask = np.zeros(env.action_space.n, dtype=np.float32)
            action_mask[valid] = 1.0

            # VFP state evaluation (normalized for value head)
            current_value = value_fn(game, Color.BLUE)
            norm_value = np.clip(current_value / 1e15, -5.0, 5.0)

            boards.append(obs["board"])
            numerics.append(obs["numeric"])
            actions.append(expert_action_idx)
            masks.append(action_mask)
            values.append(norm_value)
            total_decisions += 1

            obs, _, terminated, truncated, _ = env.step(expert_action_idx)
            done = terminated or truncated

        winning_color = env.unwrapped.game.winning_color()
        if winning_color == Color.BLUE:
            wins += 1

        # Save chunk to disk every CHUNK_SIZE games, clear RAM
        if (game_i + 1) % CHUNK_SIZE == 0 and boards:
            chunk_path = os.path.join(data_dir, f"chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(
                chunk_path,
                boards=np.array(boards, dtype=np.float32),
                numerics=np.array(numerics, dtype=np.float32),
                actions=np.array(actions, dtype=np.int64),
                masks=np.array(masks, dtype=np.float32),
                values=np.array(values, dtype=np.float32),
            )
            elapsed = time.time() - t0
            rate = (game_i + 1) / elapsed
            print(f"  Game {game_i+1}/{num_games} | "
                  f"{total_decisions} decisions | "
                  f"chunk {chunk_idx} saved | "
                  f"{rate:.0f} games/s | "
                  f"Win rate: {wins/(game_i+1)*100:.1f}%")
            boards, numerics, actions, masks, values = [], [], [], [], []
            chunk_idx += 1

    # Save remaining
    if boards:
        chunk_path = os.path.join(data_dir, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            boards=np.array(boards, dtype=np.float32),
            numerics=np.array(numerics, dtype=np.float32),
            actions=np.array(actions, dtype=np.int64),
            masks=np.array(masks, dtype=np.float32),
            values=np.array(values, dtype=np.float32),
        )
        chunk_idx += 1

    env.close()
    elapsed = time.time() - t0

    print(f"\n  Done: {total_decisions} decisions from {num_games} games")
    print(f"  Saved {chunk_idx} chunks to {data_dir}")
    print(f"  VFP win rate: {wins/num_games*100:.1f}%")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return total_decisions


class ChunkDataset(Dataset):
    """Dataset for a single chunk — lightweight, stays in RAM."""
    def __init__(self, chunk_path):
        d = np.load(chunk_path)
        self.boards = th.tensor(d["boards"], dtype=th.float32)
        self.numerics = th.tensor(d["numerics"], dtype=th.float32)
        self.actions = th.tensor(d["actions"], dtype=th.long)
        self.masks = th.tensor(d["masks"], dtype=th.bool)
        self.values = th.tensor(d["values"], dtype=th.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.numerics[idx],
            self.actions[idx],
            self.masks[idx],
            self.values[idx],
        )


def train_bc(data_dir):
    """Train MaskablePPO policy with behavioral cloning from disk chunks."""
    print(f"\n{'='*60}")
    print(f"BEHAVIORAL CLONING TRAINING")
    print(f"{'='*60}")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Create a dummy env to get observation/action spaces
    dummy_env = gym.make(
        "catanatron_gym:catanatron-v1",
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "enemies": [ValueFunctionPlayer(Color.RED)],
            "representation": "mixed",
            "normalized": True,
        },
    )
    dummy_env = ActionMasker(dummy_env, mask_fn)

    # Create MaskablePPO with exact v28 architecture
    policy_kwargs = dict(
        activation_fn=th.nn.LeakyReLU,
        net_arch=NET_ARCH,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(cnn_arch=CNN_ARCH, features_dim=FEATURES_DIM),
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        dummy_env,
        policy_kwargs=policy_kwargs,
        learning_rate=BC_LR,
        device=device,
        seed=SEED,
    )
    dummy_env.close()

    policy = model.policy
    policy.train()

    chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    num_chunks = len(chunk_files)

    optimizer = th.optim.Adam(policy.parameters(), lr=BC_LR)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BC_EPOCHS)

    print(f"  Chunks:  {num_chunks}")
    print(f"  Epochs:  {BC_EPOCHS}")
    print(f"  Batch:   {BC_BATCH_SIZE}")
    print(f"  LR:      {BC_LR}")
    print(f"  (Streaming chunk-by-chunk to stay under RAM limit)\n")

    best_accuracy = 0.0
    for epoch in range(BC_EPOCHS):
        epoch_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_val_loss = 0.0
        correct = 0
        total = 0

        # Shuffle chunk order each epoch
        chunk_order = list(range(num_chunks))
        random.shuffle(chunk_order)

        for ci, chunk_i in enumerate(chunk_order):
            dataset = ChunkDataset(chunk_files[chunk_i])
            loader = DataLoader(dataset, batch_size=BC_BATCH_SIZE, shuffle=True,
                                num_workers=0, pin_memory=True)

            for boards, numerics, expert_actions, action_masks, value_targets in loader:
                boards = boards.to(device)
                numerics = numerics.to(device)
                expert_actions = expert_actions.to(device)
                action_masks = action_masks.to(device)
                value_targets = value_targets.to(device)

                obs = {"board": boards, "numeric": numerics}

                features = policy.extract_features(obs, policy.features_extractor)
                pi_features, vf_features = policy.mlp_extractor(features)
                action_logits = policy.action_net(pi_features)
                value_pred = policy.value_net(vf_features).squeeze(-1)

                action_logits[~action_masks] = float("-inf")

                bc_loss = F.cross_entropy(action_logits, expert_actions)
                val_loss = F.mse_loss(value_pred, value_targets)
                loss = bc_loss + VALUE_LOSS_WEIGHT * val_loss

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                with th.no_grad():
                    predicted = action_logits.argmax(dim=-1)
                    correct += (predicted == expert_actions).sum().item()
                    total += expert_actions.size(0)

                epoch_loss += loss.item() * expert_actions.size(0)
                epoch_bc_loss += bc_loss.item() * expert_actions.size(0)
                epoch_val_loss += val_loss.item() * expert_actions.size(0)

            del dataset, loader  # free RAM before loading next chunk

        scheduler.step()

        accuracy = correct / total
        print(f"  Epoch {epoch+1}/{BC_EPOCHS} | "
              f"Loss: {epoch_loss/total:.4f} (BC: {epoch_bc_loss/total:.4f}, Val: {epoch_val_loss/total:.4f}) | "
              f"Accuracy: {accuracy*100:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    print(f"\n  Best accuracy: {best_accuracy*100:.1f}%")

    # Save as SB3-compatible model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_bc")
    model.save(model_path)
    print(f"  Model saved: {model_path}.zip")

    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.cuda.manual_seed_all(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "bc_data")

    t0 = time.time()

    # Step 1: Collect expert data (saves chunks to disk)
    total = collect_expert_data(NUM_GAMES, data_dir)

    # Step 2: Train BC (loads chunks from disk)
    model = train_bc(data_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BC PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Samples:    {total}")
    print(f"  Output:     model_bc.zip")
    print(f"  Next:       python train_ppo_agent_34.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
