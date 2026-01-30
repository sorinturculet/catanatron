import os
import random
import glob
from typing import List

from catanatron import Color
from catanatron_experimental.machine_learning.players.ppo import PPOPlayer


class SelfPlayOpponentManager:
    """
    Manages opponent pool for self-play training.

    Maintains a pool of recent agent checkpoints and samples from them
    with preference for more recent (stronger) opponents.
    """

    def __init__(self, checkpoint_dir="./logs", pool_size=5, experiment_name="ppo_v8b"):
        """
        Initialize the self-play opponent manager.

        Args:
            checkpoint_dir: Directory containing checkpoints
            pool_size: Maximum number of checkpoints to keep in pool
            experiment_name: Experiment name for finding checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.pool_size = pool_size
        self.experiment_name = experiment_name
        self.opponent_checkpoints: List[str] = []

    def load_existing_checkpoints(self):
        """Load existing checkpoints from directory into the pool."""
        pattern = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_*_steps.zip")
        checkpoint_files = sorted(glob.glob(pattern))

        # Take only the most recent pool_size checkpoints
        self.opponent_checkpoints = checkpoint_files[-self.pool_size:]

        if self.opponent_checkpoints:
            print(f"📚 Loaded {len(self.opponent_checkpoints)} existing checkpoints into pool")
            for cp in self.opponent_checkpoints:
                print(f"   - {os.path.basename(cp)}")
        else:
            print(f"⚠️  No existing checkpoints found for {self.experiment_name}")

    def get_initial_opponent(self, base_model_path: str) -> PPOPlayer:
        """
        Load opponent from Stage 1 checkpoint (model_v8a.zip).

        Args:
            base_model_path: Path to the Stage 1 model

        Returns:
            PPOPlayer instance loaded from the model
        """
        print(f"🎮 Creating initial opponent from {base_model_path}")
        return PPOPlayer(Color.RED, base_model_path)

    def update_opponent_pool(self, current_checkpoint: str):
        """
        Add latest checkpoint to pool and maintain pool size.

        Args:
            current_checkpoint: Path to new checkpoint to add
        """
        self.opponent_checkpoints.append(current_checkpoint)

        # Remove oldest if pool is too large
        if len(self.opponent_checkpoints) > self.pool_size:
            oldest = self.opponent_checkpoints.pop(0)
            print(f"🗑️  Removed oldest checkpoint from pool: {os.path.basename(oldest)}")

        print(f"💾 Added checkpoint to pool: {os.path.basename(current_checkpoint)}")
        print(f"📊 Pool size: {len(self.opponent_checkpoints)}/{self.pool_size}")

    def sample_opponent(self) -> PPOPlayer:
        """
        Sample opponent with preference for recent checkpoints.

        Uses weighted sampling: 60% probability for most recent checkpoint,
        40% distributed over older checkpoints.

        Returns:
            PPOPlayer instance loaded from sampled checkpoint
        """
        if not self.opponent_checkpoints:
            raise ValueError("No checkpoints in opponent pool! Call update_opponent_pool first.")

        if len(self.opponent_checkpoints) == 1:
            # Only one checkpoint available
            checkpoint = self.opponent_checkpoints[0]
        else:
            # Weighted sampling: prefer recent checkpoints
            # Most recent gets 60%, others share remaining 40%
            n = len(self.opponent_checkpoints)
            weights = [0.4 / (n - 1) for _ in range(n - 1)] + [0.6]
            checkpoint = random.choices(self.opponent_checkpoints, weights=weights)[0]

        print(f"🎲 Sampled opponent: {os.path.basename(checkpoint)}")
        return PPOPlayer(Color.RED, checkpoint)

    def get_pool_info(self) -> str:
        """Get string representation of current pool state."""
        if not self.opponent_checkpoints:
            return "Opponent pool: empty"

        info = f"Opponent pool ({len(self.opponent_checkpoints)}/{self.pool_size} checkpoints):\n"
        for i, cp in enumerate(self.opponent_checkpoints):
            info += f"  {i+1}. {os.path.basename(cp)}\n"
        return info
