import math
import torch as th
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for board positions.
    Adds learnable position embeddings to help attention understand spatial relationships.
    """
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.d_model = d_model
        # Learnable position embeddings for each board position
        self.pos_embedding = nn.Parameter(th.randn(1, height * width, d_model) * 0.02)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pos_embedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for spatial reasoning on board features.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.out_proj(out)

        return out


class CustomCNNWithAttention(BaseFeaturesExtractor):
    """
    Custom CNN with self-attention for strategic board reasoning.

    Architecture:
    1. CNN layers extract local features from board
    2. Self-attention allows global reasoning across all board positions
    3. Features are combined with numeric observations for final output

    :param observation_space: (gym.Space)
    :param cnn_arch: List of integers specifying the number of filters in each Conv layer.
    :param features_dim: (int) Number of features extracted.
    :param num_attention_heads: Number of attention heads (default: 4)
    :param attention_dropout: Dropout rate in attention (default: 0.1)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_arch,
        features_dim: int = 512,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super(CustomCNNWithAttention, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["board"].shape[0]
        board_height = observation_space["board"].shape[1]
        board_width = observation_space["board"].shape[2]

        # CNN layers (same as CustomCNN, but without flatten)
        cnn_layers = []
        in_channels = n_input_channels
        for out_channels in cnn_arch:
            cnn_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # Get CNN output dimensions
        cnn_out_channels = cnn_arch[-1]  # Last CNN output channels (512)
        num_positions = board_height * board_width  # 11 * 11 = 121

        # Positional encoding for board positions
        self.pos_encoding = PositionalEncoding2D(cnn_out_channels, board_height, board_width)

        # Self-attention layer
        self.attention = MultiHeadSelfAttention(
            embed_dim=cnn_out_channels,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )

        # Layer norm after attention (helps with training stability)
        self.layer_norm = nn.LayerNorm(cnn_out_channels)

        # Compute flattened size after attention
        n_flatten = cnn_out_channels * num_positions

        # Combine with numeric features
        n_numeric_features = observation_space["numeric"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_numeric_features, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        # CNN feature extraction: [B, C, H, W]
        board_features = self.cnn(observations["board"])
        B, C, H, W = board_features.shape

        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        board_features = board_features.flatten(2).transpose(1, 2)  # [B, 121, 512]

        # Add positional encoding
        board_features = self.pos_encoding(board_features)

        # Self-attention with residual connection
        attended_features = self.attention(board_features)
        board_features = self.layer_norm(board_features + attended_features)  # Residual + LayerNorm

        # Flatten: [B, 121, 512] -> [B, 121*512]
        board_features = board_features.flatten(1)

        # Concatenate with numeric features
        concatenated_tensor = th.cat([board_features, observations["numeric"]], dim=1)

        return self.linear(concatenated_tensor)


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the board observations.
    :param observation_space: (gym.Space)
    :param cnn_arch: List of integers specifying the number of filters in each Conv layer.
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_arch,
        features_dim: int = 256,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["board"].shape[0]

        layers = []
        in_channels = n_input_channels
        for out_channels in cnn_arch:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute the number of features after CNN
        with th.no_grad():
            sample_board = th.as_tensor(
                observation_space.sample()["board"][None]
            ).float()
            n_flatten = self.cnn(sample_board).shape[1]

        n_numeric_features = observation_space["numeric"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_numeric_features, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict) -> th.Tensor:
        board_features = self.cnn(observations["board"])
        concatenated_tensor = th.cat([board_features, observations["numeric"]], dim=1)
        return self.linear(concatenated_tensor)
