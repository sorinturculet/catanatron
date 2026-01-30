"""
GNN-based Feature Extractor for Catan (v6)

Uses Graph Attention Networks (GAT) to process the Catan board as a graph
with 54 nodes (settlement/city locations) and 72 edges (road locations).

Key design decisions:
- Uses ONLY 54 land nodes (IDs 0-53) where settlements/cities can be built
- Uses 72 land edges where roads can be built
- Does NOT include water nodes (IDs 54+) which exist in node_map for tensor geometry only
- Edge features encode road ownership directly
- Attention learns which neighbors are strategically important

Key advantages over CNN:
- Operates on actual graph topology, not 2D image approximation
- 25x fewer parameters than CNN with stronger inductive bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, NUM_EDGES
from catanatron_gym.board_tensor_features import get_node_and_edge_maps


class GraphAttentionPooling(nn.Module):
    """
    Attention-based pooling for graph features.
    Learns to weight nodes by strategic importance.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, node_dim] node features
            batch: [num_nodes] batch assignment for each node
        Returns:
            pooled: [batch_size, 2 * node_dim] (attention + max pooled)
        """
        if batch is None:
            # Single graph case
            attn_scores = self.attention(x)  # [num_nodes, 1]
            attn_weights = F.softmax(attn_scores, dim=0)
            attn_pooled = (x * attn_weights).sum(dim=0)  # [node_dim]
            max_pooled = x.max(dim=0)[0]  # [node_dim]
            return torch.cat([attn_pooled, max_pooled], dim=-1).unsqueeze(0)

        # Batched case
        batch_size = batch.max().item() + 1
        node_dim = x.shape[-1]

        attn_scores = self.attention(x)  # [total_nodes, 1]

        # Compute softmax per graph in batch
        pooled_features = []
        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]  # [nodes_in_graph, node_dim]
            scores_b = attn_scores[mask]  # [nodes_in_graph, 1]

            attn_weights = F.softmax(scores_b, dim=0)
            attn_pooled = (x_b * attn_weights).sum(dim=0)  # [node_dim]
            max_pooled = x_b.max(dim=0)[0]  # [node_dim]

            pooled_features.append(torch.cat([attn_pooled, max_pooled], dim=-1))

        return torch.stack(pooled_features, dim=0)  # [batch_size, 2 * node_dim]


class CatanGNNFeatureExtractor(BaseFeaturesExtractor):
    """
    GNN-based feature extractor for Catan board.

    Architecture:
    1. Extract node features from board tensor (54 land nodes, IDs 0-53)
    2. Extract edge features (72 road edges, road ownership per player)
    3. 3-layer GATv2 with edge features
    4. Attention + max pooling -> fixed-size vector
    5. Concatenate with numeric features
    6. Final projection to features_dim

    Args:
        observation_space: Gym Dict space with "board" and "numeric"
        features_dim: Output dimension (default: 512)
        gnn_hidden: Hidden dimension per attention head (default: 32)
        gnn_heads: Number of attention heads (default: 4)
        gnn_layers: Number of GAT layers (default: 3)
        dropout: Dropout rate in attention (default: 0.1)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        gnn_hidden: int = 32,
        gnn_heads: int = 4,
        gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.gnn_hidden = gnn_hidden
        self.gnn_heads = gnn_heads
        self.gnn_layers = gnn_layers

        # Get board dimensions
        board_shape = observation_space["board"].shape
        self.num_channels = board_shape[0]  # channels first
        self.board_height = board_shape[1]
        self.board_width = board_shape[2]

        # Determine number of players from channels
        # channels = 2*n_players + 5 (resources) + 1 (robber) + 6 (ports)
        self.num_players = (self.num_channels - 12) // 2

        # Node input features: same as board channels at each node position
        self.node_input_dim = self.num_channels

        # Edge features: road ownership per player + buildable flag
        self.edge_dim = self.num_players + 1

        # Build GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = self.node_input_dim
        for i in range(gnn_layers):
            # Concat heads for all but last layer
            concat = (i < gnn_layers - 1)

            self.convs.append(GATv2Conv(
                in_channels=in_dim,
                out_channels=gnn_hidden,
                heads=gnn_heads,
                concat=concat,
                dropout=dropout,
                edge_dim=self.edge_dim,
                add_self_loops=True,
            ))

            out_dim = gnn_hidden * gnn_heads if concat else gnn_hidden
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        # Final node dimension after GAT layers
        final_node_dim = gnn_hidden  # Last layer doesn't concat

        # Graph pooling
        self.pool = GraphAttentionPooling(final_node_dim, hidden_dim=64)
        pooled_dim = 2 * final_node_dim  # attention + max

        # Numeric features dimension
        self.n_numeric = observation_space["numeric"].shape[0]

        # Final projection
        self.linear = nn.Sequential(
            nn.Linear(pooled_dim + self.n_numeric, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(),
        )

        # Pre-compute static graph structure and coordinate mappings
        self._build_graph_and_coord_maps()

    def _build_graph_and_coord_maps(self):
        """
        Build edge_index tensor and coordinate mappings for the 54 land nodes.

        Critical: We use ONLY land nodes (0-53) and land edges (72 edges).
        The node_map from board_tensor_features contains 66 nodes (54 land + 12 water),
        but water nodes are only for tensor geometry and have no game meaning.
        """
        # Get coordinate mappings from board_tensor_features
        node_map, edge_map = get_node_and_edge_maps()

        # Get the 72 land edges (connects nodes 0-53 only)
        land_edges = get_edges(frozenset(range(NUM_NODES)))

        # Verify we have the expected structure
        assert len(land_edges) == NUM_EDGES, f"Expected {NUM_EDGES} edges, got {len(land_edges)}"

        # Build node coordinate tensor for land nodes only (0-53)
        # These coordinates tell us where to sample from the board tensor
        node_coords = torch.zeros(NUM_NODES, 2, dtype=torch.long)
        for node_id in range(NUM_NODES):
            if node_id in node_map:
                x, y = node_map[node_id]
                node_coords[node_id, 0] = x
                node_coords[node_id, 1] = y
            else:
                # This should not happen for land nodes 0-53
                raise ValueError(f"Land node {node_id} not found in node_map")
        self.register_buffer('node_coords', node_coords)

        # Build edge index tensor (bidirectional)
        # Node IDs 0-53 are already contiguous, so we can use them directly as indices
        edge_list = []
        for (src, dst) in land_edges:
            edge_list.append([src, dst])
            edge_list.append([dst, src])  # Undirected

        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # [2, 144]
        self.register_buffer('edge_index', edge_index)

        # Build edge coordinate tensor for feature extraction
        # Each edge has a position in the board tensor where road info is stored
        edge_coords = torch.zeros(NUM_EDGES, 2, dtype=torch.long)
        for i, (src, dst) in enumerate(land_edges):
            # edge_map has both (src, dst) and (dst, src) with same coords
            if (src, dst) in edge_map:
                x, y = edge_map[(src, dst)]
            elif (dst, src) in edge_map:
                x, y = edge_map[(dst, src)]
            else:
                raise ValueError(f"Edge ({src}, {dst}) not found in edge_map")
            edge_coords[i, 0] = x
            edge_coords[i, 1] = y
        self.register_buffer('edge_coords', edge_coords)

        # Store edge list for reference
        self.land_edges = land_edges

    def _extract_node_features(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract per-node features from board tensor for 54 land nodes.

        Args:
            board_tensor: [B, C, H, W] board observation (H=21, W=11)
        Returns:
            node_features: [B, 54, C] features for each land node
        """
        B, C, H, W = board_tensor.shape

        # Use advanced indexing to sample all node positions at once
        # node_coords: [54, 2] -> x coords and y coords
        x_coords = self.node_coords[:, 0]  # [54]
        y_coords = self.node_coords[:, 1]  # [54]

        # board_tensor is [B, C, H, W] where H=21 (width), W=11 (height)
        # The tensor uses (x, y) indexing: tensor[:, :, x, y]
        node_features = board_tensor[:, :, x_coords, y_coords]  # [B, C, 54]
        node_features = node_features.permute(0, 2, 1)  # [B, 54, C]

        return node_features

    def _extract_edge_features(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract per-edge features (road ownership) from board tensor for 72 land edges.

        Args:
            board_tensor: [B, C, H, W] board observation
        Returns:
            edge_features: [B, 144, edge_dim] features for each directed edge (72*2)
        """
        B, C, H, W = board_tensor.shape

        # Edge coordinates for the 72 land edges
        x_coords = self.edge_coords[:, 0]  # [72]
        y_coords = self.edge_coords[:, 1]  # [72]

        # Road channels are at indices 1, 3, 5, 7, ... (odd indices, one per player)
        # Channel layout: [P0_settle, P0_road, P1_settle, P1_road, ..., resources, robber, ports]
        road_channels = list(range(1, 2 * self.num_players, 2))

        # Extract road ownership for each edge
        # board_tensor[:, road_channels, :, :] -> [B, num_players, H, W]
        # Then index with edge coords -> [B, num_players, 72]
        road_features = board_tensor[:, road_channels, :, :][:, :, x_coords, y_coords]
        road_features = road_features.permute(0, 2, 1)  # [B, 72, num_players]

        # Add buildable flag: 1 if no road exists on this edge
        has_road = road_features.sum(dim=-1, keepdim=True) > 0  # [B, 72, 1]
        buildable = (~has_road).float()

        # Concatenate: [B, 72, num_players + 1]
        edge_features = torch.cat([road_features, buildable], dim=-1)

        # Duplicate for bidirectional edges (src->dst and dst->src have same features)
        # edge_index has shape [2, 144] with both directions
        edge_features = edge_features.repeat(1, 2, 1)  # [B, 144, edge_dim]

        return edge_features

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass through GNN feature extractor.

        Args:
            observations: Dict with "board" [B, C, H, W] and "numeric" [B, N]
        Returns:
            features: [B, features_dim] extracted features
        """
        board = observations["board"]
        numeric = observations["numeric"]

        B = board.shape[0]
        device = board.device

        # Extract graph features from board tensor
        node_features = self._extract_node_features(board)  # [B, 54, C]
        edge_features = self._extract_edge_features(board)  # [B, 144, edge_dim]

        # Process each sample through GNN
        # We use PyG's batching to process all samples efficiently
        data_list = []
        for b in range(B):
            data = Data(
                x=node_features[b],  # [54, C]
                edge_index=self.edge_index,  # [2, 144]
                edge_attr=edge_features[b],  # [144, edge_dim]
            )
            data_list.append(data)

        # Batch graphs together
        batch = Batch.from_data_list(data_list)

        # Move batch to correct device
        batch = batch.to(device)

        # Forward through GAT layers
        x = batch.x
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, edge_attr=batch.edge_attr)
            x_new = norm(x_new)
            x_new = F.leaky_relu(x_new)

            # Residual connection if dimensions match
            if x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new

        # Pool graph features
        pooled = self.pool(x, batch.batch)  # [B, 2 * gnn_hidden]

        # Concatenate with numeric features and project
        combined = torch.cat([pooled, numeric], dim=-1)

        return self.linear(combined)
