"""
Mixture-of-Experts (MoE) Policy for Catan PPO (v20).

Architecture:
    Shared CNN backbone (from v19)  ->  512-dim features
        |
        +---> Gate Network:  Linear(512,128) -> LeakyReLU -> Linear(128,2) -> Softmax
        |         Output: [w_ows, w_expansion]
        |
        +---> OWS Policy MLP:       [2048,1024,512,256] -> Linear(256,290)   (init from v19)
        +---> Expansion Policy MLP: [2048,1024,512,256] -> Linear(256,290)   (init from v19)
        |
        +---> Value MLP (shared):   [2048,1024,512,256] -> Linear(256,1)     (init from v19)

    Final logits = w_ows * logits_ows + w_expansion * logits_expansion
    Action masking applied AFTER mixing (unchanged from v19).

Key design choices:
    - Gate learned end-to-end (not pre-computed heuristic)
    - Both expert heads initialised from v19 weights (both start competent)
    - Value head shared (value estimation is strategy-agnostic)
    - Reward pipeline UNCHANGED: production_aware_rewards + ValueWeightedWrapper 0.05
"""

from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.type_aliases import Schedule


# ---------------------------------------------------------------------------
# MoE MLP Extractor
# ---------------------------------------------------------------------------

class MoEMlpExtractor(nn.Module):
    """
    Replaces SB3's MlpExtractor inside the policy.

    Contains:
      - gate:        small network that outputs per-expert mixing weights
      - policy_nets: ModuleList of `num_experts` sequential MLPs
      - value_net:   shared value MLP

    forward() passes CNN features through unchanged as latent_pi so that
    MoEMaskablePolicy._get_action_dist_from_latent() can run the expert
    routing with gradients flowing through both expert heads and the gate.
    """

    def __init__(
        self,
        feature_dim: int,
        pi_arch: List[int],
        vf_arch: List[int],
        activation_fn: Type[nn.Module],
        num_experts: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        # ---- Gate network -------------------------------------------------
        # Small: feature_dim -> 128 -> num_experts (softmax)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 128),
            activation_fn(),
            nn.Linear(128, num_experts),
        )

        # ---- Expert policy MLPs -------------------------------------------
        self.policy_nets = nn.ModuleList()
        for _ in range(num_experts):
            layers: List[nn.Module] = []
            in_dim = feature_dim
            for out_dim in pi_arch:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation_fn())
                in_dim = out_dim
            self.policy_nets.append(nn.Sequential(*layers))

        # ---- Shared value MLP ---------------------------------------------
        vf_layers: List[nn.Module] = []
        in_dim = feature_dim
        for out_dim in vf_arch:
            vf_layers.append(nn.Linear(in_dim, out_dim))
            vf_layers.append(activation_fn())
            in_dim = out_dim
        self.value_net = nn.Sequential(*vf_layers)

        # latent_dim_pi: SB3 uses this to build the (unused) standard action_net
        # We pass raw features as latent_pi so this equals feature_dim
        self.latent_dim_pi = feature_dim

        # Actual output dim of each expert MLP (used for our action_nets)
        self.latent_dim_pi_expert = pi_arch[-1]  # e.g. 256

        # Output dim of value MLP (SB3 builds value_net = Linear(latent_dim_vf, 1))
        self.latent_dim_vf = vf_arch[-1]  # e.g. 256

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent_pi:  raw CNN features (512-dim) — expert routing is done
                        later in _get_action_dist_from_latent
            latent_vf:  value MLP output (256-dim)
        """
        return features, self.value_net(features)

    # SB3 newer versions call forward_actor / forward_critic separately
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features  # pass-through; routing done in policy

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def get_expert_outputs(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute gating weights and per-expert latent vectors.
        Called by MoEMaskablePolicy._get_action_dist_from_latent.

        Returns:
            gate_weights:   [B, num_experts]  softmax weights
            expert_latents: list of [B, latent_dim_pi_expert] tensors
        """
        gate_weights = F.softmax(self.gate(features), dim=-1)
        expert_latents = [net(features) for net in self.policy_nets]
        return gate_weights, expert_latents


# ---------------------------------------------------------------------------
# MoE Maskable Policy
# ---------------------------------------------------------------------------

class MoEMaskablePolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy with a Mixture-of-Experts action head.

    Overrides:
        _build_mlp_extractor() — creates MoEMlpExtractor
        _build()               — adds per-expert action_nets; rebuilds optimizer
        _get_action_dist_from_latent() — computes gated logit mixture

    Everything else (masking, value computation, PPO training loop) is
    inherited unchanged from MaskableActorCriticPolicy.
    """

    def __init__(self, *args, num_experts: int = 2, **kwargs):
        self.num_experts = num_experts
        self._last_gate_weights: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Create MoEMlpExtractor instead of SB3's standard MlpExtractor."""
        net_arch = self.net_arch
        # SB3 stores net_arch as [dict(...)] or dict(...)
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            net_arch = net_arch[0]

        pi_arch = net_arch.get("pi", [64, 64]) if isinstance(net_arch, dict) else [64, 64]
        vf_arch = net_arch.get("vf", [64, 64]) if isinstance(net_arch, dict) else [64, 64]

        self.mlp_extractor = MoEMlpExtractor(
            feature_dim=self.features_dim,
            pi_arch=pi_arch,
            vf_arch=vf_arch,
            activation_fn=self.activation_fn,
            num_experts=self.num_experts,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build full policy:
          1. Parent _build() creates MoEMlpExtractor, standard action_net (unused),
             value_net Linear(256,1), and optimizer.
          2. We add our per-expert action_nets.
          3. Rebuild optimizer so action_nets are included.
        """
        super()._build(lr_schedule)

        # Expert-specific action heads: Linear(expert_latent_dim, n_actions)
        expert_latent_dim = self.mlp_extractor.latent_dim_pi_expert  # 256
        self.action_nets = nn.ModuleList([
            nn.Linear(expert_latent_dim, self.action_space.n)
            for _ in range(self.num_experts)
        ])

        # Rebuild optimizer to include the new action_nets parameters
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
        latent_sde: Optional[torch.Tensor] = None,
    ):
        """
        Compute mixed action distribution.

        latent_pi is the raw CNN features (512-dim) passed through from
        MoEMlpExtractor.forward().  We run the gate and expert MLPs here
        so that gradients flow through everything.
        """
        gate_weights, expert_latents = self.mlp_extractor.get_expert_outputs(latent_pi)

        # Each expert produces 290 logits
        expert_logits = []
        for latent, action_net in zip(expert_latents, self.action_nets):
            expert_logits.append(action_net(latent))  # [B, 290]

        # Weighted sum: [B, num_experts, 290] * [B, num_experts, 1] -> [B, 290]
        stacked = torch.stack(expert_logits, dim=1)       # [B, num_experts, 290]
        weights = gate_weights.unsqueeze(-1)               # [B, num_experts, 1]
        mixed_logits = (stacked * weights).sum(dim=1)      # [B, 290]

        # Detach and store for WandB logging (no gradient leak)
        self._last_gate_weights = gate_weights.detach().cpu()

        return self.action_dist.proba_distribution(action_logits=mixed_logits)


# ---------------------------------------------------------------------------
# Weight loading: v19 -> MoE
# ---------------------------------------------------------------------------

def load_v19_weights_into_moe(moe_model: MaskablePPO, v19_path: str) -> None:
    """
    Transfer v19 weights into a freshly created MoE model.

    Mapping:
        features_extractor.*          -> features_extractor.*         (direct)
        mlp_extractor.value_net.*     -> mlp_extractor.value_net.*    (direct)
        mlp_extractor.policy_net.*    -> mlp_extractor.policy_nets.0.* AND .1.*
        action_net.*                  -> action_nets.0.* AND action_nets.1.*
        value_net.*                   -> value_net.*                  (direct)

    The gate (mlp_extractor.gate.*) stays randomly initialised so both
    experts get equal weight at the start of training.
    """
    from catanatron_experimental.machine_learning.custom_cnn import CustomCNN

    print(f"\n{'='*60}")
    print("Loading v19 weights into MoE model...")
    print(f"  Source: {v19_path}")
    print(f"{'='*60}")

    # Load v19 (CPU to avoid GPU memory doubling)
    v19 = MaskablePPO.load(
        v19_path,
        device="cpu",
        custom_objects={"features_extractor_class": CustomCNN},
    )
    v19_state = v19.policy.state_dict()
    moe_state = moe_model.policy.state_dict()

    new_state = {k: v.clone() for k, v in moe_state.items()}  # start from MoE defaults

    matched, skipped = 0, 0

    for v19_key, v19_val in v19_state.items():

        # 1. Direct copies: features_extractor, value_net (final linear), mlp_extractor.value_net
        if (
            v19_key.startswith("features_extractor.")
            or v19_key.startswith("value_net.")
            or v19_key.startswith("mlp_extractor.value_net.")
        ):
            if v19_key in new_state and new_state[v19_key].shape == v19_val.shape:
                new_state[v19_key] = v19_val.clone()
                matched += 1
            else:
                print(f"  SKIP (shape mismatch or missing): {v19_key}")
                skipped += 1

        # 2. Policy MLP -> both expert heads
        elif v19_key.startswith("mlp_extractor.policy_net."):
            suffix = v19_key[len("mlp_extractor.policy_net."):]
            for i in range(moe_model.policy.num_experts):
                moe_key = f"mlp_extractor.policy_nets.{i}.{suffix}"
                if moe_key in new_state and new_state[moe_key].shape == v19_val.shape:
                    new_state[moe_key] = v19_val.clone()
                    matched += 1
                else:
                    print(f"  SKIP: {v19_key} -> {moe_key} (shape or missing)")
                    skipped += 1

        # 3. Action net -> both expert action heads
        elif v19_key.startswith("action_net."):
            suffix = v19_key[len("action_net."):]
            for i in range(moe_model.policy.num_experts):
                moe_key = f"action_nets.{i}.{suffix}"
                if moe_key in new_state and new_state[moe_key].shape == v19_val.shape:
                    new_state[moe_key] = v19_val.clone()
                    matched += 1
                else:
                    print(f"  SKIP: {v19_key} -> {moe_key} (shape or missing)")
                    skipped += 1

        # 4. Gate stays randomly initialised — skip
        elif v19_key.startswith("mlp_extractor.gate."):
            pass  # intentionally not copied

        else:
            skipped += 1

    moe_model.policy.load_state_dict(new_state)

    # Verify features_dim match
    v19_features_dim = v19.policy.features_dim
    moe_features_dim = moe_model.policy.features_dim
    if v19_features_dim != moe_features_dim:
        raise ValueError(
            f"features_dim mismatch! v19={v19_features_dim}, MoE={moe_features_dim}. "
            f"Make sure features_dim in train_ppo_agent_20.py matches v19."
        )

    # Quick sanity: gate weights should be near-uniform after random init
    with torch.no_grad():
        device = next(moe_model.policy.mlp_extractor.gate.parameters()).device
        dummy = torch.zeros(1, moe_features_dim, device=device)
        gate_out = F.softmax(moe_model.policy.mlp_extractor.gate(dummy), dim=-1)
        print(f"\n  Gate weights after init (should be ~[0.5, 0.5]): {gate_out.cpu().numpy()}")

    print(f"\n  Weights matched/transferred: {matched}")
    print(f"  Skipped:                     {skipped}")
    print(f"  features_dim:                {moe_features_dim}")
    print(f"  Gate:                        randomly initialised (near-uniform)")
    print(f"{'='*60}\n")

    del v19  # free memory
