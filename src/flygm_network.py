"""
flygm_network.py
================
Fly‑connectomic Graph Model (FlyGM) adapted as an Actor‑Critic policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlyGMNetwork(nn.Module):
    """
    Connectome‑structured Actor‑Critic network.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension (e.g., 75).
    n_actions : int
        Number of discrete actions.
    W : torch.Tensor, shape (n_total, n_total)
        Fixed signed synaptic weight matrix.
    partition : dict
        Slices for afferent/efferent populations.
    enc_dim : int
        Encoder output dimension.
    channel_dim : int
        Neuron state channel dimension C.
    desc_dim : int
        Dimension of per‑neuron intrinsic descriptors η.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        W: torch.Tensor,
        partition: dict,
        enc_dim: int = 32,
        channel_dim: int = 32,
        desc_dim: int = 16,
    ) -> None:
        super().__init__()

        self.C = channel_dim
        self.D = desc_dim
        self.n_total: int = partition["n_total"]

        aff_sl: slice = partition["afferent_slice"]
        eff_sl: slice = partition["efferent_slice"]
        self.aff_sl = aff_sl
        self.eff_sl = eff_sl

        self.n_afferent: int = aff_sl.stop - aff_sl.start
        self.n_efferent: int = eff_sl.stop - eff_sl.start
        efferent_flat: int = self.n_efferent * channel_dim

        # Fixed structural prior (not trainable)
        self.register_buffer("W", W)

        # Trainable per‑neuron intrinsic descriptors
        self.neuron_descriptors = nn.Parameter(
            torch.randn(self.n_total, desc_dim) * 0.01
        )

        # Step 1: Encoder
        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, enc_dim),
            nn.ReLU(),
        )

        # Step 2: Afferent gating
        self.afferent_gate = nn.Linear(channel_dim + enc_dim, channel_dim)

        # Step 4: Shared state‑update MLP f_ψ
        self.update_mlp = nn.Sequential(
            nn.Linear(channel_dim + desc_dim, channel_dim * 2),
            nn.ELU(),
            nn.Linear(channel_dim * 2, channel_dim),
            nn.Tanh(),
        )

        # Step 5a: Actor decoder
        self.actor = nn.Sequential(
            nn.Linear(efferent_flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Step 5b: Critic decoder
        self.critic = nn.Sequential(
            nn.Linear(efferent_flat, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if "critic" in name else 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Zero initial hidden state H_0."""
        return torch.zeros(
            batch_size, self.n_total, self.C, device=self.W.device
        )

    def forward(
        self, obs: torch.Tensor, H: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One‑step forward pass.

        Returns
        -------
        action_logits : (B, n_actions)
        value         : (B, 1)
        H_new         : (B, n_total, C)
        """
        B = obs.shape[0]

        # Step 1: Encode observation
        x_enc = self.encoder(obs)  # (B, enc_dim)

        # Step 2: Afferent injection (clone to avoid modifying input)
        H = H.clone()
        H_aff = H[:, self.aff_sl, :]  # (B, n_aff, C)
        x_enc_exp = x_enc.unsqueeze(1).expand(-1, self.n_afferent, -1)
        gate_in = torch.cat([H_aff, x_enc_exp], dim=-1)
        H[:, self.aff_sl, :] = torch.tanh(self.afferent_gate(gate_in))

        # Step 3: Synaptic aggregation M = W @ H
        M = torch.einsum("vu, buc -> bvc", self.W, H)  # (B, n_total, C)

        # Step 4: Per‑neuron update
        eta = self.neuron_descriptors.unsqueeze(0).expand(B, -1, -1)
        update_in = torch.cat([M, eta], dim=-1)
        H_new = self.update_mlp(update_in)

        # Step 5: Decode from efferent neurons
        H_eff = H_new[:, self.eff_sl, :]  # (B, n_eff, C)
        H_eff_flat = H_eff.reshape(B, -1)

        action_logits = self.actor(H_eff_flat)
        value = self.critic(H_eff_flat)

        return action_logits, value, H_new

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_fixed_synapses(self) -> int:
        """Number of non‑zero entries in the fixed weight matrix W."""
        return int((self.W != 0).sum().item())