"""
a2c_agent.py
============
Advantage Actor‑Critic (A2C) training algorithm wrapping the FlyGM policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .flygm_network import FlyGMNetwork


@dataclass
class Rollout:
    """Container for one T‑step rollout across B parallel environments."""
    obs: torch.Tensor           # (T, B, obs_dim)
    actions: torch.Tensor       # (T, B)
    rewards: List[torch.Tensor] # T items, each (B,)
    dones: List[torch.Tensor]   # T items, each (B,)
    last_values: torch.Tensor   # (B,)
    hidden_0: torch.Tensor      # (B, n_total, C)


class A2CAgent:
    """
    Advantage Actor‑Critic agent using FlyGM as its policy network.

    Parameters
    ----------
    network : FlyGMNetwork
        The connectome‑structured actor‑critic.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor γ.
    value_coef : float
        Coefficient c_v for the critic loss.
    entropy_coef : float
        Coefficient c_e for the entropy bonus.
    max_grad_norm : float
        Maximum L2 norm for gradient clipping.
    """

    def __init__(
        self,
        network: FlyGMNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.network = network
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimiser = torch.optim.Adam(
            network.parameters(), lr=lr, eps=1e-5
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimiser,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=2000,
        )

        self._update_count: int = 0

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the current policy.

        Returns
        -------
        actions_np : np.ndarray (B,)
        values     : torch.Tensor (B,)
        new_hidden : torch.Tensor (B, n_total, C)
        """
        logits, values, new_hidden = self.network(obs, hidden)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return (
            actions.cpu().numpy(),
            values.squeeze(-1),
            new_hidden,
        )

    def _compute_returns(
        self,
        rewards: List[torch.Tensor],
        dones: List[torch.Tensor],
        last_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discounted n‑step returns via backward accumulation."""
        T = len(rewards)
        B = rewards[0].shape[0]
        device = last_values.device
        returns = torch.zeros(T, B, device=device)
        R = last_values.clone()

        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t])
            returns[t] = R

        return returns

    def update(self, rollout: Rollout) -> dict:
        """
        Perform one A2C gradient update from a completed rollout.
        """
        T, B = rollout.obs.shape[0], rollout.obs.shape[1]
        device = rollout.obs.device

        # Compute discounted returns
        returns = self._compute_returns(
            rollout.rewards, rollout.dones, rollout.last_values
        )  # (T, B)

        # Re‑run network with gradient tracking (T‑BPTT)
        H = rollout.hidden_0.detach()
        logits_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []

        for t in range(T):
            logits_t, val_t, H = self.network(rollout.obs[t], H)
            H = H.detach()  # truncated BPTT
            logits_list.append(logits_t)
            values_list.append(val_t.squeeze(-1))

        logits_all = torch.stack(logits_list, dim=0)  # (T, B, n_actions)
        values_all = torch.stack(values_list, dim=0)  # (T, B)

        # Advantage Â_t = R_t − V(s_t)
        advantages = returns.detach() - values_all.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss
        dist = Categorical(logits=logits_all)
        log_probs = dist.log_prob(rollout.actions)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(values_all, returns.detach())

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        # Gradient step
        self.optimiser.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.network.parameters(), self.max_grad_norm
        )
        self.optimiser.step()
        self.lr_scheduler.step()

        self._update_count += 1

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item(),
            "mean_value": values_all.mean().item(),
            "advantage_std": advantages.std().item(),
            "update_count": self._update_count,
      }
