"""
train.py
========
Main training script for the FlyGM + A2C navigation agent.
"""

from __future__ import annotations

import os
import time
import json
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from .environment import GridMazeEnv
from .graph_builder import build_navigation_connectome, graph_stats
from .flygm_network import FlyGMNetwork
from .a2c_agent import A2CAgent, Rollout


@dataclass
class Config:
    # Environment
    maze_size: int = 15
    obs_radius: int = 2
    max_episode_steps: int = 400

    # Parallelism
    n_envs: int = 8
    n_steps: int = 16

    # Training budget
    total_timesteps: int = 500_000

    # FlyGM graph
    n_afferent: int = 48
    n_intrinsic: int = 144
    n_efferent: int = 24
    graph_seed: int = 42

    # FlyGM network
    enc_dim: int = 32
    channel_dim: int = 32
    desc_dim: int = 16

    # A2C
    lr: float = 3e-4
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5

    # Logging / saving
    log_interval_steps: int = 10_000
    eval_interval_steps: int = 50_000
    n_eval_episodes: int = 20
    output_dir: str = "outputs"
    seed: int = 0


@torch.no_grad()
def evaluate(
    network: FlyGMNetwork,
    cfg: Config,
    device: torch.device,
    n_episodes: int = 20,
) -> dict:
    """Run the greedy policy on fresh random mazes."""
    env = GridMazeEnv(
        maze_size=cfg.maze_size,
        obs_radius=cfg.obs_radius,
        max_steps=cfg.max_episode_steps,
    )

    successes: list[bool] = []
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    cells_explored: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep + 10000)
        H = network.init_hidden(1)
        ep_reward = 0.0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits, _, H = network(obs_t, H)
            action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        successes.append(info["success"])
        ep_rewards.append(ep_reward)
        ep_lengths.append(info["steps"])

        open_cells = int((env.maze == 0).sum())
        cells_explored.append(info["cells_visited"] / max(1, open_cells) * 100)

    return {
        "eval/success_rate_pct": float(np.mean(successes) * 100),
        "eval/mean_episode_reward": float(np.mean(ep_rewards)),
        "eval/mean_episode_length": float(np.mean(ep_lengths)),
        "eval/mean_cells_explored_pct": float(np.mean(cells_explored)),
    }


def plot_metrics(history: list[dict], output_dir: str) -> None:
    """Save training curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("FlyGM + A2C Navigation — Training Curves", fontsize=14)

    def _plot(ax, key, label, colour):
        vals = [m.get(key) for m in history if key in m]
        xs = [m["step"] for m in history if key in m]
        ax.plot(xs, vals, colour, linewidth=1.5)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "train/mean_episode_reward", "Mean Episode Reward", "steelblue")
    _plot(axes[0, 1], "eval/success_rate_pct", "Success Rate (%)", "forestgreen")
    _plot(axes[0, 2], "eval/mean_episode_length", "Mean Episode Length", "darkorange")
    _plot(axes[1, 0], "update/total_loss", "Total Loss", "crimson")
    _plot(axes[1, 1], "update/entropy", "Policy Entropy", "purple")
    _plot(axes[1, 2], "eval/mean_cells_explored_pct", "Cells Explored (%)", "teal")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved training curves → {out_path}")


def train(cfg: Config | None = None) -> tuple[FlyGMNetwork, list[dict]]:
    """Full training run."""
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build connectome graph
    W, partition = build_navigation_connectome(
        n_afferent=cfg.n_afferent,
        n_intrinsic=cfg.n_intrinsic,
        n_efferent=cfg.n_efferent,
        seed=cfg.graph_seed,
    )
    stats = graph_stats(W, partition)

    print("=" * 60)
    print("  FlyGM + A2C  |  Unknown-Environment Navigation")
    print("=" * 60)
    print(f"  Maze size        : {cfg.maze_size}×{cfg.maze_size}")
    print(f"  Neurons total    : {stats['n_total_neurons']}")
    print(f"  Fixed synapses   : {stats['n_nonzero_synapses']:,}")
    print(f"  Graph density    : {stats['graph_density_pct']} %")
    print(f"  Exc : Inh ratio  : {stats['exc_inh_ratio']}")

    # Instantiate network and agent
    obs_dim = 3 * (2 * cfg.obs_radius + 1) ** 2  # 75
    n_actions = 4

    network = FlyGMNetwork(
        obs_dim=obs_dim,
        n_actions=n_actions,
        W=W,
        partition=partition,
        enc_dim=cfg.enc_dim,
        channel_dim=cfg.channel_dim,
        desc_dim=cfg.desc_dim,
    ).to(device)

    agent = A2CAgent(
        network=network,
        lr=cfg.lr,
        gamma=cfg.gamma,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
    )

    print(f"  Trainable params : {network.count_parameters():,}")
    print("=" * 60)

    # Create parallel training environments
    envs = [
        GridMazeEnv(
            maze_size=cfg.maze_size,
            obs_radius=cfg.obs_radius,
            max_steps=cfg.max_episode_steps,
        )
        for _ in range(cfg.n_envs)
    ]

    obs_list = [env.reset(seed=cfg.seed + i)[0] for i, env in enumerate(envs)]
    obs_np = np.stack(obs_list, axis=0)  # (B, obs_dim)
    hiddens = network.init_hidden(cfg.n_envs).to(device)

    ep_reward = np.zeros(cfg.n_envs)
    ep_length = np.zeros(cfg.n_envs, dtype=int)

    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_success = deque(maxlen=100)

    total_steps = 0
    next_log_step = cfg.log_interval_steps
    next_eval_step = cfg.eval_interval_steps
    metrics_history: list[dict] = []
    start_time = time.time()

    pbar = tqdm(
        total=cfg.total_timesteps,
        unit="step",
        desc="Training",
        dynamic_ncols=True,
    )

    while total_steps < cfg.total_timesteps:
        # Collect T‑step rollout
        rollout_obs: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []

        hidden_0 = hiddens.clone()

        for _ in range(cfg.n_steps):
            obs_t = torch.FloatTensor(obs_np).to(device)

            actions_np, values_t, new_hiddens = agent.act(obs_t, hiddens)

            next_obs_list: list[np.ndarray] = []
            rewards_list: list[float] = []
            dones_list: list[float] = []

            for i, (env, a) in enumerate(zip(envs, actions_np)):
                next_obs, reward, terminated, truncated, info = env.step(int(a))
                done = terminated or truncated

                next_obs_list.append(next_obs)
                rewards_list.append(reward)
                dones_list.append(float(done))

                ep_reward[i] += reward
                ep_length[i] += 1

                if done:
                    recent_rewards.append(ep_reward[i])
                    recent_lengths.append(ep_length[i])
                    recent_success.append(float(info.get("success", False)))
                    ep_reward[i] = 0.0
                    ep_length[i] = 0

                    # Reset hidden state for finished environment
                    new_hiddens[i] = network.init_hidden(1).squeeze(0)
                    # Reset environment
                    next_obs_list[-1] = env.reset()[0]

            rollout_obs.append(obs_t)
            rollout_actions.append(torch.LongTensor(actions_np).to(device))
            rollout_rewards.append(torch.FloatTensor(rewards_list).to(device))
            rollout_dones.append(torch.FloatTensor(dones_list).to(device))

            obs_np = np.stack(next_obs_list, axis=0)
            hiddens = new_hiddens
            total_steps += cfg.n_envs
            pbar.update(cfg.n_envs)

        # Bootstrap value
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).to(device)
            _, last_vals, _ = network(obs_t, hiddens)
            last_vals = last_vals.squeeze(-1)

        rollout = Rollout(
            obs=torch.stack(rollout_obs, dim=0),
            actions=torch.stack(rollout_actions, dim=0),
            rewards=rollout_rewards,
            dones=rollout_dones,
            last_values=last_vals.detach(),
            hidden_0=hidden_0.detach(),
        )

        # Gradient update
        update_metrics = agent.update(rollout)

        # Logging
        if total_steps >= next_log_step and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            fps = int(total_steps / elapsed)
            record = {
                "step": total_steps,
                "train/mean_episode_reward": float(np.mean(recent_rewards)),
                "train/mean_episode_length": float(np.mean(recent_lengths)),
                "train/success_rate_pct": float(np.mean(recent_success) * 100),
                "train/fps": fps,
                **{f"update/{k}": v for k, v in update_metrics.items()},
            }

            if total_steps >= next_eval_step:
                eval_metrics = evaluate(network, cfg, device, cfg.n_eval_episodes)
                record.update(eval_metrics)
                next_eval_step += cfg.eval_interval_steps

                print(
                    f"\n  [eval @ {total_steps:,}]  "
                    f"success={eval_metrics['eval/success_rate_pct']:.1f}%  "
                    f"reward={eval_metrics['eval/mean_episode_reward']:.3f}  "
                    f"length={eval_metrics['eval/mean_episode_length']:.1f}  "
                    f"explored={eval_metrics['eval/mean_cells_explored_pct']:.1f}%"
                )

            metrics_history.append(record)

            pbar.set_postfix({
                "rew": f"{np.mean(recent_rewards):.2f}",
                "succ": f"{np.mean(recent_success)*100:.1f}%",
                "loss": f"{update_metrics['total_loss']:.3f}",
            })

            next_log_step += cfg.log_interval_steps

    pbar.close()

    # Save model and metrics
    model_path = os.path.join(cfg.output_dir, "flygm_navigation.pt")
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "config": asdict(cfg),
            "graph_stats": stats,
            "total_steps": total_steps,
        },
        model_path,
    )
    print(f"\nModel saved → {model_path}")

    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    plot_metrics(metrics_history, cfg.output_dir)

    return network, metrics_history


if __name__ == "__main__":
    train(Config())