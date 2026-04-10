# FlyGM + A2C — Navigation Through Unknown Environments

> **Paper basis:** Jin et al. (2026) *Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly*, arXiv:2602.17997v2

---

## Project Overview

This project implements a **Fly-connectomic Graph Model (FlyGM)** policy network trained with **Advantage Actor-Critic (A2C)** reinforcement learning on a partially-observable grid-maze navigation task. The agent must navigate from a random start position to a goal in a freshly-generated random maze — without any map — using only a 5×5 local observation window.

The core contribution of the FlyGM paper is that a static biological wiring diagram (the *Drosophila* connectome) can be used as a structural prior for a neural network controller. Rather than hand-crafting an architecture, the graph topology itself acts as an *inductive bias* that improves sample efficiency and stability. This project adapts that idea to a robotics navigation problem.

---

## File Structure

```
flygm_navigation/
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── environment.py      # GridMazeEnv — Gymnasium-compatible maze
    ├── graph_builder.py    # Bio-inspired connectome graph construction
    ├── flygm_network.py    # FlyGMNetwork — connectome-structured actor-critic
    ├── a2c_agent.py        # A2C update logic + Rollout dataclass
    └── train.py            # Training loop, evaluation, logging, plotting
```

---

## Problem Definition (Step 1)

**Task:** Navigate a 15×15 random maze from a random start to a random goal under partial observability.

The maze is regenerated on every episode reset using iterative recursive backtracking, so the agent cannot memorise any layout. This forces it to learn a general navigation *strategy* rather than a map.

| Property | Value |
|---|---|
| Maze size | 15×15 grid |
| Observation | 5×5 local window, 3 channels (walls / agent / goal) = **75-dim** |
| Actions | 4 discrete (up / right / down / left) |
| Episode length | 400 steps maximum |
| Partial observability | Yes — agent sees only ±2 cells in each direction |

---

## State, Action, and Reward Spaces (Step 3)

### Observation Space

The agent receives a flattened 75-dimensional float32 vector comprising three stacked 5×5 channels:

- **Channel 0 — Walls:** 1.0 for wall cells, 0.0 for open passages. Out-of-bounds padding defaults to 1.0 (wall).
- **Channel 1 — Agent:** 1.0 at the agent's current cell, 0.0 elsewhere.
- **Channel 2 — Goal:** 1.0 at the goal cell if it falls within the observation window, 0.0 otherwise.

### Action Space

Four discrete actions map to cardinal directions: up (−row), right (+col), down (+row), left (−col).

### Reward Function

```
r = −0.01          # per-step penalty (discourages dawdling)
r −= 0.05          # if movement was blocked by a wall
r += 1.00          # terminal reward on reaching the goal
r += 0.01 × Δd     # dense shaping: reward proportional to distance reduction
r += 0.005         # exploration bonus for visiting a new cell
```

The dense shaping term and exploration bonus are included to accelerate early learning in the sparse-reward maze setting, consistent with standard practice in navigation RL research.

---

## A2C Architecture (Step 4)

### FlyGM Network

The policy follows Algorithm 1 of the FlyGM paper. At each timestep $t$:

> - **Step 1 — Encoder**
$$\tilde{x}_t = \text{Enc}_\theta(x_t) \in \mathbb{R}^{d_\text{enc}}$$
A LayerNorm → Linear → ReLU block maps the 75-dim observation to a 32-dim encoding.

> - **Step 2 — Afferent Injection** *(Equation 4)*
$$H_t[V_a] \leftarrow \tanh\!\bigl(W_g[H_t[V_a] \,\|\, \mathbf{1}\tilde{x}_t^\top] + b_g\bigr)$$
Each of the $N_a = 48$ afferent neurons receives its current state concatenated with $\tilde{x}_t$ and passes through a learned linear gate.

> - **Step 3 — Synaptic Aggregation** *(Equation 5)*
> $$M_t = W H_t \in \mathbb{R}^{N \times C}$$
> $W \in \mathbb{R}^{N \times N}$ is the **fixed** signed connectome weight matrix. It is registered as a non-trainable buffer — the optimiser never updates it.

> - **Step 4 — Neuron State Update** *(Equation 6)*
> - $$H_{t+1}[v] = f_\psi\!\bigl([M_t[v] \,\|\, \eta_v]\bigr), \quad \forall v \in V$$
> A shared two-layer MLP $f_\psi$ (ELU → Tanh) updates each neuron's state conditioned on its trainable intrinsic descriptor $\eta_v \in \mathbb{R}^D$.

> - **Step 5 — Decoding**
> The efferent neuron states $H_{t+1}[V_e]$ are flattened and passed through two separate MLP heads:
> - **Actor:** → action logits $\in \mathbb{R}^4$
> - **Critic:** → scalar value $V(s_t) \in \mathbb{R}$

### Network Size

| Component | Shape / Count |
|---|---|
| Afferent neurons $N_a$ | 48 |
| Intrinsic neurons $N_i$ | 144 |
| Efferent neurons $N_e$ | 24 |
| Total neurons $N$ | 216 |
| Channel dim $C$ | 32 |
| Descriptor dim $D$ | 16 |
| Fixed synapses in $W$ | ~18,000 (sparse) |
| Trainable parameters | ~120,000 |

### A2C Loss Function

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{actor} + c_v \mathcal{L}_\text{critic} - c_e \mathcal{L}_\text{entropy}$$

$$\mathcal{L}_\text{actor}  = -\mathbb{E}_t\!\left[\log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

$$\mathcal{L}_\text{critic} = \tfrac{1}{2}\,\mathbb{E}_t\!\left[\left(V_\theta(s_t) - R_t\right)^2\right]$$

$$\mathcal{L}_\text{entropy} = -\mathbb{E}_t\!\left[H(\pi_\theta(\cdot|s_t))\right]$$

where $\hat{A}_t = R_t - V_\theta(s_t)$ (normalised per batch), and $R_t$ is the discounted n-step return bootstrapped from $V_\theta(s_{T+1})$.

---

## Training Strategy (Step 5)

| Hyperparameter | Value |
|---|---|
| Parallel environments | 8 |
| Rollout length $T$ | 16 steps |
| Total timesteps | 500,000 |
| Optimiser | Adam ($\epsilon = 10^{-5}$) |
| Learning rate | $3 \times 10^{-4}$ (linear decay to 10%) |
| Discount $\gamma$ | 0.99 |
| Value coefficient $c_v$ | 0.5 |
| Entropy coefficient $c_e$ | 0.02 |
| Gradient clip norm | 0.5 |
| Recurrent handling | Truncated BPTT (detach H each step) |

The two-stage training described in the FlyGM paper (imitation learning → PPO fine-tuning) is simplified here to a single A2C stage, since no expert demonstration data exists for this custom maze task.

---

## Evaluation Metrics (Step 6)

Evaluation is performed every 50,000 environment steps using the **greedy** (argmax) policy on 20 freshly-generated mazes. The following metrics are tracked:

| Metric | Description |
|---|---|
| `success_rate_pct` | Fraction of evaluation episodes where the agent reaches the goal (primary metric) |
| `mean_episode_reward` | Mean total undiscounted reward per episode |
| `mean_episode_length` | Mean steps per episode; lower is better when success rate is high |
| `mean_cells_explored_pct` | Percentage of open cells visited; measures exploration coverage |
| `update/entropy` | Policy entropy; low entropy signals premature convergence |
| `update/total_loss` | Composite A2C loss; used to monitor training stability |

---

## Installation and Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training from the project root
python -m src.train

# 3. Or customise hyperparameters programmatically
python - <<'EOF'
from src.train import Config, train
cfg = Config(total_timesteps=1_000_000, n_envs=16, maze_size=19)
network, history = train(cfg)
EOF
```

Outputs are written to `outputs/`:
- `flygm_navigation.pt` — saved model weights
- `metrics.json` — full training history
- `training_curves.png` — six-panel learning curve plot

---

## Relationship to FlyGM Paper

| FlyGM paper (locomotion) | This project (navigation) |
|---|---|
| *Drosophila* FlyWire connectome (~131k neurons) | Synthetic bio-inspired graph (216 neurons) |
| Locomotion tasks: walk, turn, fly | Navigation: reach goal in random maze |
| PPO + imitation learning initialisation | A2C (single stage) |
| MuJoCo biomechanical simulation | Custom Gymnasium GridMazeEnv |
| Baselines: rewired graph, random graph, MLP | To replicate: swap FlyGM for MLP baseline |

The structural principle is identical: a fixed graph $W$ acts as a structural inductive bias encoded into the neural controller, with only the encoder, gate, update MLP, and decoder being learned. The paper's claim — that biological wiring improves sample efficiency over random or hand-crafted graphs — can be tested by replacing `build_navigation_connectome()` with a random Erdős–Rényi graph of the same density and comparing learning curves.

---

## Limitations and Future Work

The current implementation simplifies several aspects of the full FlyGM pipeline. The graph is synthetically constructed rather than derived from actual connectome data, so it does not inherit the full structural inductive bias demonstrated in the paper. The single-stage A2C training lacks the imitation-learning initialisation that accelerated convergence in the original work. Extending to continuous action spaces, larger mazes with multiple rooms, or multi-goal tasks would provide a more comprehensive test of the architecture's generality.

## Credits

[arXiv:2602.17997v2](https://arxiv.org/abs/2602.17997v2)

Anthropic Claude Sonnet 4.6 Extended | moonshot.ai Kimi K2.5 Thinking | Google Gemini 3 Thinking + Deep Research
