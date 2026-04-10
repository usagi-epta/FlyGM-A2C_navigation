"""
graph_builder.py
================
Constructs a bio-inspired directed synapse graph for the navigation agent.
"""

from __future__ import annotations

import numpy as np
import torch


def build_navigation_connectome(
    n_afferent: int = 48,
    n_intrinsic: int = 144,
    n_efferent: int = 24,
    excitatory_fraction: float = 0.70,
    seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """
    Build a bio-inspired signed synaptic weight matrix W.

    Returns
    -------
    W : torch.Tensor, shape (n_total, n_total), dtype=float32
        Row‑normalised signed weight matrix.
    partition : dict
        Slices and total neuron count.
    """
    rng = np.random.default_rng(seed)
    n_total = n_afferent + n_intrinsic + n_efferent

    aff_s, aff_e = 0, n_afferent
    int_s, int_e = n_afferent, n_afferent + n_intrinsic
    eff_s, eff_e = n_afferent + n_intrinsic, n_total

    polarity = np.where(
        rng.random(n_total) < excitatory_fraction, 1.0, -1.0
    ).astype(np.float32)

    W = np.zeros((n_total, n_total), dtype=np.float32)

    def _connect(
        src_range: tuple[int, int],
        dst_range: tuple[int, int],
        density: float,
        w_lo: float = 0.3,
        w_hi: float = 0.8,
    ) -> None:
        src_neurons = np.arange(*src_range)
        dst_neurons = np.arange(*dst_range)
        n_dst = len(dst_neurons)

        for s in src_neurons:
            k = max(1, int(n_dst * density))
            k = min(k, n_dst)
            targets = rng.choice(dst_neurons, size=k, replace=False)
            magnitudes = rng.uniform(w_lo, w_hi, size=k).astype(np.float32)
            for t, mag in zip(targets, magnitudes):
                W[t, s] += polarity[s] * mag

    # Afferent → Intrinsic
    _connect((aff_s, aff_e), (int_s, int_e), density=0.12)

    # Intrinsic → Intrinsic (clustered small‑world)
    n_clusters = max(4, n_intrinsic // 24)
    cluster_size = n_intrinsic // n_clusters

    for ci in range(n_clusters):
        c_lo = int_s + ci * cluster_size
        c_hi = min(int_s + (ci + 1) * cluster_size, int_e)
        _connect((c_lo, c_hi), (c_lo, c_hi), density=0.35, w_lo=0.2, w_hi=0.6)

        next_ci = (ci + 1) % n_clusters
        nc_lo = int_s + next_ci * cluster_size
        nc_hi = min(int_s + (next_ci + 1) * cluster_size, int_e)
        _connect((c_lo, c_hi), (nc_lo, nc_hi), density=0.08, w_lo=0.1, w_hi=0.4)

    # Intrinsic → Efferent
    _connect((int_s, int_e), (eff_s, eff_e), density=0.20)

    # Row‑normalise (spectral radius ≤ 1)
    row_l1 = np.abs(W).sum(axis=1, keepdims=True)
    row_l1 = np.where(row_l1 == 0.0, 1.0, row_l1)
    W = W / row_l1

    partition = {
        "afferent_slice": slice(aff_s, aff_e),
        "intrinsic_slice": slice(int_s, int_e),
        "efferent_slice": slice(eff_s, eff_e),
        "n_total": n_total,
    }

    return torch.tensor(W, dtype=torch.float32), partition


def graph_stats(W: torch.Tensor, partition: dict) -> dict:
    """Return summary statistics of the constructed graph."""
    W_np = W.numpy()
    n_total = partition["n_total"]
    nonzero = int(np.count_nonzero(W_np))
    density = nonzero / (n_total * n_total)
    n_excitatory = int((W_np > 0).sum())
    n_inhibitory = int((W_np < 0).sum())

    return {
        "n_total_neurons": n_total,
        "n_afferent": partition["afferent_slice"].stop - partition["afferent_slice"].start,
        "n_intrinsic": partition["intrinsic_slice"].stop - partition["intrinsic_slice"].start,
        "n_efferent": partition["efferent_slice"].stop - partition["efferent_slice"].start,
        "n_nonzero_synapses": nonzero,
        "graph_density_pct": round(density * 100, 2),
        "n_excitatory_synapses": n_excitatory,
        "n_inhibitory_synapses": n_inhibitory,
        "exc_inh_ratio": round(n_excitatory / max(1, n_inhibitory), 2),
    }