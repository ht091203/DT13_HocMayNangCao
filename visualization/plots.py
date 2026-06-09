"""
visualization/plots.py
Plotting utilities: learning curves, agent comparison, policy heatmap, policy arrows.
"""
import json
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import networkx as nx

from envs.packet_routing_env import NEIGHBORS

# Reuse same fixed graph/layout as render.py
_G = nx.Graph()
_G.add_nodes_from(range(10))
for u, neighbors in NEIGHBORS.items():
    for v in neighbors:
        if u < v:
            _G.add_edge(u, v)
_POS = nx.spring_layout(_G, seed=42)

AGENT_ORDER = ["random", "shortest_path", "congestion_aware", "q_learning", "double_q"]
AGENT_LABELS = {
    "random": "Random",
    "shortest_path": "Shortest\nPath",
    "congestion_aware": "Congestion\nAware",
    "q_learning": "Q-Learning",
    "double_q": "Double Q",
}

Q_TABLE_PATH = "experiments/q_table.npy"
Q_HISTORY_PATH = "experiments/q_table_history.json"
DQ_HISTORY_PATH = "experiments/double_q_table_history.json"
EVAL_RESULTS_PATH = "experiments/eval_results.json"


# ---------------------------------------------------------------------------
# Rolling average helper
# ---------------------------------------------------------------------------
def _rolling_avg(values, window=100):
    if len(values) < window:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(
    q_history_path: str = Q_HISTORY_PATH,
    dq_history_path: str = DQ_HISTORY_PATH,
) -> Optional[plt.Figure]:
    """
    Plot learning curves (reward, hops, delivery) for Q-Learning and/or Double Q.

    Returns:
        Figure or None if neither history file exists.
    """
    histories = {}

    for label, path in [("Q-Learning", q_history_path), ("Double Q-Learning", dq_history_path)]:
        try:
            with open(path, "r") as f:
                histories[label] = json.load(f)
        except FileNotFoundError:
            print(f"Training history not found: {path}", file=sys.stderr)

    if not histories:
        return None

    fields = ["episode_rewards", "episode_hops", "delivery_success"]
    ylabels = [
        "Avg Reward (100-ep rolling)",
        "Avg Hops (100-ep rolling)",
        "Delivery Rate (100-ep rolling)",
    ]
    colors = {"Q-Learning": "#2266CC", "Double Q-Learning": "#FF8800"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")

    for ax, field, ylabel in zip(axes, fields, ylabels):
        for label, hist in histories.items():
            raw = hist.get(field, [])
            if not raw:
                continue
            smoothed = _rolling_avg(raw, window=100)
            episodes = np.arange(len(smoothed))
            ax.plot(episodes, smoothed, color=colors[label], label=label, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Agent comparison bar charts
# ---------------------------------------------------------------------------
def plot_agent_comparison(
    eval_path: str = EVAL_RESULTS_PATH,
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Two bar charts: delivery rate and hop count for all agents.

    Returns:
        (delivery_rate_fig, hop_count_fig) — either can be None if data missing.
    """
    try:
        with open(eval_path, "r") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

    # Only plot agents with data
    agents = [a for a in AGENT_ORDER if a in results]
    if not agents:
        return None, None

    labels = [AGENT_LABELS.get(a, a) for a in agents]
    dr_means = [results[a].get("delivery_rates_mean", 0) for a in agents]
    dr_stds  = [results[a].get("delivery_rates_std", 0)  for a in agents]
    hc_means = [results[a].get("hop_counts_mean", 0)     for a in agents]
    hc_stds  = [results[a].get("hop_counts_std", 0)      for a in agents]

    x = np.arange(len(agents))
    bar_colors = ["#CC4444", "#4488CC", "#44AA44", "#8844CC", "#FF8800"][:len(agents)]

    # --- Delivery rate figure ---
    fig_dr, ax_dr = plt.subplots(figsize=(8, 5))
    bars = ax_dr.bar(x, dr_means, yerr=dr_stds, capsize=5,
                     color=bar_colors, alpha=0.85, edgecolor="black")
    ax_dr.axhline(85, color="red", linestyle="--", linewidth=1.5, label="Target: 85%")
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(labels, fontsize=10)
    ax_dr.set_ylabel("Delivery Rate (%)")
    ax_dr.set_ylim(0, 110)
    ax_dr.set_title("Delivery Rate Comparison (10 Seeds)", fontweight="bold")
    ax_dr.legend()
    ax_dr.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, dr_means):
        ax_dr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    # --- Hop count figure ---
    fig_hc, ax_hc = plt.subplots(figsize=(8, 5))
    bars_hc = ax_hc.bar(x, hc_means, yerr=hc_stds, capsize=5,
                        color=bar_colors, alpha=0.85, edgecolor="black")
    # 1.5× shortest path reference line
    sp_idx = agents.index("shortest_path") if "shortest_path" in agents else None
    if sp_idx is not None:
        ref = hc_means[sp_idx] * 1.5
        ax_hc.axhline(ref, color="orange", linestyle="--", linewidth=1.5,
                      label=f"1.5× Shortest Path ({ref:.1f})")
        ax_hc.legend()
    ax_hc.set_xticks(x)
    ax_hc.set_xticklabels(labels, fontsize=10)
    ax_hc.set_ylabel("Average Hop Count")
    ax_hc.set_title("Hop Count Comparison (10 Seeds)", fontweight="bold")
    ax_hc.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_hc, hc_means):
        ax_hc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    return fig_dr, fig_hc


# ---------------------------------------------------------------------------
# 3. Policy heatmap
# ---------------------------------------------------------------------------
def plot_policy_heatmap(q_table_path: str = Q_TABLE_PATH) -> Optional[plt.Figure]:
    """
    Heatmap (10×10): rows = source node, cols = destination node.
    Each cell shows max Q-value averaged over all 16 congestion codes.

    Returns:
        Figure or None if Q-table not found / wrong shape.
    """
    try:
        q_table = np.load(q_table_path)
    except FileNotFoundError:
        print(f"Q-table not found: {q_table_path}", file=sys.stderr)
        return None

    if q_table.shape != (1600, 4):
        print(f"Q-table shape mismatch: expected (1600, 4), got {q_table.shape}", file=sys.stderr)
        return None

    num_nodes = 10
    # Build heatmap matrix
    matrix = np.zeros((num_nodes, num_nodes))
    for src in range(num_nodes):
        for dst in range(num_nodes):
            q_vals = []
            mask = _get_mask(src)
            for cong in range(16):
                state_id = src * 160 + dst * 16 + cong
                valid = [i for i, m in enumerate(mask) if m == 1]
                if valid:
                    q_vals.append(max(q_table[state_id][a] for a in valid))
            matrix[src, dst] = np.mean(q_vals) if q_vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Max Q-value (avg over congestion codes)")

    ax.set_xticks(range(num_nodes))
    ax.set_yticks(range(num_nodes))
    ax.set_xticklabels([str(i) for i in range(num_nodes)])
    ax.set_yticklabels([str(i) for i in range(num_nodes)])
    ax.set_xlabel("Destination Node")
    ax.set_ylabel("Source Node")
    ax.set_title("Q-Learning Policy Heatmap\n(Max Q-value per source→destination)", fontweight="bold")

    # Annotate cells
    threshold = matrix.mean()
    for i in range(num_nodes):
        for j in range(num_nodes):
            color = "#000000" if matrix[i, j] >= threshold else "#FFFFFF"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Policy arrows
# ---------------------------------------------------------------------------
def plot_policy_arrows(
    destination: int,
    q_table_path: str = Q_TABLE_PATH,
) -> Optional[plt.Figure]:
    """
    Draw next-hop arrows on the network graph for a given destination.

    Returns:
        Figure or None if Q-table not found / wrong shape.
    """
    try:
        q_table = np.load(q_table_path)
    except FileNotFoundError:
        print(f"Q-table not found: {q_table_path}", file=sys.stderr)
        return None

    if q_table.shape != (1600, 4):
        print(f"Q-table shape mismatch: expected (1600, 4), got {q_table.shape}", file=sys.stderr)
        return None

    num_nodes = 10
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()
    ax.set_title(f"Policy Arrows — Destination: Node {destination}", fontweight="bold", fontsize=12)

    # Draw base graph
    node_colors = ["#FF4444" if n == destination else "#6699CC" for n in range(num_nodes)]
    nx.draw_networkx_nodes(_G, _POS, ax=ax, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(_G, _POS, ax=ax, edge_color="#CCCCCC", width=1.0, alpha=0.5)
    nx.draw_networkx_labels(_G, _POS, ax=ax, font_color="#FFFFFF", font_size=10, font_weight="bold")

    # Draw arrows for each source node
    for src in range(num_nodes):
        if src == destination:
            continue
        mask = _get_mask(src)
        valid = [i for i, m in enumerate(mask) if m == 1]
        if not valid:
            continue

        # Average Q over 16 congestion codes
        q_avg = np.zeros(4)
        for cong in range(16):
            state_id = src * 160 + destination * 16 + cong
            q_avg += q_table[state_id]
        q_avg /= 16

        best_action = max(valid, key=lambda a: q_avg[a])
        next_node = NEIGHBORS[src][best_action]
        best_q = q_avg[best_action]

        arrow_color = "#006400" if best_q >= 0 else "#CC0000"

        src_pos = _POS[src]
        dst_pos = _POS[next_node]
        dx = dst_pos[0] - src_pos[0]
        dy = dst_pos[1] - src_pos[1]

        ax.annotate(
            "", xy=(dst_pos[0] - dx * 0.15, dst_pos[1] - dy * 0.15),
            xytext=(src_pos[0] + dx * 0.15, src_pos[1] + dy * 0.15),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2.0),
        )

    legend_handles = [
        mpatches.Patch(color="#006400", label="Best action (Q ≥ 0)"),
        mpatches.Patch(color="#CC0000", label="Best action (Q < 0)"),
        mpatches.Patch(color="#FF4444", label="Destination"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: get action mask for a node
# ---------------------------------------------------------------------------
def _get_mask(node: int):
    num_neighbors = len(NEIGHBORS[node])
    mask = [0] * 4
    for i in range(num_neighbors):
        mask[i] = 1
    return mask
