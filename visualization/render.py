"""
visualization/render.py
Render the packet routing network graph using NetworkX + Matplotlib.
"""
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from envs.packet_routing_env import NEIGHBORS

# Build the undirected graph once at module level (no side effects — just structure)
_G = nx.Graph()
_G.add_nodes_from(range(10))
for u, neighbors in NEIGHBORS.items():
    for v in neighbors:
        if u < v:
            _G.add_edge(u, v)

# Fixed layout — computed once, reused every call
_POS = nx.spring_layout(_G, seed=42)

# Color constants
COLOR_EDGE_LOW = "#44BB44"
COLOR_EDGE_HIGH = "#FF4444"
COLOR_EDGE_PATH = "#0044FF"
COLOR_NODE_NORMAL = "#6699CC"
COLOR_NODE_CURRENT = "#FFDD00"
COLOR_NODE_DEST = "#FF0000"
LW_LOW = 1.5
LW_HIGH = 3.0
LW_PATH = 4.0


def render_network(
    congestion_states: Optional[Dict[Tuple[int, int], int]] = None,
    current_node: Optional[int] = None,
    destination: Optional[int] = None,
    path: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Render the 10-node packet routing network.

    Args:
        congestion_states: dict mapping (u, v) -> 0 (LOW) or 1 (HIGH).
                           If None, all edges are rendered as LOW.
        current_node: current packet position (highlighted yellow).
        destination: destination node (highlighted red star).
        path: list of nodes visited so far (edges drawn as dashed blue).
        figsize: figure size tuple (width, height) in inches.

    Returns:
        matplotlib.figure.Figure (caller owns it; module does NOT keep a ref).

    Raises:
        ValueError: if congestion_states contains invalid keys or values.
    """
    # --- Validate figsize ---
    if not (isinstance(figsize, tuple) and len(figsize) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in figsize)):
        raise ValueError(f"figsize must be a tuple of 2 positive numbers, got {figsize!r}")

    # --- Validate congestion_states ---
    if congestion_states is not None:
        for k, v in congestion_states.items():
            if not (isinstance(k, tuple) and len(k) == 2):
                raise ValueError(f"congestion_states key must be a 2-tuple, got {k!r}")
            if v not in (0, 1):
                raise ValueError(f"congestion_states value must be 0 or 1, got {v!r}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_title("Packet Routing Network", fontsize=13, fontweight="bold")

    # --- Classify edges ---
    path_edges = set()
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            path_edges.add((min(u, v), max(u, v)))

    low_edges, high_edges = [], []
    for u, v in _G.edges():
        key = (u, v)
        rev_key = (v, u)
        cong = 0
        if congestion_states is not None:
            cong = congestion_states.get(key, congestion_states.get(rev_key, 0))
        if cong == 1:
            high_edges.append((u, v))
        else:
            low_edges.append((u, v))

    # --- Separate path edges from normal edges ---
    low_normal = [(u, v) for u, v in low_edges if (min(u,v), max(u,v)) not in path_edges]
    high_normal = [(u, v) for u, v in high_edges if (min(u,v), max(u,v)) not in path_edges]
    path_edge_list = [(u, v) for u, v in _G.edges() if (min(u,v), max(u,v)) in path_edges]

    # --- Draw edges ---
    if low_normal:
        nx.draw_networkx_edges(_G, _POS, edgelist=low_normal, ax=ax,
                               edge_color=COLOR_EDGE_LOW, width=LW_LOW)
    if high_normal:
        nx.draw_networkx_edges(_G, _POS, edgelist=high_normal, ax=ax,
                               edge_color=COLOR_EDGE_HIGH, width=LW_HIGH)
    if path_edge_list:
        nx.draw_networkx_edges(_G, _POS, edgelist=path_edge_list, ax=ax,
                               edge_color=COLOR_EDGE_PATH, width=LW_PATH,
                               style="dashed")

    # --- Node colors ---
    node_colors = []
    for node in _G.nodes():
        if node == destination and node == current_node:
            node_colors.append(COLOR_NODE_DEST)  # destination wins
        elif node == destination:
            node_colors.append(COLOR_NODE_DEST)
        elif node == current_node:
            node_colors.append(COLOR_NODE_CURRENT)
        else:
            node_colors.append(COLOR_NODE_NORMAL)

    # Draw normal nodes
    normal_nodes = [n for n in _G.nodes() if n != destination]
    normal_colors = [node_colors[n] for n in normal_nodes]
    nx.draw_networkx_nodes(_G, _POS, nodelist=normal_nodes, ax=ax,
                           node_color=normal_colors, node_size=700)

    # Draw destination with star marker
    if destination is not None:
        nx.draw_networkx_nodes(_G, _POS, nodelist=[destination], ax=ax,
                               node_color=COLOR_NODE_DEST, node_size=1200,
                               node_shape="*", linewidths=3,
                               edgecolors="#880000")

    # --- Labels ---
    nx.draw_networkx_labels(_G, _POS, ax=ax,
                            font_color="#FFFFFF", font_size=10, font_weight="bold")

    # --- Legend (only elements present in this render) ---
    legend_handles = []
    legend_handles.append(mpatches.Patch(color=COLOR_NODE_NORMAL, label="Node"))
    if current_node is not None and current_node != destination:
        legend_handles.append(mpatches.Patch(color=COLOR_NODE_CURRENT, label="Current Node"))
    if destination is not None:
        legend_handles.append(mpatches.Patch(color=COLOR_NODE_DEST, label="Destination"))
    if low_normal:
        legend_handles.append(mpatches.Patch(color=COLOR_EDGE_LOW, label="LOW congestion"))
    if high_normal:
        legend_handles.append(mpatches.Patch(color=COLOR_EDGE_HIGH, label="HIGH congestion"))
    if path_edge_list:
        legend_handles.append(mpatches.Patch(color=COLOR_EDGE_PATH, label="Packet path"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.8)

    plt.tight_layout()
    # Do NOT call plt.close(fig) here — caller (Streamlit) owns the figure.
    # Closing it would make st.pyplot(fig) render a blank canvas.
    return fig
