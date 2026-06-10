"""
dashboard/app.py
Interactive Streamlit demo for Packet Routing RL — Đề tài 13.

Run from project root:
    streamlit run dashboard/app.py
"""
import os
import sys
import json

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from envs.packet_routing_env import PacketRoutingEnv
from agents.random_agent import RandomAgent
from agents.shortest_path_agent import ShortestPathAgent
from agents.congestion_aware_agent import CongestionAwareAgent
from agents.q_learning import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent
from visualization.render import render_network
from visualization.plots import (
    plot_learning_curves,
    plot_agent_comparison,
    plot_policy_heatmap,
    plot_policy_arrows,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Packet Routing RL — Đề tài 13",
    page_icon="🌐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Q_TABLE_PATH = "experiments/q_table.npy"
DQ_TABLE_PATH = "experiments/double_q_table.npz"
Q_HISTORY_PATH = "experiments/q_table_history.json"
DQ_HISTORY_PATH = "experiments/double_q_table_history.json"
EVAL_PATH = "experiments/eval_results.json"

AGENT_NAMES = [
    "Random",
    "Shortest Path",
    "Congestion Aware",
    "Q-Learning",
    "Double Q-Learning",
]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "env": None,
        "agent": None,
        "agent_name": None,
        "state": None,
        "info": None,
        "path": [],
        "ep_reward": 0.0,
        "ep_hops": 0,
        "congested_hits": 0,
        "invalid_actions": 0,
        "done": False,
        "done_msg": "",
        "step_log": [],
        "episode_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("🌐 Packet Routing RL")
st.sidebar.markdown("**Đề tài 13 — Nhóm 05**")
st.sidebar.divider()

source = st.sidebar.selectbox("Source Node", list(range(10)), index=0)
destination = st.sidebar.selectbox("Destination Node", list(range(10)), index=1)
agent_choice = st.sidebar.selectbox("Agent", AGENT_NAMES, index=0)

# Validation
same_node = source == destination
q_missing = agent_choice == "Q-Learning" and not os.path.exists(Q_TABLE_PATH)
dq_missing = agent_choice == "Double Q-Learning" and not os.path.exists(DQ_TABLE_PATH)
can_start = not same_node and not q_missing and not dq_missing

if same_node:
    st.sidebar.warning("⚠️ Source và Destination phải khác nhau")
if q_missing:
    st.sidebar.warning("⚠️ Q-table chưa được huấn luyện. Vui lòng chạy `train.py` trước.")
if dq_missing:
    st.sidebar.warning("⚠️ Q-table chưa được huấn luyện. Vui lòng chạy `train.py` trước.")

st.sidebar.divider()
# Episode stats display
if st.session_state.episode_started:
    st.sidebar.markdown("### 📊 Episode Stats")
    st.sidebar.metric("Hop Count", st.session_state.ep_hops)
    st.sidebar.metric("Total Reward", f"{st.session_state.ep_reward:.1f}")
    st.sidebar.metric("Congested Edges Hit", st.session_state.congested_hits)
    st.sidebar.metric("Invalid Actions", st.session_state.invalid_actions)

# ---------------------------------------------------------------------------
# Build agent helper
# ---------------------------------------------------------------------------
def _build_agent(name: str):
    if name == "Random":
        return RandomAgent(seed=42)
    elif name == "Shortest Path":
        return ShortestPathAgent()
    elif name == "Congestion Aware":
        return CongestionAwareAgent()
    elif name == "Q-Learning":
        agent = QLearningAgent(seed=42)
        agent.load(Q_TABLE_PATH)
        return agent
    elif name == "Double Q-Learning":
        agent = DoubleQLearningAgent(seed=42)
        agent.load(DQ_TABLE_PATH)
        return agent
    raise ValueError(f"Unknown agent: {name}")


def _get_action(agent, agent_name: str, state: int, mask, env: PacketRoutingEnv):
    if agent_name == "Random":
        return agent.select_action(state, mask)
    elif agent_name == "Shortest Path":
        return agent.select_action(state, mask, env.current_node, env.destination)
    elif agent_name == "Congestion Aware":
        return agent.select_action(state, mask, env.current_node, env.destination, env.congestion_states)
    else:  # Q-Learning / Double Q
        return agent.select_action(state, mask, epsilon=0.0)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_replay, tab_policy, tab_results = st.tabs(
    ["🎬 Replay", "🗺️ Policy Visualization", "📈 Kết quả Thực nghiệm"]
)

# ===========================================================================
# TAB 1: REPLAY
# ===========================================================================
with tab_replay:
    st.header("🎬 Demo — Replay Định Tuyến Gói Tin")

    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 3])

    # --- Reset button ---
    with col_btn1:
        if st.button("🔄 Bắt đầu Episode Mới", disabled=not can_start, use_container_width=True):
            env = PacketRoutingEnv(max_hops=30)
            state, info = env.reset(options={"start": source, "destination": destination})
            st.session_state.env = env
            st.session_state.agent = _build_agent(agent_choice)
            st.session_state.agent_name = agent_choice
            st.session_state.state = state
            st.session_state.info = info
            st.session_state.path = [env.current_node]
            st.session_state.ep_reward = 0.0
            st.session_state.ep_hops = 0
            st.session_state.congested_hits = 0
            st.session_state.invalid_actions = 0
            st.session_state.done = False
            st.session_state.done_msg = ""
            st.session_state.step_log = []
            st.session_state.episode_started = True
            st.rerun()

    # --- Next step button ---
    with col_btn2:
        step_disabled = (
            not st.session_state.episode_started
            or st.session_state.done
        )
        if st.button("▶️ Bước tiếp theo", disabled=step_disabled, use_container_width=True):
            env: PacketRoutingEnv = st.session_state.env
            agent = st.session_state.agent
            agent_name = st.session_state.agent_name
            state = st.session_state.state

            mask = env.get_action_mask(env.current_node)
            prev_node = env.current_node
            action = _get_action(agent, agent_name, state, mask, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            st.session_state.ep_reward += reward
            st.session_state.ep_hops += 1
            st.session_state.state = next_state
            st.session_state.info = info
            st.session_state.path.append(env.current_node)

            if info.get("congested_edge_hit", False):
                st.session_state.congested_hits += 1
            if info.get("invalid_action", False):
                st.session_state.invalid_actions += 1

            # Log this step
            neighbors_list = env.neighbors.get(prev_node, [])
            next_node_name = neighbors_list[action] if action < len(neighbors_list) else "N/A"
            st.session_state.step_log.append({
                "Step": st.session_state.ep_hops,
                "From": prev_node,
                "Action (idx)": action,
                "To": env.current_node,
                "Reward": f"{reward:.1f}",
                "Congested": "⚠️ YES" if info.get("congested_edge_hit") else "OK",
                "Invalid": "❌" if info.get("invalid_action") else "—",
            })

            if terminated:
                st.session_state.done = True
                st.session_state.done_msg = (
                    f"✅ Gói tin đã đến đích {destination} "
                    f"sau {st.session_state.ep_hops} bước!"
                )
            elif truncated:
                st.session_state.done = True
                st.session_state.done_msg = (
                    f"❌ Timeout! Gói tin không đến được đích sau 30 bước."
                )

            st.rerun()

    # --- Run full episode button ---
    with col_btn3:
        run_disabled = (
            not st.session_state.episode_started
            or st.session_state.done
        )
        if st.button("⚡ Chạy toàn bộ Episode", disabled=run_disabled, use_container_width=True):
            env: PacketRoutingEnv = st.session_state.env
            agent = st.session_state.agent
            agent_name = st.session_state.agent_name
            state = st.session_state.state

            while True:
                mask = env.get_action_mask(env.current_node)
                prev_node = env.current_node
                action = _get_action(agent, agent_name, state, mask, env)
                next_state, reward, terminated, truncated, info = env.step(action)

                st.session_state.ep_reward += reward
                st.session_state.ep_hops += 1
                st.session_state.state = next_state
                st.session_state.path.append(env.current_node)
                state = next_state

                if info.get("congested_edge_hit", False):
                    st.session_state.congested_hits += 1
                if info.get("invalid_action", False):
                    st.session_state.invalid_actions += 1

                neighbors_list = env.neighbors.get(prev_node, [])
                st.session_state.step_log.append({
                    "Step": st.session_state.ep_hops,
                    "From": prev_node,
                    "Action (idx)": action,
                    "To": env.current_node,
                    "Reward": f"{reward:.1f}",
                    "Congested": "⚠️ YES" if info.get("congested_edge_hit") else "OK",
                    "Invalid": "❌" if info.get("invalid_action") else "—",
                })

                if terminated:
                    st.session_state.done = True
                    st.session_state.done_msg = (
                        f"✅ Gói tin đã đến đích {destination} "
                        f"sau {st.session_state.ep_hops} bước!"
                    )
                    break
                if truncated:
                    st.session_state.done = True
                    st.session_state.done_msg = (
                        f"❌ Timeout! Gói tin không đến được đích sau 30 bước."
                    )
                    break

            st.session_state.info = info
            st.rerun()

    # --- Done message ---
    if st.session_state.done and st.session_state.done_msg:
        if st.session_state.done_msg.startswith("✅"):
            st.success(st.session_state.done_msg)
        else:
            st.error(st.session_state.done_msg)

    # --- Graph display ---
    col_graph, col_log = st.columns([3, 2])

    with col_graph:
        if st.session_state.episode_started and st.session_state.env is not None:
            env: PacketRoutingEnv = st.session_state.env
            fig = render_network(
                congestion_states=env.congestion_states,
                current_node=env.current_node,
                destination=env.destination,
                path=st.session_state.path,
            )
            st.pyplot(fig)
            plt_mod = sys.modules.get("matplotlib.pyplot")
            if plt_mod:
                plt_mod.close(fig)
        else:
            # Show empty graph before first episode
            fig = render_network()
            st.pyplot(fig)
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            st.info("👆 Chọn Source, Destination, Agent rồi nhấn **Bắt đầu Episode Mới**")

    with col_log:
        st.markdown("### 📋 Step Log")
        if st.session_state.step_log:
            df = pd.DataFrame(st.session_state.step_log)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.caption("Chưa có bước nào.")

# ===========================================================================
# TAB 2: POLICY VISUALIZATION
# ===========================================================================
with tab_policy:
    st.header("🗺️ Policy Visualization — Q-Learning")

    q_exists = os.path.exists(Q_TABLE_PATH)
    if not q_exists:
        st.warning("⚠️ Chưa có Q-table. Vui lòng huấn luyện Q-Learning trước.")
        st.code("python experiments/train.py --agent q_learning")
    else:
        pol_col1, pol_col2 = st.columns(2)

        with pol_col1:
            st.subheader("Policy Heatmap (Max Q-value)")
            fig_heat = plot_policy_heatmap(Q_TABLE_PATH)
            if fig_heat:
                st.pyplot(fig_heat)
                import matplotlib.pyplot as _plt
                _plt.close(fig_heat)

        with pol_col2:
            st.subheader("Policy Arrows (Next-hop per destination)")
            pol_dest = st.selectbox(
                "Chọn Destination cho Policy Arrows",
                list(range(10)),
                index=0,
                key="pol_dest",
            )
            fig_arrows = plot_policy_arrows(pol_dest, Q_TABLE_PATH)
            if fig_arrows:
                st.pyplot(fig_arrows)
                import matplotlib.pyplot as _plt
                _plt.close(fig_arrows)

# ===========================================================================
# TAB 3: RESULTS
# ===========================================================================
with tab_results:
    st.header("📈 Kết quả Thực nghiệm")

    # --- Learning curves ---
    st.subheader("Learning Curves")
    fig_lc = plot_learning_curves(Q_HISTORY_PATH, DQ_HISTORY_PATH)
    if fig_lc:
        st.pyplot(fig_lc)
        import matplotlib.pyplot as _plt
        _plt.close(fig_lc)
    else:
        st.info("Chưa có lịch sử huấn luyện. Vui lòng chạy `train.py` trước.")
        st.code("python experiments/train.py --agent q_learning\npython experiments/train.py --agent double_q")

    st.divider()

    # --- Agent comparison ---
    st.subheader("So sánh các Agent (10 seeds, mean ± std)")
    fig_dr, fig_hc = plot_agent_comparison(EVAL_PATH)

    if fig_dr and fig_hc:
        col_dr, col_hc = st.columns(2)
        with col_dr:
            st.pyplot(fig_dr)
            import matplotlib.pyplot as _plt
            _plt.close(fig_dr)
        with col_hc:
            st.pyplot(fig_hc)
            import matplotlib.pyplot as _plt
            _plt.close(fig_hc)

        # Summary table — dùng st.markdown HTML để tránh lỗi pandas Styler
        st.subheader("Bảng tóm tắt")
        try:
            with open(EVAL_PATH) as f:
                results = json.load(f)

            agent_order = ["random", "shortest_path", "congestion_aware", "q_learning", "double_q"]
            agent_display = {
                "random": "Random",
                "shortest_path": "Shortest Path",
                "congestion_aware": "Congestion Aware",
                "q_learning": "Q-Learning",
                "double_q": "Double Q-Learning",
            }

            html = """
<style>
.result-table {width:100%;border-collapse:collapse;font-size:14px;}
.result-table th {background:#2c3e50;color:white;padding:8px 12px;text-align:left;}
.result-table td {padding:7px 12px;border-bottom:1px solid #ddd;}
.result-table tr:hover td {background:#f5f5f5;}
.highlight-green {background:#b6f5b6;color:#1a5c1a;font-weight:bold;}
.badge-pass {background:#27ae60;color:white;border-radius:4px;padding:2px 6px;font-size:12px;}
.badge-fail {background:#e74c3c;color:white;border-radius:4px;padding:2px 6px;font-size:12px;}
</style>
<table class="result-table">
<tr>
  <th>Agent</th>
  <th>Delivery Rate (%)</th>
  <th>Avg Hop Count</th>
  <th>Congestion Cost</th>
  <th>Timeout Rate (%)</th>
</tr>
"""
            for a in agent_order:
                if a not in results:
                    continue
                r = results[a]
                dr_mean = r.get("delivery_rates_mean", 0)
                dr_std  = r.get("delivery_rates_std", 0)
                hc_mean = r.get("hop_counts_mean", 0)
                hc_std  = r.get("hop_counts_std", 0)
                cc_mean = r.get("congestion_costs_mean", 0)
                cc_std  = r.get("congestion_costs_std", 0)
                to_mean = r.get("timeout_rates_mean", 0)
                to_std  = r.get("timeout_rates_std", 0)

                is_rl = a in ("q_learning", "double_q")
                dr_class = "highlight-green" if (is_rl and dr_mean >= 85) else ""
                badge = '<span class="badge-pass">✅ ≥85%</span>' if (is_rl and dr_mean >= 85) else ""

                html += f"""<tr>
  <td><b>{agent_display[a]}</b></td>
  <td class="{dr_class}">{dr_mean:.1f} ± {dr_std:.1f} {badge}</td>
  <td>{hc_mean:.2f} ± {hc_std:.2f}</td>
  <td>{cc_mean:.2f} ± {cc_std:.2f}</td>
  <td>{to_mean:.1f} ± {to_std:.1f}</td>
</tr>"""

            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi đọc eval_results.json: {e}")
    else:
        st.info("Chưa có kết quả đánh giá. Vui lòng chạy `evaluate.py` trước.")
        st.code("python experiments/evaluate.py")
