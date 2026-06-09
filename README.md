# Đề tài 13 — Định Tuyến Gói Tin Trong Mạng Nhỏ

> **Môn học:** Học Máy Nâng Cao — ĐH Công Thương TP.HCM  
> **Nhóm 05** | Trần Văn Toàn · Huỳnh Thị Mộng Tuyền · Lê Thị Trúc Anh · Danh Gia Huy · Nguyễn Vũ Giang

---

## Giới thiệu

Agent học cách **định tuyến gói tin** qua mạng 10 node, chọn next-hop tối ưu để đưa gói tin từ nguồn đến đích nhanh nhất trong khi **tránh các cạnh đang bị tắc nghẽn** (congestion).

### Môi trường (MDP)

| Thành phần | Mô tả |
|---|---|
| **State** | `(current_node, destination, local_congestion_code)` — 1 600 trạng thái |
| **Action** | Chọn 1 trong 4 neighbor (action mask cho node có < 4 neighbor) |
| **Reward** | Hop: −1 \| Congestion: −5 \| Invalid: −10 \| Goal: +30 \| Timeout: −20 |
| **Terminal** | `current_node == destination` |
| **Truncated** | `hop_count > 30` |
| **Dynamics** | Congestion thay đổi ngẫu nhiên sau mỗi bước theo xác suất cạnh |

### Topology mạng

```
10 node (0–9), mỗi node tối đa 4 neighbor, neighbor list cố định để đảm bảo action index nhất quán.

HIGH_RISK edges  (P_high = 0.8): (1,4), (4,5), (7,8)
MEDIUM_RISK edges (P_high = 0.5): (0,1), (1,3), (2,5), (3,6), (5,8), (6,9)
LOW_RISK edges   (P_high = 0.2): các cạnh còn lại
```

---

## Cấu trúc dự án

```
.
├── envs/
│   ├── base_env.py              # Abstract base class (reset, step, render, encode/decode)
│   └── packet_routing_env.py   # Môi trường RL chính
├── agents/
│   ├── random_agent.py          # Random Agent (baseline)
│   ├── shortest_path_agent.py   # Shortest Path — BFS (baseline)
│   ├── congestion_aware_agent.py# Congestion-Aware — Dijkstra + penalty (heuristic)
│   ├── q_learning.py            # Q-Learning với epsilon-greedy + action mask
│   └── double_q_learning.py     # Double Q-Learning (giảm overestimation bias)
├── experiments/
│   ├── configs.yaml             # Hyperparameters
│   ├── train.py                 # Huấn luyện Q-Learning / Double Q
│   ├── evaluate.py              # Đánh giá 10 seed, mean ± std
│   └── sweep.py                 # Reward sensitivity study + ablation study
├── visualization/
│   ├── render.py                # Vẽ đồ thị mạng (NetworkX + Matplotlib)
│   └── plots.py                 # Learning curves, agent comparison, policy heatmap/arrows
├── dashboard/
│   └── app.py                   # Interactive demo — Streamlit
├── tests/
│   ├── test_env.py              # Kiểm thử transition, terminal, seed
│   ├── test_encoder.py          # Kiểm thử encode/decode round-trip
│   ├── test_rewards.py          # Kiểm thử các reward conditions
│   └── test_boundary.py         # Kiểm thử biên, congestion statistics
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
# Clone repo
git clone https://github.com/<username>/packet-routing-rl.git
cd packet-routing-rl

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

---

## Cách chạy

### 1. Huấn luyện agent

```bash
# Huấn luyện Q-Learning (lưu experiments/q_table.npy)
python experiments/train.py --agent q_learning

# Huấn luyện Double Q-Learning (lưu experiments/double_q_table.npz)
python experiments/train.py --agent double_q
```

Output mỗi 500 episode:
```
Episode 500/5000 | Epsilon: 0.877 | Avg Reward (last 100): -12.34 | Avg Hops: 8.2 | Delivery Rate: 71.0%
```

### 2. Đánh giá (10 seed)

```bash
python experiments/evaluate.py
```

In bảng Markdown `mean ± std` cho 5 agents và lưu vào `experiments/eval_results.json`.

### 3. Reward sensitivity + Ablation study

```bash
python experiments/sweep.py
```

Kết quả lưu vào `experiments/sweep_results.json`.

### 4. Chạy unit tests

```bash
pytest tests/ -v
```

### 5. Chạy Dashboard Demo ⭐

```bash
streamlit run dashboard/app.py
```

Mở trình duyệt tại **http://localhost:8501**

---

## Demo Dashboard

Dashboard có 3 tab chính:

### 🎬 Tab Replay

| Bước | Thao tác |
|---|---|
| 1 | Chọn **Source Node** và **Destination Node** từ sidebar |
| 2 | Chọn **Agent** (Random / Shortest Path / Congestion Aware / Q-Learning / Double Q) |
| 3 | Nhấn **🔄 Bắt đầu Episode Mới** — đồ thị mạng hiển thị với màu congestion |
| 4a | Nhấn **▶️ Bước tiếp theo** để xem từng hop một |
| 4b | Hoặc nhấn **⚡ Chạy toàn bộ Episode** để agent tự chạy đến cuối |

**Trực quan hóa:**
- 🟢 Cạnh xanh = LOW congestion | 🔴 Cạnh đỏ = HIGH congestion
- 🟡 Node vàng = vị trí hiện tại của gói tin
- ⭐ Node đỏ (sao) = destination
- Nét đứt xanh dương = đường đi gói tin đã qua

### 🗺️ Tab Policy Visualization

- **Policy Heatmap**: ma trận 10×10 hiển thị Q-value tối ưu cho mọi cặp (source, destination)
- **Policy Arrows**: chọn destination → xem mũi tên next-hop tối ưu vẽ trên đồ thị

### 📈 Tab Kết quả Thực nghiệm

- **Learning curves**: Reward, Hop Count, Delivery Rate theo episode (rolling avg 100)
- **Bar charts**: so sánh 5 agents với error bars (mean ± std, 10 seeds)
- **Bảng tổng hợp**: delivery rate được tô xanh khi đạt ≥ 85%

---

## Kết quả mục tiêu

| Chỉ số | Mục tiêu |
|---|---|
| Delivery Rate (Q-Learning / Double Q) | ≥ 85% |
| Hop Count | ≤ 1.5× Shortest Path trung bình |
| Congestion Cost | Thấp hơn Shortest Path thuần |
| Kết quả ổn định | qua 10 seed khác nhau |

---

## Công thức cốt lõi

**Q-Learning update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

- $\alpha = 0.1$: tốc độ học
- $\gamma = 0.95$: hệ số chiết khấu
- $r + \gamma \max Q$: TD target
- Phần trong ngoặc vuông: TD error
- Nếu $s_{t+1}$ là terminal: target $= r$ (không có phần bootstrap)

**Double Q-Learning** dùng hai bảng $Q_A$, $Q_B$ để tránh overestimation bias trong môi trường stochastic congestion.

---

## Hyperparameters

```yaml
alpha: 0.1
gamma: 0.95
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 4000
episodes: 5000
max_hops: 30
```

---

## Checklist nghiệm thu

- [x] Môi trường RL tự viết: `reset(seed)`, `step(action)`, `render()`, `encode_state()`, `decode_state()`
- [x] Random Agent (chỉ random trong valid actions)
- [x] Heuristic Agent (Shortest Path BFS + Congestion-Aware Dijkstra)
- [x] Q-Learning với epsilon-greedy + action mask
- [x] Double Q-Learning
- [x] Unit tests: transition, reward, encoder, seed, boundary, congestion statistics
- [x] Đánh giá 10 seed, mean ± std
- [x] Learning curves
- [x] Reward sensitivity study (penalty = −3, −5, −7)
- [x] Action mask ablation study
- [x] Dashboard: graph visualization, live congestion, replay, policy visualization
