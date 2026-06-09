import pytest
import numpy as np
from envs.packet_routing_env import PacketRoutingEnv, get_edge_congestion_prob

def test_start_equals_dest():
    """Verify state when start and destination are initialized to the same node."""
    env = PacketRoutingEnv()
    # Force start == destination = 5
    state, info = env.reset(seed=42, options={"start": 5, "destination": 5})
    assert env.current_node == 5
    assert env.destination == 5
    # If we step, it should be immediately terminated or handled
    # Since current_node == destination, terminated is True
    # Let's verify what happens if we step
    state_id, reward, terminated, truncated, info = env.step(0)
    # The agent was already at destination. Taking a step from the destination
    # will transition to neighbor (index 0 is node 2), so it's no longer at destination.
    assert env.current_node == 2
    assert not terminated

def test_hop_count_boundaries():
    """Verify truncation occurs exactly at max_hops (30)."""
    env = PacketRoutingEnv(max_hops=30)
    env.reset(seed=1, options={"start": 0, "destination": 9})
    
    # Run 29 steps. None should be truncated.
    for i in range(29):
        # We can take action 0 repeatedly. Even if it goes back and forth, it's fine.
        state, reward, term, trunc, info = env.step(0)
        assert not trunc, f"Truncated prematurely at step {i+1}"
        
    # Step 30
    state, reward, term, trunc, info = env.step(0)
    assert trunc, "Should be truncated at exactly 30 steps"
    # Timeout penalty is included (-20)
    # So reward = hop (-1) + timeout (-20) + congestion (-5 if congested)
    expected_reward = -21.0 - (5.0 if info["congested_edge_hit"] else 0.0)
    assert reward == expected_reward

def test_congestion_probability_statistics():
    """Verify that congestion occurrences match the designed probabilities over 1000 steps."""
    env = PacketRoutingEnv()
    env.reset(seed=42)
    
    # Let's count congestion status for a set of edges over 1000 samples.
    # We will sample congestion by calling step (or just sample_congestion directly).
    # Since step() samples congestion every time, we can run 1000 dummy steps or call env._sample_congestion() directly.
    # Calling env._sample_congestion() is faster and cleaner!
    edge_samples = {edge: [] for u in env.neighbors for v in env.neighbors[u] for edge in [(u, v)]}
    
    for _ in range(1000):
        env._sample_congestion()
        for edge in edge_samples:
            edge_samples[edge].append(env.congestion_states[edge])
            
    # Check average congestion for high, medium, and low risk edges
    for edge, samples in edge_samples.items():
        p = get_edge_congestion_prob(edge[0], edge[1])
        empirical_p = np.mean(samples)
        # Check that empirical probability is close to theoretical probability (within tolerance)
        # Using 95% confidence intervals, tolerance = 0.05 is safe for 1000 samples
        assert abs(empirical_p - p) < 0.06, f"Edge {edge} expected P(high)={p}, got {empirical_p:.3f}"
