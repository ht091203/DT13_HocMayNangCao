import pytest
from envs.packet_routing_env import PacketRoutingEnv

def test_invalid_action_reward():
    """Verify reward and state transitions for invalid actions."""
    env = PacketRoutingEnv()
    # Node 0 has neighbors [1, 2] -> valid action masks are [1, 1, 0, 0]
    # Reset env with start node 0 and destination node 3
    state, info = env.reset(seed=42, options={"start": 0, "destination": 3})
    
    # Action 2 (neighbor[2]) and 3 (neighbor[3]) are invalid for node 0
    # Test action 2
    next_state, reward, terminated, truncated, info = env.step(2)
    assert reward == -10.0
    assert env.current_node == 0  # Did not move
    assert not terminated
    
    # Action 3
    next_state, reward, terminated, truncated, info = env.step(3)
    assert reward == -10.0
    assert env.current_node == 0
    assert not terminated

def test_normal_versus_congested_reward():
    """Verify that congested moves incur a congestion penalty (-5.0) in addition to hop cost (-1.0)."""
    env = PacketRoutingEnv()
    
    # We will test multiple transitions to find a clean path
    # Let's seed the env and look at congestion states
    state, info = env.reset(seed=10, options={"start": 0, "destination": 9})
    
    # Check if edge (0, 1) or (0, 2) is congested
    # neighbors of 0 are 1 (idx 0) and 2 (idx 1)
    cong_0_1 = env.congestion_states.get((0, 1), 0)
    cong_0_2 = env.congestion_states.get((0, 2), 0)
    
    # Let's step to 1
    next_state, reward, terminated, truncated, info = env.step(0)
    expected_reward = -1.0 - (5.0 if cong_0_1 else 0.0)
    assert reward == expected_reward
    assert env.current_node == 1
    
    # Reset again with seed 23
    state, info = env.reset(seed=23, options={"start": 0, "destination": 9})
    cong_0_2 = env.congestion_states.get((0, 2), 0)
    # Step to 2 (action index 1)
    next_state, reward, terminated, truncated, info = env.step(1)
    expected_reward = -1.0 - (5.0 if cong_0_2 else 0.0)
    assert reward == expected_reward
    assert env.current_node == 2

def test_goal_reward():
    """Verify that reaching the goal yields goal reward (+30) along with hop cost and congestion."""
    env = PacketRoutingEnv()
    # Reset env with start 0 and destination 1
    state, info = env.reset(seed=12, options={"start": 0, "destination": 1})
    cong_0_1 = env.congestion_states.get((0, 1), 0)
    
    # Step to 1 (action index 0)
    next_state, reward, terminated, truncated, info = env.step(0)
    assert env.current_node == 1
    assert terminated
    assert not truncated
    
    # Expected reward: hop (-1) + congestion (-5 if congested) + goal (+30)
    expected_reward = -1.0 - (5.0 if cong_0_1 else 0.0) + 30.0
    assert reward == expected_reward

def test_timeout_reward():
    """Verify that exceeding max steps yields timeout penalty (-20) and truncated flag."""
    # Set max_hops = 3
    env = PacketRoutingEnv(max_hops=3)
    state, info = env.reset(seed=42, options={"start": 0, "destination": 9})
    
    # Step 1: 0 -> 1
    state, reward, term, trunc, info = env.step(0)
    assert not term and not trunc
    
    # Step 2: 1 -> 3
    state, reward, term, trunc, info = env.step(1)
    assert not term and not trunc
    
    # Step 3: 3 -> 6
    # This step should trigger timeout
    state, reward, term, trunc, info = env.step(1)
    assert not term
    assert trunc
    # Expected reward: hop (-1) + timeout (-20) + congestion (-5 if congested)
    expected_reward = -1.0 - (20.0) - (5.0 if info["congested_edge_hit"] else 0.0)
    # Wait, the info returned contains "congested_edge_hit" representing the hit *before* congestion is updated
    # Let's check reward value
    assert reward == expected_reward
