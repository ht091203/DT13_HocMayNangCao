import pytest
from envs.packet_routing_env import PacketRoutingEnv

def test_transition_correctness():
    """Verify that action indices result in transition to correct neighbors."""
    env = PacketRoutingEnv()
    
    # 0 -> neighbors [1, 2]
    # action 0 -> neighbor 1, action 1 -> neighbor 2
    state, info = env.reset(seed=123, options={"start": 0, "destination": 9})
    assert env.current_node == 0
    
    # Step to 2 (action index 1)
    state, reward, term, trunc, info = env.step(1)
    assert env.current_node == 2
    
    # 2 -> neighbors [0, 5]
    # action 0 -> neighbor 0, action 1 -> neighbor 5
    # Step to 5 (action index 1)
    state, reward, term, trunc, info = env.step(1)
    assert env.current_node == 5
    
    # 5 -> neighbors [2, 4, 8]
    # action 0 -> 2, action 1 -> 4, action 2 -> 8
    # Step to 4 (action index 1)
    state, reward, term, trunc, info = env.step(1)
    assert env.current_node == 4

def test_terminal_state():
    """Verify that when agent reaches destination, episode is terminated."""
    env = PacketRoutingEnv()
    # Reset starting at 0, dest at 2
    state, info = env.reset(seed=1, options={"start": 0, "destination": 2})
    
    # Step to 2 (action index 1)
    state, reward, term, trunc, info = env.step(1)
    assert env.current_node == 2
    assert term
    assert not trunc

def test_seed_replication():
    """Verify that passing the same seed produces identical starting configurations."""
    env1 = PacketRoutingEnv()
    env2 = PacketRoutingEnv()
    
    # Reset both with seed 42
    s1, info1 = env1.reset(seed=42)
    s2, info2 = env2.reset(seed=42)
    
    assert s1 == s2
    assert info1["current_node"] == info2["current_node"]
    assert info1["destination"] == info2["destination"]
    
    # Take the same steps and verify they match
    for action in [0, 0, 0]:
        ns1, r1, term1, trunc1, inf1 = env1.step(action)
        ns2, r2, term2, trunc2, inf2 = env2.step(action)
        assert ns1 == ns2
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
