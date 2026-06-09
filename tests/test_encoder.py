import pytest
from envs.packet_routing_env import PacketRoutingEnv

def test_encoder_roundtrip():
    """
    Verify that decode(encode(state)) == state for all valid states in the space.
    """
    env = PacketRoutingEnv()
    for current_node in range(10):
        for destination in range(10):
            for congestion_code in range(16):
                # Encode state
                state_id = env.encode_state(current_node, destination, congestion_code)
                
                # Verify bounds
                assert 0 <= state_id < 1600, f"Encoded ID {state_id} out of bounds for {current_node, destination, congestion_code}"
                
                # Decode state
                dec_curr, dec_dest, dec_cong = env.decode_state(state_id)
                
                # Assert equality
                assert dec_curr == current_node, f"Mismatch current_node: expected {current_node}, got {dec_curr}"
                assert dec_dest == destination, f"Mismatch destination: expected {destination}, got {dec_dest}"
                assert dec_cong == congestion_code, f"Mismatch congestion_code: expected {congestion_code}, got {dec_cong}"
                
def test_invalid_state_encoding():
    """Verify that assertion error is raised when encoding values out of bounds."""
    env = PacketRoutingEnv()
    with pytest.raises(AssertionError):
        env.encode_state(10, 5, 5)  # Invalid current_node
    with pytest.raises(AssertionError):
        env.encode_state(5, 10, 5)  # Invalid destination
    with pytest.raises(AssertionError):
        env.encode_state(5, 5, 16)  # Invalid congestion_code
