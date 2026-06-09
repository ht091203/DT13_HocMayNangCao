import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from envs.base_env import BaseEnv

# Define network topology (10 nodes)
# Neighbors list must be sorted to ensure consistent action indexing
NEIGHBORS: Dict[int, List[int]] = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1, 6],
    4: [1, 5, 7],
    5: [2, 4, 8],
    6: [3, 7, 9],
    7: [4, 6, 8],
    8: [5, 7, 9],
    9: [6, 8]
}

# Define edge risks
HIGH_RISK_EDGES = {(1, 4), (4, 1), (4, 5), (5, 4), (7, 8), (8, 7)}
MEDIUM_RISK_EDGES = {
    (0, 1), (1, 0),
    (1, 3), (3, 1),
    (2, 5), (5, 2),
    (3, 6), (6, 3),
    (5, 8), (8, 5),
    (6, 9), (9, 6)
}

def get_edge_congestion_prob(u: int, v: int) -> float:
    """Returns the probability of HIGH congestion for edge (u, v)."""
    if (u, v) in HIGH_RISK_EDGES:
        return 0.8
    elif (u, v) in MEDIUM_RISK_EDGES:
        return 0.5
    elif v in NEIGHBORS.get(u, []):
        return 0.2
    return 0.0

class PacketRoutingEnv(BaseEnv):
    """
    Custom packet routing environment with 10 nodes.
    The agent seeks to route a packet from a source to a destination node,
    minimizing hops and avoiding dynamic congestion.
    """
    
    def __init__(self, max_hops: int = 30, congestion_penalty: float = 5.0):
        self.num_nodes = 10
        self.neighbors = NEIGHBORS
        self.max_hops = max_hops
        self.congestion_penalty = congestion_penalty
        
        # State components
        self.current_node: int = 0
        self.destination: int = 0
        # congestion_states[(u, v)] = 0 (LOW) or 1 (HIGH)
        self.congestion_states: Dict[Tuple[int, int], int] = {}
        
        self.hop_count = 0
        self.rng = np.random.default_rng()
        
        # Initialize congestion states to default LOW
        self._initialize_congestion()

    def _initialize_congestion(self):
        """Reset congestion of all valid edges."""
        self.congestion_states = {}
        for u in self.neighbors:
            for v in self.neighbors[u]:
                self.congestion_states[(u, v)] = 0

    def _sample_congestion(self):
        """Randomly sample congestion for all edges based on their risk probabilities."""
        for u in self.neighbors:
            for v in self.neighbors[u]:
                p = get_edge_congestion_prob(u, v)
                # Sample 1 (HIGH) or 0 (LOW)
                self.congestion_states[(u, v)] = int(self.rng.random() < p)

    def get_local_congestion_code(self, node: int) -> int:
        """
        Encode the congestion of outgoing edges of a node into a bitmask.
        Bit i (0 to len(neighbors)-1) represents congestion of neighbor i.
        """
        node_neighbors = self.neighbors[node]
        code = 0
        for i, neighbor in enumerate(node_neighbors):
            is_high = self.congestion_states.get((node, neighbor), 0)
            if is_high:
                code |= (1 << i)
        return code

    def encode_state(self, current_node: int, destination: int, congestion_code: int) -> int:
        """
        Encodes discrete state components to a unique integer state ID in [0, 1599].
        Equation: current_node * 160 + destination * 16 + congestion_code
        """
        assert 0 <= current_node < self.num_nodes, f"Invalid current_node: {current_node}"
        assert 0 <= destination < self.num_nodes, f"Invalid destination: {destination}"
        assert 0 <= congestion_code < 16, f"Invalid congestion_code: {congestion_code}"
        return current_node * 160 + destination * 16 + congestion_code

    def decode_state(self, state_id: int) -> Tuple[int, int, int]:
        """Decodes integer state ID back into (current_node, destination, congestion_code)."""
        congestion_code = state_id % 16
        temp = state_id // 16
        destination = temp % 10
        current_node = temp // 10
        return current_node, destination, congestion_code

    def get_action_mask(self, current_node: int) -> List[int]:
        """
        Returns a binary list mask of length 4.
        1 indicates action is valid (neighbor exists), 0 otherwise.
        """
        num_neighbors = len(self.neighbors[current_node])
        mask = [0] * 4
        for i in range(num_neighbors):
            mask[i] = 1
        return mask

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Resets the environment.
        options: Dict containing optional keys:
          - "start": int (pre-selected start node)
          - "destination": int (pre-selected destination node)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        # Select start and destination nodes
        start = None
        dest = None
        
        if options is not None:
            start = options.get("start")
            dest = options.get("destination")
            
        if start is None or not (0 <= start < self.num_nodes):
            start = int(self.rng.choice(self.num_nodes))
            
        # If destination not provided in options, or if it is same as start but not explicitly requested:
        if dest is None or not (0 <= dest < self.num_nodes):
            choices = [n for n in range(self.num_nodes) if n != start]
            dest = int(self.rng.choice(choices))
        elif dest == start and (options is None or "destination" not in options):
            choices = [n for n in range(self.num_nodes) if n != start]
            dest = int(self.rng.choice(choices))
            
        self.current_node = start
        self.destination = dest
        self.hop_count = 0
        
        # Sample initial congestion
        self._sample_congestion()
        
        local_code = self.get_local_congestion_code(self.current_node)
        state_id = self.encode_state(self.current_node, self.destination, local_code)
        
        info = {
            "current_node": self.current_node,
            "destination": self.destination,
            "hop_count": self.hop_count,
            "congestion_states": self.congestion_states.copy(),
            "action_mask": self.get_action_mask(self.current_node)
        }
        
        return state_id, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Executes one action.
        action: int in {0, 1, 2, 3} corresponding to the neighbor index.
        """
        mask = self.get_action_mask(self.current_node)
        
        # Check action validity
        if action < 0 or action >= 4 or mask[action] == 0:
            # Invalid action: stand still, receive heavy penalty
            self.hop_count += 1
            reward = -10.0
            terminated = False
            truncated = (self.hop_count >= self.max_hops)
            
            if truncated:
                reward += -20.0
                
            # Keep state, but congestion might change or update? 
            # In typical RL, congestion state changes every step
            self._sample_congestion()
            local_code = self.get_local_congestion_code(self.current_node)
            state_id = self.encode_state(self.current_node, self.destination, local_code)
            
            info = {
                "current_node": self.current_node,
                "destination": self.destination,
                "hop_count": self.hop_count,
                "congestion_states": self.congestion_states.copy(),
                "action_mask": self.get_action_mask(self.current_node),
                "invalid_action": True,
                "congested_edge_hit": False
            }
            return state_id, reward, terminated, truncated, info

        # Valid action: transition to neighbor
        src = self.current_node
        dest_neighbor = self.neighbors[src][action]
        
        # Check congestion status *before* moving
        is_congested = self.congestion_states.get((src, dest_neighbor), 0)
        
        # Perform move
        self.current_node = dest_neighbor
        self.hop_count += 1
        
        # Calculate reward
        reward = -1.0  # Hop cost
        if is_congested:
            reward += -self.congestion_penalty  # Congestion penalty
            
        terminated = (self.current_node == self.destination)
        if terminated:
            reward += 30.0  # Goal reward
            
        truncated = (self.hop_count >= self.max_hops and not terminated)
        if truncated:
            reward += -20.0  # Timeout penalty
            
        # Sample new congestion for next step
        self._sample_congestion()
        local_code = self.get_local_congestion_code(self.current_node)
        state_id = self.encode_state(self.current_node, self.destination, local_code)
        
        info = {
            "current_node": self.current_node,
            "destination": self.destination,
            "hop_count": self.hop_count,
            "congestion_states": self.congestion_states.copy(),
            "action_mask": self.get_action_mask(self.current_node),
            "invalid_action": False,
            "congested_edge_hit": bool(is_congested)
        }
        
        return state_id, reward, terminated, truncated, info

    def render(self) -> str:
        local_code = self.get_local_congestion_code(self.current_node)
        neighbors_list = self.neighbors[self.current_node]
        congestion_info = []
        for i, neighbor in enumerate(neighbors_list):
            status = "HIGH" if (local_code & (1 << i)) else "LOW"
            congestion_info.append(f"{neighbor}({status})")
            
        return (f"Hop: {self.hop_count} | Node: {self.current_node} -> Dest: {self.destination} | "
                f"Neighbors: [{', '.join(congestion_info)}]")
