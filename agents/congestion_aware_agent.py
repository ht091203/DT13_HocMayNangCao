import heapq
from typing import List, Dict, Tuple
from envs.packet_routing_env import NEIGHBORS

class CongestionAwareAgent:
    """
    Congestion-Aware Heuristic Agent (Baseline 2).
    Uses Dijkstra's algorithm to compute the shortest path dynamically
    based on the current congestion state of all edges.
    Edge cost: 1.0 + 5.0 * is_congested.
    """
    def __init__(self, congestion_penalty: float = 5.0):
        self.neighbors = NEIGHBORS
        self.congestion_penalty = congestion_penalty

    def _dijkstra_shortest_path(self, start: int, target: int, congestion_states: Dict[Tuple[int, int], int]) -> List[int]:
        if start == target:
            return [start]
            
        distances = {node: float('inf') for node in self.neighbors}
        distances[start] = 0.0
        # Priority queue stores tuples of (distance, node, path)
        pq = [(0.0, start, [start])]
        
        while pq:
            dist, node, path = heapq.heappop(pq)
            if node == target:
                return path
            if dist > distances[node]:
                continue
                
            for neighbor in self.neighbors.get(node, []):
                # Check congestion status
                is_congested = congestion_states.get((node, neighbor), 0)
                edge_cost = 1.0 + self.congestion_penalty * is_congested
                
                new_dist = dist + edge_cost
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor, path + [neighbor]))
        return []

    def select_action(self, state: int, mask: List[int], current_node: int, destination: int, congestion_states: Dict[Tuple[int, int], int]) -> int:
        """
        Selects the action corresponding to the next hop on the Dijkstra path.
        """
        if current_node == destination:
            return 0
            
        path = self._dijkstra_shortest_path(current_node, destination, congestion_states)
        if len(path) < 2:
            valid_actions = [i for i, val in enumerate(mask) if val == 1]
            return valid_actions[0] if valid_actions else 0
            
        next_node = path[1]
        
        # Find action index corresponding to next_node
        node_neighbors = self.neighbors[current_node]
        for idx, neighbor in enumerate(node_neighbors):
            if neighbor == next_node:
                return idx
                
        valid_actions = [i for i, val in enumerate(mask) if val == 1]
        return valid_actions[0] if valid_actions else 0
