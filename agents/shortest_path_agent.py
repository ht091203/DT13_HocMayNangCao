from typing import List, Dict
from envs.packet_routing_env import NEIGHBORS

class ShortestPathAgent:
    """
    Shortest Path Agent (Baseline 1).
    Computes the shortest path (minimum hops) on the static graph using BFS,
    ignoring dynamic congestion.
    """
    def __init__(self):
        self.neighbors = NEIGHBORS

    def _bfs_shortest_path(self, start: int, target: int) -> List[int]:
        if start == target:
            return [start]
        queue = [[start]]
        visited = {start}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in self.neighbors.get(node, []):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def select_action(self, state: int, mask: List[int], current_node: int, destination: int) -> int:
        """
        Selects the action corresponding to the next hop on the shortest path.
        
        Args:
            state: The encoded state ID.
            mask: The action mask.
            current_node: Decoded current node.
            destination: Decoded destination node.
        
        Returns:
            The action index in [0, 3].
        """
        if current_node == destination:
            return 0  # Already at destination
            
        path = self._bfs_shortest_path(current_node, destination)
        if len(path) < 2:
            # Fallback: choose first valid action
            valid_actions = [i for i, val in enumerate(mask) if val == 1]
            return valid_actions[0] if valid_actions else 0
            
        next_node = path[1]
        
        # Find action index corresponding to next_node
        node_neighbors = self.neighbors[current_node]
        for idx, neighbor in enumerate(node_neighbors):
            if neighbor == next_node:
                return idx
                
        # Fallback if next_node is not found in neighbors
        valid_actions = [i for i, val in enumerate(mask) if val == 1]
        return valid_actions[0] if valid_actions else 0
