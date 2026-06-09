import numpy as np
from typing import List, Optional

class QLearningAgent:
    """
    Tabular Q-Learning Agent with Action Masking.
    Q-table shape: (1600, 4)
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((1600, 4), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, mask: List[int], epsilon: float = 0.0) -> int:
        """
        Selects an action using Epsilon-Greedy policy with Action Masking.
        """
        valid_actions = [i for i, val in enumerate(mask) if val == 1]
        if not valid_actions:
            return 0
            
        if self.rng.random() < epsilon:
            # Exploration: choose random valid action
            return int(self.rng.choice(valid_actions))
        else:
            # Exploitation: choose argmax over valid actions
            q_values = self.q_table[state]
            # Filter to valid action values
            valid_q = [q_values[a] for a in valid_actions]
            max_q = max(valid_q)
            # Find all valid actions that achieve max_q (tie-breaking)
            best_actions = [a for a in valid_actions if q_values[a] == max_q]
            return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, next_mask: List[int], terminated: bool):
        """
        Updates the Q-table using the Q-Learning update rule.
        """
        if terminated:
            target = reward
        else:
            valid_actions = [i for i, val in enumerate(next_mask) if val == 1]
            if valid_actions:
                max_next_q = max(self.q_table[next_state][a] for a in valid_actions)
            else:
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q

        # Q-update
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def save(self, filepath: str):
        """Saves the Q-table to a file."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str):
        """Loads the Q-table from a file."""
        self.q_table = np.load(filepath)
