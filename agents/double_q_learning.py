import numpy as np
from typing import List, Optional

class DoubleQLearningAgent:
    """
    Double Q-Learning Agent with Action Masking.
    Maintains two tables (q_a, q_b) to reduce overestimation bias.
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.q_a = np.zeros((1600, 4), dtype=np.float32)
        self.q_b = np.zeros((1600, 4), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, mask: List[int], epsilon: float = 0.0) -> int:
        """
        Selects an action using Epsilon-Greedy policy based on QA + QB.
        """
        valid_actions = [i for i, val in enumerate(mask) if val == 1]
        if not valid_actions:
            return 0
            
        if self.rng.random() < epsilon:
            # Exploration
            return int(self.rng.choice(valid_actions))
        else:
            # Exploitation: argmax over QA + QB
            q_values = self.q_a[state] + self.q_b[state]
            valid_q = [q_values[a] for a in valid_actions]
            max_q = max(valid_q)
            best_actions = [a for a in valid_actions if q_values[a] == max_q]
            return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, next_mask: List[int], terminated: bool):
        """
        Updates one of the Q-tables (QA or QB) with probability 0.5.
        """
        valid_actions = [i for i, val in enumerate(next_mask) if val == 1]
        
        if self.rng.random() < 0.5:
            # Update QA using QB for bootstrap value
            if terminated:
                target = reward
            else:
                if valid_actions:
                    # Find a* = argmax_{a} QA(s', a)
                    q_a_next = self.q_a[next_state]
                    max_q_a = max(q_a_next[a] for a in valid_actions)
                    best_a = [a for a in valid_actions if q_a_next[a] == max_q_a]
                    a_star = self.rng.choice(best_a)
                    bootstrap_value = self.q_b[next_state][a_star]
                else:
                    bootstrap_value = 0.0
                target = reward + self.gamma * bootstrap_value
                
            self.q_a[state, action] += self.alpha * (target - self.q_a[state, action])
        else:
            # Update QB using QA for bootstrap value
            if terminated:
                target = reward
            else:
                if valid_actions:
                    # Find b* = argmax_{a} QB(s', a)
                    q_b_next = self.q_b[next_state]
                    max_q_b = max(q_b_next[a] for a in valid_actions)
                    best_b = [a for a in valid_actions if q_b_next[a] == max_q_b]
                    b_star = self.rng.choice(best_b)
                    bootstrap_value = self.q_a[next_state][b_star]
                else:
                    bootstrap_value = 0.0
                target = reward + self.gamma * bootstrap_value
                
            self.q_b[state, action] += self.alpha * (target - self.q_b[state, action])

    def save(self, filepath: str):
        """Saves both Q-tables in a single .npz file."""
        np.savez(filepath, q_a=self.q_a, q_b=self.q_b)

    def load(self, filepath: str):
        """Loads Q-tables from a .npz file."""
        data = np.load(filepath)
        self.q_a = data['q_a']
        self.q_b = data['q_b']
