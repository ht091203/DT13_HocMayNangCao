import random
from typing import List, Optional

class RandomAgent:
    """
    Random Agent that selects action indices uniformly from valid actions (masked).
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_action(self, state: int, mask: List[int]) -> int:
        """
        Selects a random action from the list of valid actions.
        
        Args:
            state: The current integer state (unused by random agent).
            mask: Binary mask [m0, m1, m2, m3] where 1 indicates neighbor exists.
        
        Returns:
            An action index in [0, 3].
        """
        valid_actions = [i for i, val in enumerate(mask) if val == 1]
        if not valid_actions:
            return 0  # Fallback to action 0 if mask is all 0
        return self.rng.choice(valid_actions)
