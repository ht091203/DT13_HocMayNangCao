from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List

class BaseEnv(ABC):
    """
    Abstract base class for custom reinforcement learning environments.
    Enforces Gym-like interface with step, reset, and state encoding/decoding.
    """

    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[int, Dict[str, Any]]:
        """
        Resets the environment to an initial state.
        
        Args:
            seed: Optional random seed.
            
        Returns:
            initial_state: The encoded integer state.
            info: Auxiliary diagnostic information.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: An action provided by the agent.
            
        Returns:
            observation (state): The encoded next state.
            reward: Amount of reward returned after previous action.
            terminated: Whether the agent reaches the terminal state.
            truncated: Whether the episode was truncated (e.g. timeout).
            info: Auxiliary diagnostic information.
        """
        pass

    @abstractmethod
    def render(self) -> str:
        """
        Renders the environment's current state.
        
        Returns:
            A string representation of the state.
        """
        pass

    @abstractmethod
    def encode_state(self, current_node: int, destination: int, congestion_code: int) -> int:
        """
        Encodes the discrete state components into a single integer index.
        """
        pass

    @abstractmethod
    def decode_state(self, state_id: int) -> Tuple[int, int, int]:
        """
        Decodes the state index back into discrete components.
        """
        pass

    @abstractmethod
    def get_action_mask(self, current_node: int) -> List[int]:
        """
        Returns a binary list mask indicating valid action indices.
        """
        pass
