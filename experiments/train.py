import os
import sys
import yaml
import argparse
import numpy as np
import json
from typing import Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.packet_routing_env import PacketRoutingEnv
from agents.q_learning import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent

def train_agent(agent_type: str, config: Dict[str, Any]):
    # Parameters
    episodes = config.get("episodes", 5000)
    alpha = config.get("alpha", 0.1)
    gamma = config.get("gamma", 0.95)
    epsilon_start = config.get("epsilon_start", 1.0)
    epsilon_end = config.get("epsilon_end", 0.05)
    epsilon_decay_steps = config.get("epsilon_decay_steps", 4000)
    max_hops = config.get("max_hops", 30)
    seed = config.get("seed", 42)
    
    # Initialize environment with seed
    env = PacketRoutingEnv(max_hops=max_hops)
    env.reset(seed=seed)
    
    # Initialize Agent
    if agent_type == "q_learning":
        agent = QLearningAgent(alpha=alpha, gamma=gamma, seed=seed)
        save_path = config.get("save_path_q", "experiments/q_table.npy")
    elif agent_type == "double_q":
        agent = DoubleQLearningAgent(alpha=alpha, gamma=gamma, seed=seed)
        save_path = config.get("save_path_double_q", "experiments/double_q_table.npz")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Metrics logging
    history = {
        "episode_rewards": [],
        "episode_hops": [],
        "delivery_success": [],
        "congested_hits": [],
        "invalid_actions": []
    }

    print(f"Starting training for {agent_type} (Episodes: {episodes})...")
    
    for ep in range(episodes):
        # Reset environment
        state, info = env.reset()
        
        ep_reward = 0.0
        ep_hops = 0
        ep_congested_hits = 0
        ep_invalid_actions = 0
        terminated = False
        truncated = False
        
        # Calculate decaying epsilon
        # Linear decay
        if ep < epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_steps)
        else:
            epsilon = epsilon_end
            
        while not (terminated or truncated):
            mask = env.get_action_mask(env.current_node)
            
            # Select action
            action = agent.select_action(state, mask, epsilon=epsilon)
            
            # Step env
            next_state, reward, terminated, truncated, next_info = env.step(action)
            
            # Update metrics
            ep_reward += reward
            ep_hops += 1
            if next_info.get("congested_edge_hit", False):
                ep_congested_hits += 1
            if next_info.get("invalid_action", False):
                ep_invalid_actions += 1
                
            # Update agent
            next_mask = env.get_action_mask(env.current_node)
            agent.update(state, action, reward, next_state, next_mask, terminated)
            
            state = next_state
            
        # Log episode metrics
        history["episode_rewards"].append(float(ep_reward))
        history["episode_hops"].append(int(ep_hops))
        history["delivery_success"].append(int(terminated))  # Terminated means delivered
        history["congested_hits"].append(int(ep_congested_hits))
        history["invalid_actions"].append(int(ep_invalid_actions))
        
        # Verbose print every 500 episodes
        if (ep + 1) % 500 == 0 or ep == 0:
            avg_rew = np.mean(history["episode_rewards"][-100:])
            avg_hops = np.mean(history["episode_hops"][-100:])
            success_rate = np.mean(history["delivery_success"][-100:]) * 100
            print(f"Episode {ep+1}/{episodes} | Epsilon: {epsilon:.3f} | Avg Reward (last 100): {avg_rew:.2f} | "
                  f"Avg Hops: {avg_hops:.1f} | Delivery Rate: {success_rate:.1f}%")
            
    # Save trained Q-table
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Training complete. Agent saved to {save_path}")
    
    # Save training history
    history_path = save_path.replace(".npy", "_history.json").replace(".npz", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    return agent, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for packet routing.")
    parser.add_argument("--agent", type=str, default="q_learning", choices=["q_learning", "double_q"],
                        help="Type of RL Agent to train.")
    parser.add_argument("--config", type=str, default="experiments/configs.yaml",
                        help="Path to yaml config file.")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        
    train_agent(args.agent, config_data)
