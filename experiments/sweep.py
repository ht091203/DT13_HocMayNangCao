import os
import sys
import yaml
import json
import numpy as np
from typing import Dict, Any, List

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.packet_routing_env import PacketRoutingEnv
from agents.q_learning import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent

def train_helper(env: PacketRoutingEnv, agent: Any, episodes: int, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int, use_mask: bool) -> List[float]:
    """Helper to train an agent and return episode rewards."""
    rewards = []
    
    for ep in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        
        # Calculate epsilon
        if ep < epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_steps)
        else:
            epsilon = epsilon_end
            
        while not (terminated or truncated):
            if use_mask:
                mask = env.get_action_mask(env.current_node)
            else:
                mask = [1, 1, 1, 1]  # Mask disabled: agent sees all actions as valid
                
            action = agent.select_action(state, mask, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if use_mask:
                next_mask = env.get_action_mask(env.current_node)
            else:
                next_mask = [1, 1, 1, 1]
                
            agent.update(state, action, reward, next_state, next_mask, terminated)
            state = next_state
            
        rewards.append(ep_reward)
        
    return rewards

def evaluate_helper(env: PacketRoutingEnv, agent: Any, agent_type: str, use_mask: bool, seeds: List[int], episodes_per_seed: int = 100) -> Dict[str, float]:
    """Helper to evaluate an agent and return summary metrics."""
    deliveries = []
    hops = []
    congestion_hits = []
    invalid_actions = []
    
    for seed in seeds:
        env.reset(seed=seed)
        seed_deliveries = 0
        seed_hops = 0
        seed_congestions = 0
        seed_invalids = 0
        total_steps = 0
        
        for _ in range(episodes_per_seed):
            state, info = env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                if use_mask:
                    mask = env.get_action_mask(env.current_node)
                else:
                    mask = [1, 1, 1, 1]
                    
                if agent_type in ["q_learning", "double_q"]:
                    action = agent.select_action(state, mask, epsilon=0.0)
                else:
                    action = agent.select_action(state, mask)
                    
                next_state, reward, terminated, truncated, next_info = env.step(action)
                
                total_steps += 1
                seed_hops += 1
                if next_info.get("congested_edge_hit", False):
                    seed_congestions += 1
                if next_info.get("invalid_action", False):
                    seed_invalids += 1
                    
                state = next_state
                
            if terminated:
                seed_deliveries += 1
                
        deliveries.append((seed_deliveries / episodes_per_seed) * 100.0)
        hops.append(seed_hops / episodes_per_seed)
        congestion_hits.append(seed_congestions / episodes_per_seed)
        invalid_actions.append((seed_invalids / total_steps * 100.0) if total_steps > 0 else 0.0)
        
    return {
        "delivery_rate_mean": float(np.mean(deliveries)),
        "delivery_rate_std": float(np.std(deliveries)),
        "hop_count_mean": float(np.mean(hops)),
        "hop_count_std": float(np.std(hops)),
        "congestion_hits_mean": float(np.mean(congestion_hits)),
        "congestion_hits_std": float(np.std(congestion_hits)),
        "invalid_action_rate_mean": float(np.mean(invalid_actions)),
        "invalid_action_rate_std": float(np.std(invalid_actions))
    }

def run_sensitivity_study(config: Dict[str, Any], eval_seeds: List[int]) -> Dict[str, Any]:
    print("\n--- Running Reward Sensitivity Study (Congestion Penalty: 3.0, 5.0, 7.0) ---")
    results = {}
    penalties = [3.0, 5.0, 7.0]
    
    for penalty in penalties:
        print(f"Training Q-Learning with congestion penalty = {penalty}...")
        env = PacketRoutingEnv(max_hops=config["max_hops"], congestion_penalty=penalty)
        agent = QLearningAgent(alpha=config["alpha"], gamma=config["gamma"], seed=config["seed"])
        
        train_helper(
            env=env,
            agent=agent,
            episodes=config["episodes"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay_steps=config["epsilon_decay_steps"],
            use_mask=True
        )
        
        print(f"Evaluating penalty = {penalty}...")
        results[str(penalty)] = evaluate_helper(env, agent, "q_learning", use_mask=True, seeds=eval_seeds)
        
    return results

def run_ablation_study(config: Dict[str, Any], eval_seeds: List[int]) -> Dict[str, Any]:
    print("\n--- Running Action Mask Ablation Study ---")
    results = {}
    
    # 1. Q-learning + Mask
    print("Training Q-Learning with Action Mask...")
    env = PacketRoutingEnv(max_hops=config["max_hops"])
    agent_q_mask = QLearningAgent(alpha=config["alpha"], gamma=config["gamma"], seed=config["seed"])
    train_helper(env, agent_q_mask, config["episodes"], config["epsilon_start"], config["epsilon_end"], config["epsilon_decay_steps"], use_mask=True)
    results["q_learning_with_mask"] = evaluate_helper(env, agent_q_mask, "q_learning", use_mask=True, seeds=eval_seeds)
    
    # 2. Q-learning WITHOUT Mask
    print("Training Q-Learning WITHOUT Action Mask...")
    agent_q_nomask = QLearningAgent(alpha=config["alpha"], gamma=config["gamma"], seed=config["seed"])
    train_helper(env, agent_q_nomask, config["episodes"], config["epsilon_start"], config["epsilon_end"], config["epsilon_decay_steps"], use_mask=False)
    results["q_learning_no_mask"] = evaluate_helper(env, agent_q_nomask, "q_learning", use_mask=False, seeds=eval_seeds)
    
    # 3. Double Q-learning + Mask
    print("Training Double Q-Learning with Action Mask...")
    agent_double_q = DoubleQLearningAgent(alpha=config["alpha"], gamma=config["gamma"], seed=config["seed"])
    # Helper handles double Q agent interface correctly because double_q has same update signature
    train_helper(env, agent_double_q, config["episodes"], config["epsilon_start"], config["epsilon_end"], config["epsilon_decay_steps"], use_mask=True)
    results["double_q_with_mask"] = evaluate_helper(env, agent_double_q, "double_q", use_mask=True, seeds=eval_seeds)
    
    return results

def main():
    with open("experiments/configs.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # We will train for a slightly smaller number of episodes (e.g. 4000) for speed in sweep,
    # or use config values. Let's use config episodes (5000) or override if too slow.
    # 3000 episodes is usually more than enough to converge and much faster!
    # Let's set it to 3000 for efficiency in the sweep.
    config["episodes"] = 3000
    config["epsilon_decay_steps"] = 2500
    
    eval_seeds = list(range(100, 110))
    
    sensitivity_results = run_sensitivity_study(config, eval_seeds)
    ablation_results = run_ablation_study(config, eval_seeds)
    
    sweep_data = {
        "sensitivity": sensitivity_results,
        "ablation": ablation_results
    }
    
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/sweep_results.json", "w") as f:
        json.dump(sweep_data, f, indent=4)
        
    print("\nSweep completed! Results saved to experiments/sweep_results.json")
    
    # Print a summary comparison of the ablation study
    print("\n### Ablation Study Summary")
    print("| Configuration | Delivery Rate (%) | Hop Count | Congestion Hits | Invalid Action Rate (%) |")
    print("|---|---|---|---|---|")
    for variant, metrics in sweep_data["ablation"].items():
        print(f"| **{variant}** | "
              f"{metrics['delivery_rate_mean']:.2f}% ± {metrics['delivery_rate_std']:.2f} | "
              f"{metrics['hop_count_mean']:.2f} ± {metrics['hop_count_std']:.2f} | "
              f"{metrics['congestion_hits_mean']:.2f} ± {metrics['congestion_hits_std']:.2f} | "
              f"{metrics['invalid_action_rate_mean']:.2f}% ± {metrics['invalid_action_rate_std']:.2f} |")

if __name__ == "__main__":
    main()
