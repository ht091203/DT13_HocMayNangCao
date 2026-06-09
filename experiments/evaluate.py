import os
import sys
import yaml
import argparse
import numpy as np
import json
from typing import Dict, Any, List

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.packet_routing_env import PacketRoutingEnv
from agents.random_agent import RandomAgent
from agents.shortest_path_agent import ShortestPathAgent
from agents.congestion_aware_agent import CongestionAwareAgent
from agents.q_learning import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent

def evaluate_agent(agent: Any, env: PacketRoutingEnv, agent_name: str, episodes_per_seed: int = 100, seeds: List[int] = []) -> Dict[str, Any]:
    """Runs evaluation for a single agent across multiple seeds."""
    seed_metrics = {
        "delivery_rates": [],
        "hop_counts": [],
        "congestion_costs": [],
        "timeout_rates": [],
        "invalid_action_rates": []
    }
    
    for seed in seeds:
        # Reset environment with specific seed for this evaluation run
        env.reset(seed=seed)
        
        deliveries = 0
        timeouts = 0
        total_hops = 0
        total_congestion_cost = 0
        total_invalid_actions = 0
        total_steps = 0
        
        for ep in range(episodes_per_seed):
            # Reset environment for each episode within the seed
            state, info = env.reset()
            current_node, destination, _ = env.decode_state(state)
            
            terminated = False
            truncated = False
            
            ep_hops = 0
            ep_congestion_cost = 0
            ep_invalid_actions = 0
            
            while not (terminated or truncated):
                mask = env.get_action_mask(env.current_node)
                
                # Action selection based on agent type
                if agent_name == "random":
                    action = agent.select_action(state, mask)
                elif agent_name == "shortest_path":
                    action = agent.select_action(state, mask, env.current_node, env.destination)
                elif agent_name == "congestion_aware":
                    action = agent.select_action(state, mask, env.current_node, env.destination, env.congestion_states)
                elif agent_name in ["q_learning", "double_q"]:
                    action = agent.select_action(state, mask, epsilon=0.0)  # epsilon=0 for eval
                else:
                    raise ValueError(f"Unknown agent: {agent_name}")
                
                # Step environment
                next_state, reward, terminated, truncated, next_info = env.step(action)
                
                ep_hops += 1
                total_steps += 1
                
                if next_info.get("congested_edge_hit", False):
                    ep_congestion_cost += 5.0  # -5.0 reward penalty -> positive cost
                if next_info.get("invalid_action", False):
                    ep_invalid_actions += 1
                
                state = next_state
                
            # Accumulate episode metrics
            if terminated:
                deliveries += 1
            if truncated:
                timeouts += 1
                
            total_hops += ep_hops
            total_congestion_cost += ep_congestion_cost
            total_invalid_actions += ep_invalid_actions
            
        # Calculate rates for this seed
        delivery_rate = (deliveries / episodes_per_seed) * 100.0
        timeout_rate = (timeouts / episodes_per_seed) * 100.0
        avg_hops = total_hops / episodes_per_seed
        avg_congestion_cost = total_congestion_cost / episodes_per_seed
        invalid_rate = (total_invalid_actions / total_steps * 100.0) if total_steps > 0 else 0.0
        
        seed_metrics["delivery_rates"].append(delivery_rate)
        seed_metrics["hop_counts"].append(avg_hops)
        seed_metrics["congestion_costs"].append(avg_congestion_cost)
        seed_metrics["timeout_rates"].append(timeout_rate)
        seed_metrics["invalid_action_rates"].append(invalid_rate)
        
    # Aggregate statistics (mean and std across seeds)
    summary = {}
    for key, values in seed_metrics.items():
        summary[key + "_mean"] = float(np.mean(values))
        summary[key + "_std"] = float(np.std(values))
        
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate all agents over multiple seeds.")
    parser.add_argument("--config", type=str, default="experiments/configs.yaml", help="Path to config file.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per seed.")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Evaluate 10 seeds from 100 to 109 to keep them distinct from training seeds
    eval_seeds = list(range(100, 110))
    env = PacketRoutingEnv(max_hops=config.get("max_hops", 30))
    
    # Initialize all agents
    agents = {
        "random": RandomAgent(seed=42),
        "shortest_path": ShortestPathAgent(),
        "congestion_aware": CongestionAwareAgent(),
        "q_learning": QLearningAgent(),
        "double_q": DoubleQLearningAgent()
    }
    
    # Load trained Q-tables
    q_path = config.get("save_path_q", "experiments/q_table.npy")
    double_q_path = config.get("save_path_double_q", "experiments/double_q_table.npz")
    
    if os.path.exists(q_path):
        agents["q_learning"].load(q_path)
        print(f"Loaded trained Q-table from {q_path}")
    else:
        print(f"WARNING: Trained Q-table not found at {q_path}. Evaluating untrained Q-Learning.")
        
    if os.path.exists(double_q_path):
        agents["double_q"].load(double_q_path)
        print(f"Loaded trained Double Q-table from {double_q_path}")
    else:
        print(f"WARNING: Trained Double Q-table not found at {double_q_path}. Evaluating untrained Double Q-Learning.")

    results = {}
    print("\nEvaluating all agents over 10 seeds (100 episodes per seed)...")
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_agent(agent, env, name, episodes_per_seed=args.episodes, seeds=eval_seeds)
        
    # Print results as Markdown Table
    print("\n### Evaluation Results (10 Seeds, Mean ± Std)")
    print("| Agent | Delivery Rate (%) | Hop Count | Congestion Cost | Timeout Rate (%) | Invalid Action Rate (%) |")
    print("|---|---|---|---|---|---|")
    
    for name in agents:
        r = results[name]
        print(f"| **{name}** | "
              f"{r['delivery_rates_mean']:.2f}% ± {r['delivery_rates_std']:.2f} | "
              f"{r['hop_counts_mean']:.2f} ± {r['hop_counts_std']:.2f} | "
              f"{r['congestion_costs_mean']:.2f} ± {r['congestion_costs_std']:.2f} | "
              f"{r['timeout_rates_mean']:.2f}% ± {r['timeout_rates_std']:.2f} | "
              f"{r['invalid_action_rates_mean']:.2f}% ± {r['invalid_action_rates_std']:.2f} |")
        
    # Save results to json for visualization
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to experiments/eval_results.json")

if __name__ == "__main__":
    main()
