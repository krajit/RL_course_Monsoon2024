import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Define the RL algorithm configuration
config = PPOConfig().environment("CartPole-v1").framework("torch")


import os

storage_dir = os.path.abspath("./run2")
# Train the PPO agent
tune.run(
    config.algo_class,
    config=config.to_dict(),
    stop={"episode_reward_mean": 200},  # Stop training once reward reaches 200
    #local_dir="./results",  # Save results locally
    storage_path = storage_dir
)

# Shutdown Ray after use
ray.shutdown()
