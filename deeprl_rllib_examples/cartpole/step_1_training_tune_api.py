#import gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
from ray import tune, air
import os

# Configure PPO with a learning rate grid search
config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
)

log_dir = os.path.join(os.getcwd(), "deeprl_rllib_examples/cartpole/ray_logs")

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up the tuner with a stop criterion and checkpoint configuration
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"env_runners/episode_reward_mean": 300.0},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True  # Saves a checkpoint at the end of training
        ),
        storage_path=log_dir
    ),
)

# Start the tuning process
results = tuner.fit()

# Retrieve the best checkpoint from the results
best_checkpoint = results.get_best_result(metric="env_runners/episode_reward_mean", mode="max").checkpoint
print("Best checkpoint path:", best_checkpoint.path)
