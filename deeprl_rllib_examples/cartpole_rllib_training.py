import gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
from ray import tune, air

# Configure and create the CartPole environment
env = gym.make('CartPole-v1', render_mode="human")

# Configure PPO with a learning rate grid search
config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
)

# Set up the tuner with a stop criterion and checkpoint configuration
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150.0},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True  # Saves a checkpoint at the end of training
        )
    ),
)

# Start the tuning process
results = tuner.fit()

# Retrieve the best checkpoint from the results
best_checkpoint = results.get_best_result().checkpoint
print("Best checkpoint path:", best_checkpoint.uri)
