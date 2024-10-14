from ray import train, tune

import os

# Set TMPDIR to a shorter path
os.environ["TMPDIR"] = "C:/temp"

# Verify that the directory exists or create it
os.makedirs("C:/temp", exist_ok=True)

# def objective(config):  # ①
#     score = config["a"] ** 2 + config["b"]
#     return {"score": score}


# search_space = {  # ②
#     "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
#     "b": tune.choice([1, 2, 3]),
# }

# tuner = tune.Tuner(objective, param_space=search_space)  # ③

# results = tuner.fit()
# print(results.get_best_result(metric="score", mode="min").config)

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Define the RL algorithm configuration
config = PPOConfig().environment("CartPole-v1").framework("torch")


import os

storage_dir = os.path.abspath("./run_oct_14")
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
