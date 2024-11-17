
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from ray.tune.registry import register_env

config = {"gSize":5, "pFrozen":0.8}

def create_env(config):
    gSize = config["gSize"] # grid size
    desc= generate_random_map(size= gSize, p= config["pFrozen"])
    env = gym.make( 'FrozenLake-v1', render_mode="human",desc= desc, map_name=desc, is_slippery=False)
    return env 



for i in range(10):
    env = create_env(config=config)
    #env = gym.make( 'FrozenLake-v1', render_mode="human")

    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done,_, extra_info = env.step(action)
        env.render()
    
print("done")