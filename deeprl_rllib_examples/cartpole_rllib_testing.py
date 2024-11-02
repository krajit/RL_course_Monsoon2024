
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
)
algo = config.build()  

import os
checkpoint_dir = os.path.join("C:\\Users\\ajit.kumar\\AppData\\Local\\Temp\\c87152c0-18aa-455d-b5d4-b299ef286062")
algo.restore(checkpoint_dir)
print("restoration done")

import gym
env = gym.make( 'CartPole-v1', render_mode="human")
obs = env.reset()
done = False

while not done:
    #action = env.action_space.sample()
    action = algo.compute_single_action(obs)
    obs, reward, done,_, extra_info = env.step(action)
    env.render()
print("done")