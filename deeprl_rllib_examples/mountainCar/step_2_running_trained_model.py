
#import gym
import os

#/home/ubuntu/Desktop/RL_course_Monsoon2024/deeprl_rllib_examples/mountainCar/ray_logs/PPO_2024-11-14_09-41-46/PPO_MountainCar-v0_abd5d_00002_2_lr=0.0001_2024-11-14_09-41-49/checkpoint_000000

path = os.path.join(os.getcwd(),"deeprl_rllib_examples/mountainCar/ray_logs/PPO_2024-11-14_09-41-46/PPO_MountainCar-v0_abd5d_00002_2_lr=0.0001_2024-11-14_09-41-49/checkpoint_000000")

from ray.rllib.algorithms.algorithm import Algorithm


my_new_ppo = Algorithm.from_checkpoint(path)

import gymnasium as gym
env = gym.make( 'MountainCar-v0', render_mode="human")

state, _ = env.reset()
done = False
while not done:
    #    action = env.action_space.sample()
    action = my_new_ppo.compute_single_action(state)
    state, reward, done,_, extra_info = env.step(action)
    env.render()
print("done")
