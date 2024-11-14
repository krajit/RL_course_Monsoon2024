#path = "/home/ubuntu/ray_results/PPO_2024-11-11_08-40-15/PPO_CartPole-v1_93a16_00000_0_lr=0.0100_2024-11-11_08-40-16/checkpoint_000000"
#path = "/home/ubuntu/ray_results/PPO_2024-11-12_09-22-42/PPO_CartPole-v1_ac66e_00002_2_lr=0.0001_2024-11-12_09-22-44/checkpoint_000000"
#import gym
import os

path = os.path.join(os.getcwd(),"deeprl_rllib_examples/cartpole/ray_logs/PPO_2024-11-14_02-03-47/PPO_CartPole-v1_b048f_00002_2_lr=0.0001_2024-11-14_02-03-49/checkpoint_000000")


from ray.rllib.algorithms.algorithm import Algorithm


my_new_ppo = Algorithm.from_checkpoint(path)

import gymnasium as gym
env = gym.make( 'CartPole-v1', render_mode="human")

state, _ = env.reset()
done = False
while not done:
    #    action = env.action_space.sample()
    action = my_new_ppo.compute_single_action(state)
    state, reward, done,_, extra_info = env.step(action)
    env.render()
print("done")
