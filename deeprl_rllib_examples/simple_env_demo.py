
import gym

import numpy as np
# from IPython import display
# from matplotlib import pyplot as plt

#env = gym.make('MountainCar-v0', render_mode="human")
#env = gym.make( 'Taxi-v3', render_mode="human")
env = gym.make( 'CartPole-v1', render_mode="human")



state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done,_, extra_info = env.step(action)
    env.render()
print("done")