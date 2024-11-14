
import gymnasium as gym
env = gym.make( 'MountainCar-v0', render_mode="human")

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done,_, extra_info = env.step(action)
    env.render()
print("done")