
import gymnasium as gym
env = gym.make( 'CartPole-v1', render_mode="human")

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done,_, extra_info = env.step(action)
    print("state: ", next_state)
    print("reward: ", reward)
    print("---")
    env.render()
print("done")