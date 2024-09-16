
import matplotlib
from matplotlib import animation
from IPython.display import HTML


def display_video(frames):
    # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matplotlib.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=50, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def test_env(environment, episodes=10):
    frames = []
    for episode in range(episodes):
        state = environment.reset()
        done = False
        frames.append(environment.render())#mode="rgb_array"))

        while not done:
            action = environment.action_space.sample()
            next_state, reward, done, extra_info = environment.step(action)
            img = environment.render()#mode="rgb_array")
            frames.append(img)
            state = next_state

    return display_video(frames)


import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
plt.close('all')

env = gym.make('Acrobot-v1')
test_env(env, 10)
env.close()




print("done")