from typing import Tuple, Dict, Iterable, Optional
import numpy as np
import gym
from gym import spaces
import pygame
from pygame import gfxdraw
from collections import defaultdict

import maze

import time
# Running the environment and displaying the Pygame window

# CODE STARTS RUNNING FROM HERE
if __name__ == "__main__":

    Q = {}
   
        
    # placeholder for saving q(s,a)s
    q = defaultdict(lambda: { a: {"val": -20 * np.random.rand(), "count": 1} for a in range(4)})
        # policy function
    def pi(state, epsilon = 0.1):
        if np.random.rand() < epsilon:            # with epsilon probability return random action
            action = env.action_space.sample()
        else:
            action = max(q[state], key=lambda a: q[state][a]["val"])
        return action
    
    #SARSA loop
    epsilon = 0.1
    alpha = 0.05
    gamma = 0.90

    for episode in range(30):
        print("episode number:", episode)
        env = maze.Maze()
        state = env.reset()
        action = pi(state,epsilon=max(0.05, 0.9**episode))
        env.render()

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            newState, reward, done, _ = env.step(action)
            newAction = pi(newState,epsilon=max(0.05, 0.9**episode))
            q[state][action]["val"] = q[state][action]["val"] + alpha*(reward + gamma*q[newState][newAction]["val"] -q[state][action]["val"]  )
            
            state = newState
            action = newAction

            Q = {}
            for (row,col) in q:
                Qsa = [q[row,col][aa]["val"] for aa in range(4)]
                QsaSort = list(np.argsort(Qsa))
                QQ = [QsaSort.index(aa) for aa in range(4)]
                for aa in range(4):
                    Q[row,col,aa] = QQ[aa]/3
            env.render(Q=Q)
            time.sleep(0.01)

