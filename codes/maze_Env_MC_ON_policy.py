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
    env = maze.Maze()
    state = env.reset()
    env.render()
    Q = {}
   
        
    # placeholder for saving q(s,a)s
    # q = defaultdict(lambda: defaultdict(lambda: {"val": -20 * np.random.rand(), "count": 1}))
    q = defaultdict(lambda: { a: {"val": -20 * np.random.rand(), "count": 1} for a in range(4)})
    


    # policy function
    def pi(state, epsilon = 0.1):
        if np.random.rand() < epsilon:            # with epsilon probability return random action
            action = env.action_space.sample()
        else:
            action = max(q[state], key=lambda a: q[state][a]["val"])
        return action

    def generateOneEpisode(Q,wantToView = False, epsilon = 0):
        state = env.reset()
        if wantToView:
            env.render(Q=Q)
        
        episode = []
        done = False
        while not done:
            if wantToView:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            action = pi(state,epsilon=epsilon)
            newState, reward, done, _ = env.step(action)
            episode.append((state,action,reward))
            state = newState

            if wantToView:
                env.render(Q=Q)  
                time.sleep(0.03)
            #print(action, state, reward)
        return episode

    gamma = 0.9
    for i in range(100): # generate 20 episodes
        Q = {}
        for (row,col) in q:
            Qsa = [q[row,col][aa]["val"] for aa in range(4)]
            QsaSort = list(np.argsort(Qsa))
            QQ = [QsaSort.index(aa) for aa in range(4)]
            for aa in range(4):
                Q[row,col,aa] = QQ[aa]/3

        print("Episode # ", i)
        env.current_episode  = i
        episode = generateOneEpisode(Q,wantToView=True,epsilon = 1/(1+i))
        C = 0
        for (state,action, reward) in reversed(episode):
            C = reward + gamma*C

            q[state][action]["val"] +=(C -q[state][action]["val"] )/(q[state][action]["count"]) 
            q[state][action]["count"] = q[state][action]["count"]+1 

    # run on Trained agent
    env.render(Q=Q)
    generateOneEpisode(Q,wantToView=True)
    print("done")