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
   
        
    # placeholder for saving q(s,a)s
    q = defaultdict(lambda: defaultdict(lambda: {"val": -20 * np.random.rand(), "count": 1}))


    # policy function
    def pi(state, epsilon = 0.1):
        if np.random.rand() < epsilon:            # with epsilon probability return random action
            action = env.action_space.sample()
        else:
            #action = max(q[state], key = q[state].get)  # return action with highest value
            action = max(q[state], key=lambda a: q[state][a]["val"])
        return action

    def generateOneEpisode(wantToView = False, epsilon = 0):
        state = env.reset()
        if state not in q: q[state] = {}
        if wantToView:
            env.render()
        
        episode = []
   
        done = False
        while not done:
            if wantToView:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            action = pi(state,epsilon=epsilon)
            

 #          actionMeaning = env.action_space.action_meanings[action]
            newState, reward, done, _ = env.step(action)
            
            episode.append((state,action,reward))
            state = newState
            if state not in q:
                q[state] = {}

            if wantToView:
                env.render(Q=Q)  
                time.sleep(0.03)
            #print(action, state, reward)
        return episode


    gamma = 0.9

    for i in range(20): # generate 20 episodes
        Q = {}
        for (row,col) in q:
            if (row,col) == (4,4):
                pass
            qmax =  max([q[(row,col)][aa]["val"] for aa in q[(row,col)]])
            qmin =  min([q[(row,col)][aa]["val"] for aa in q[(row,col)]])
            for a in q[(row,col)]:
                Q[(row,col,a)] = -(q[(row,col)][a]["val"] - qmin)/(qmax - qmin +0.1)


        print("Episode # ", i)
        env.current_episode  = i
        episode = generateOneEpisode(wantToView=True,epsilon = 1/(1+i))
        C = 0
        for (state,action, reward) in reversed(episode):
            if state not in q:
                q[state] = {}
            if action not in q[state]:
                q[state][action] = {"val":-20*np.random.rand(), "count":1}
            C = reward + gamma*C

            q[state][action]["val"] +=(C -q[state][action]["val"] )/(q[state][action]["count"]) 
            q[state][action]["count"] = q[state][action]["count"]+1 

    # run on Trained agent
    generateOneEpisode(wantToView=True)
    print("done")