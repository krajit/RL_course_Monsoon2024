import numpy as np
import pygame
import itertools, time
from maze import Maze
import random

# CODE STARTS RUNNING FROM HERE
env = Maze()
state = env.reset()

theta = 1e-5
gamma = 0.9

# value function arbitrarily initialized
V = {(s1,s2): -20*np.random.rand() for (s1,s2) in itertools.product(range(5),range(5))}

# arbitrary policy evaluation
pi = {s:np.random.randint(0,4) for s in V}
 # normlaize to make it a distribution


# policy iteration loop
policyStable = False
while not policyStable: 

    #policy evaluation
    Delta = 10
    while Delta > theta:
        Delta = 0
        for s in V:
            v = V[s]
            a = pi[s]
            sp, r, done, _ = env.simulate_step(s,a)
            V[s] = (r + gamma*V[sp])
            Delta = max(Delta, np.abs(v - V[s]))

    #policy improvement
    policyStable = True
    for s in V:
        oldAction = pi[s]
        Va = [0., 0., 0., 0.]
        for a in range(4):
            sp, r, done, _ = env.simulate_step(s,a)
            Va[a] = r+ gamma*V[sp]
        pi[s] = np.argmax(Va)
        if oldAction != pi[s]:
            policyStable = False

print("training done")

# lets visualize this V
A = np.random.rand(5,5)
for (s1,s2) in itertools.product(range(5),range(5)):
   A[s1][s2] = V[(s1,s2)] 


import matplotlib.pyplot as plt
plt.imshow(A)
plt.show()

# simulate the trained game
state = env.reset()
env.render()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    action = pi[state]

    state, reward, done, _ = env.step(action)
    env.render()  
    time.sleep(0.1)


print("done")