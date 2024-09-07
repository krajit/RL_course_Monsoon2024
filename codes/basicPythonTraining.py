
# line 1 of the value iteration code
theta = 1e-5
gamma = 0.9

import numpy as np
import maze

# V will be save in a dictionary
V = {}
for i in range(5):
    for j in range(5):
        V[(i,j)] = -10*np.random.rand()
V[(4,4)] = 0

##  HW
# Implemment line 3 to 11


env = maze.Maze()
env.reset()
env.render()

#  helpful function
#env.simulate_step()


print("done")

