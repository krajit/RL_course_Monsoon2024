
# line 1 of the value iteration code
theta = 1e-5
gamma = 0.9

import numpy as np


# V will be save in a dictionary
V = {}
for i in range(5):
    for j in range(5):
        V[(i,j)] = -10*np.random.rand()


print("done")