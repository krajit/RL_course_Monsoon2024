import numpy as np
import itertools

# Create the Q dictionary with initial values
Q = {(s1, s2, a): -20 * np.random.rand() for (s1, s2, a) in itertools.product(range(5), range(5), range(4))}

# Print the dictionary to verify
print(Q)
