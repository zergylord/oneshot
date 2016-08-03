import numpy as np
eps = 1e-10
def compute_return(rewards,gamma):
    length = len(rewards)
    R = np.zeros((length,))
    for t in reversed(range(length)):
        R[:t+1] = R[:t+1]*gamma + rewards[t]
    return list(R)
