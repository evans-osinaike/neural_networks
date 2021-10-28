import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    
    Y = np.array(Y).astype(np.float)
    P = np.array(P).astype(np.float)
    
    cross_entropy = -np.sum(Y * np.log(P) + (1-Y)*np.log(1-P))
    
    return cross_entropy