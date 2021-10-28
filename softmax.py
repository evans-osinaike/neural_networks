import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    
    given_nums = np.array(L)
    
    softmax_result = [np.exp(x)/np.sum([np.exp(y) for y in given_nums]) for x in given_nums]
    
    return softmax_result
    