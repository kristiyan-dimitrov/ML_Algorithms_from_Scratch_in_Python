import numpy as np 
import math

def euclidean_distance(v, w):
    """Returns the Euclidean distance between two vectors"""

    return math.sqrt(sum([(v[ii]-w[ii])**2 for ii in range(len(v))]))

def manhattan_distance(v, w):
    """Returns the Manhattan distance between two vectors"""

    return sum([abs(v[ii]-w[ii]) for ii in range(len(v))])

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    
    result = np.ndarray((X.shape[0], Y.shape[0]))
    
    for ii in range(X.shape[0]):
        for kk in range(Y.shape[0]):
            
            result[ii,kk] = euclidean_distance(X[ii], Y[kk])
    
    return result


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    result = np.ndarray((X.shape[0], Y.shape[0]))
    
    for ii in range(X.shape[0]):
        for kk in range(Y.shape[0]):
            
            result[ii,kk] = manhattan_distance(X[ii], Y[kk])
    
    return result

