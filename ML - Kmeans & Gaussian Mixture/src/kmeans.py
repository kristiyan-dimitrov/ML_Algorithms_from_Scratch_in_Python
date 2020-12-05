import numpy as np
import math

def euclidean_distance(v, w):
    """Returns the Euclidean distance between two vectors"""

    return math.sqrt(sum([(v[ii]-w[ii])**2 for ii in range(len(v))]))

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


def update_assignments(features, means):
    """
    Takes features & means and returns label assignments 

    Args:
        features (np.ndarray): data that needs labels
        means (np.ndarray): # of rows equal to number of means; 
                            # of columns equal to # of columns of feature
                            i.e. dimensionality is the same

    Returns:
        labels (np.ndarray): array of labels based on which cluster mean is closes
            Has length equal to number of rows in features i.e. one label per feature
    """

    # First, calculate distances between features and means
    # A row are all the distances between a given feature and all the means
    # A column are all the distances between a mean and all the features
    distances = euclidean_distances(features, means)
    # the dimensions of distances are:
    # rows = number of rows in features
    # columns = number of means i.e. number of clusters we are looking for

    # Then, we label each feature based on the mean that's closest to it
    # The below gives the index at each row where the minimum value is found in that row
    feature_labels = np.argmin(distances, axis=1)

    return feature_labels

def update_means(features, feature_labels, n_means):
    """
    Takes features & their labels and returns the means for each label class

    Args:
        features (np.ndarray): dataset to determine new means
        feature_labels (np.ndarray): 1D array with the labels for the features
        n_means (int): number of means to find. Should be equal to n_clusters

    Returns:
        new_means (np.ndarray): 
            # of rows = number of means
            # of columns = dimensionality of features
    """

    new_means = []

    for label in range(n_means):

        new_means.append(np.mean(features[feature_labels==label], axis=0))
    
    return np.array(new_means)


class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        
        # INITIALIZING MEANS:

        # First, I will find the min and max values along each dimension
        mins = np.amin(features, axis=0)
        maxs = np.amax(features, axis=0)

        # Then I will initialize the means as equally distances along all dimensions
        # We are ignoring the 0-th point, because that is at the very minimum
        # We are generating n_clusters+1 points, because we are ignoring 1 point
        self.means = np.linspace(mins, maxs, self.n_clusters+1, endpoint=False)[1:,]

        # Initialize labels
        labels = np.zeros(features.shape[0])

        while sum(labels != update_assignments(features, self.means)) > 0:
            labels = update_assignments(features, self.means)
            self.means = update_means(features, labels, self.n_clusters)


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        return update_assignments(features, self.means)