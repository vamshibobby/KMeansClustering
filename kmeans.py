import numpy as np


def get_sse(obs, centroids, labels):
    """
    Finds the sum of the square error (distance) between
    centroids and its labeled observations

    Arguments:
        obs  numpy array of observations (m observations x n attributes)
        centroids  k centroids of m dimensions
        labels  m labels; one for each observation
     """

    return sum((np.sum(np.power(obs[i] - centroids[labels[i]], 2)) for i in range(len(obs))))


def find_labels(obs, centroids):
    """
    Labels each observation in df based upon
    the nearest centroid. 
    
    Args:
        obs  numpy array of observations (m observations x n attributes)
        centroids  k centroids of m dimensions
    
    Returns:
        a numpy array of labels, one for each observation
    """
    return np.array([np.argmin(np.array([np.linalg.norm(j-i) for i in centroids])) for j in obs])
    

def recompute_centroids(obs, centroids, labels):
    """
    Find the new location of the centroids by
    finding the mean location of all points assigned
    to each centroid
    
    Arguments:
        obs  numpy array of observations (m obserations x n attributes)
        centroids  k centroids of m dimensions
        labels  m labels; one for each observation
    
    Returns:
        None; the centroids data frame is updated
    """

    centroids_updated = np.zeros_like(centroids)
    centroids_freq = np.zeros(len(centroids))

    for i in range(len(obs)):
        centroids_freq[labels[i]] += 1
        centroids_updated[labels[i]] = centroids_updated[labels[i]] + obs[i]
        
    centroids_updated1 = np.zeros_like(centroids_updated).astype(float)
    for j in range(len(centroids_updated)):
        if centroids_freq[j] == 0:
            centroids_updated1[j] = centroids[j]
        else:
            centroids_updated1[j] = (centroids_updated[j]/centroids_freq[j])
        
    return centroids_updated1


def cluster_kmeans(obs, k):
    """
    Clusters the m observations of n attributes 
    in the Pandas' dataframe df into k clusters.
    
    Euclidean distance is used as the proximity metric.
    
    Arguments:
        obs  numpy array of observations (m obserations x n attributes)
        k    the number of clusters to search for
        
    Returns:
        a m-sized numpy array of the cluster labels
        
        the final Sum-of-Error-Squared (SSE) from the clustering

        a k x n numpy array of the centroid locations
    """

        
    if k < 1:  # There are no clusters, so I'm returning None
        return None
    
    centroids = np.random.random((k, obs.shape[1]))
    sse_old = get_sse(obs, centroids, np.zeros(len(obs)).astype(int))
    sse_new = 0
    del_sse = 1000
    lbl = np.zeros(len(obs)).astype(int)
    while np.abs(del_sse) > 0.000001:
        sse_old = sse_new
        lbl = find_labels(obs, centroids)
        centroids = recompute_centroids(obs, centroids, lbl)
        sse_new = get_sse(obs, centroids, lbl)
        del_sse = sse_old - sse_new

    return lbl, sse_new, centroids

