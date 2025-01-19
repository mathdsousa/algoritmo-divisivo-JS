import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

def _init_kmeans_plus_plus(X, num_clusters):
    pass

def _create_seeds(X, num_clusters, init='random'):
    if init == 'random':
        seeds = X[np.random.choice(X.shape[0], num_clusters)] # Randomly select the seeds
    elif init == 'k-means++':
        seeds = _init_kmeans_plus_plus(X, num_clusters)
    else:
        seeds = X[:num_clusters] # Use the first num_clusters samples as seeds
    return seeds

def _update_labels(X, seeds, labels):
    for i in range(X.shape[0]):
        distances = [euclidean(X[i], seed) for seed in seeds]
        labels[i] = np.argmin(distances)

def _update_seeds(X, seeds, labels):
    for i in range(seeds.shape[0]):
        seeds[i] = np.mean(X[labels == i], axis=0)

class KMeans:
    def __init__(self, num_clusters=2, num_interations=200, init='random'):
        self.num_clusters = num_clusters
        self.num_interations = num_interations
        self.init = init
    
    def fit_predict(self, X):
        # Verifying whether X is an np.ndarray
        if not isinstance(X, np.ndarray):
            X = X.values # Convert the DataFrame to NumPy
    
        seeds = _create_seeds(X, self.num_clusters, self.init)

        labels = -1 * np.ones(X.shape[0])
        count = 0

        while True:
            _update_labels(X, seeds, labels)
            _update_seeds(X, seeds, labels)

