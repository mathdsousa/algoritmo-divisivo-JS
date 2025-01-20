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

def _update_clusters(X, seeds, clusters):
    updated = False

    for i in range(X.shape[0]):
        distances = [euclidean(X[i], seed) for seed in seeds]
        newlabel = np.argmin(distances)
        if updated is False and not np.array_equal(newlabel, clusters[i]):
            updated = True
        clusters[i] = newlabel
    
    return updated

def _update_seeds(X, seeds, clusters):
    updated = False

    for i in range(seeds.shape[0]):
        if np.any(clusters == i): # Avoiding cases with empty clusters 
            newseed = np.mean(X[clusters == i], axis=0)
        else:
            newseed = seeds[i]
        if updated is False and not np.array_equal(newseed, seeds[i]):
            updated = True
        seeds[i] = newseed

    return updated

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

        clusters = np.ones(X.shape[0])
        count = 0
        update_clusters = True
        update_seeds = True

        while (count < self.num_interations):
            print("Atualizando Clusters")
            update_clusters = _update_clusters(X, seeds, clusters)
            print(f'Atualizou? {update_clusters}')
            if (not update_clusters):
                break
            print("Atualizando Sementes")
            update_seeds = _update_seeds(X, seeds, clusters)
            print(f'Atualizou? {update_seeds}')
            if (not update_seeds):
                break
            count += 1
            print(f'Iteração {count}')


        self.seeds = seeds
        return clusters 
        

