import numpy as np
from scipy.spatial.distance import euclidean
import sys

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def _init_kmeans_plus_plus(X, num_clusters):

    seeds = []

    # Randomly select the first seed
    seeds.append(X[np.random.randint(X.shape[0], replace=False), :])

    # Select the remaining seeds
    for c_id in range(num_clusters - 1):

        dist = []

        # For each point in the dataset, find the distance to the closest seed
        for i in range(X.shape[0]):
            point = X[i, :]
            d = sys.maxsize

            # For each seed, find the minimum distance
            for j in range(len(seeds)):
                temp_dist = distance(point, seeds[j])
                d = min(d, temp_dist)
            dist.append(d)


        dist = np.array(dist)

        # Select the next seed as the point with the maximum distance from all the already selected seeds
        next_seed = X[np.argmax(dist), :]
        seeds.append(next_seed)
        dist = []

    return seeds


def _create_seeds(X, num_clusters, init='random'):
    if init == 'random':
        seeds = X[np.random.choice(X.shape[0], num_clusters, replace=False)] # Randomly select the seeds
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

    for i in range(len(seeds)):
        if np.any(clusters == i): # Avoiding cases with empty clusters 
            newseed = np.mean(X[clusters == i], axis=0)
        else:
            newseed = seeds[i]
        if updated is False and not np.array_equal(newseed, seeds[i]):
            updated = True
        seeds[i] = newseed

    return updated

class KMeans:
    def __init__(self, num_clusters=2, max_iter=300, init='random'):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.init = init
    
    def fit(self, X):
        # Verifying whether X is an np.ndarray
        if not isinstance(X, np.ndarray):
            X = X.values # Convert the DataFrame to NumPy
    
        seeds = _create_seeds(X, self.num_clusters, self.init)

        clusters = np.zeros(X.shape[0])
        count = 0
        update_clusters = True
        update_seeds = True

        while (count < self.max_iter):
            #print("Atualizando Clusters")
            update_clusters = _update_clusters(X, seeds, clusters)
            #print(f'Atualizou? {update_clusters}')
            if (not update_clusters):
                break
            #print("Atualizando Sementes")
            update_seeds = _update_seeds(X, seeds, clusters)
            #print(f'Atualizou? {update_seeds}')
            if (not update_seeds):
                break
            count += 1
            #print(f'Iteração {count}')

        # Making available the clusters and seeds
        self.labels_ = clusters.astype(int)
        self.cluster_centers_ = seeds

        

