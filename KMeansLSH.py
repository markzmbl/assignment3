import numpy as np
from scipy.spatial import distance



class HashFunction():
    def __init__(self, hash_width, dimensions):
        self.hash_width = hash_width
        self.random_coefficients = np.random.randn(dimensions)
        self.random_bias = np.random.uniform() * hash_width

    def compute_hash(self, x):
        # New compute_hash method implementation
        hash_value = np.sum(np.floor(x * self.random_coefficients)) + self.random_bias
        return int(hash_value % self.hash_width)


class HashTable():
    def __init__(self, hash_width, dimensions, num_rows):
        self.num_rows = num_rows
        self.hash_functions = [HashFunction(hash_width, dimensions) for _ in range(num_rows)]
        self.buckets = dict()
    
    def compute_hash(self, x):
        hash_values = [hash_function.compute_hash(x) for hash_function in self.hash_functions]
        
        return tuple(hash_values)

    def add(self, x, index): 
        hash_value = self.compute_hash(x)
        self.buckets[hash_value] = self.buckets.get(hash_value, set()).union({index})

    def get_similar_items(self, x):
        return self.buckets.get(self.compute_hash(x), set())

class LocalitySensitiveHashing(): 
    def __init__(self, num_bands, num_rows, dimensions, hash_width, num_buckets=None, prime=None):
        self.dimensions = dimensions          
        self.hash_width = hash_width
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.num_buckets = num_buckets
        self.prime = prime
        self.hash_tables = [HashTable(hash_width, dimensions, num_rows) for _ in range(num_bands)]
    
    def fit(self, X):
        for i in range(self.num_bands):
            for index_x in range(len(X)): 
                self.hash_tables[i].add(X[index_x], index_x)


    def get_similar_items(self, x):
        result = set()
        for i in range(self.num_bands):
            result = result.union(self.hash_tables[i].get_similar_items(x))
        return result


class KMeansLSH:
    def __init__(self, num_clusters=2, max_iterations=1,atol=.1, distance_metric="euclidean", num_bands=1, num_rows=1, hash_width=1):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.hash_width = hash_width
        self.lsh = None
        self.centroids = None
        self.centroids_history = []
        self.distance_computations = 0

    def fit(self, X):
        self.lsh = LocalitySensitiveHashing(num_bands=self.num_bands, num_rows=self.num_rows, dimensions=X.shape[1], hash_width=self.hash_width)
        self.lsh.fit(X)
        

        # Randomly initialize centroids
        indices = np.random.choice(X.shape[0], self.num_clusters, replace=False)
        self.centroids = X[indices]
        iteration_count = 0
        for _ in range(self.max_iterations):
            iteration_count += 1

            # Get the data points assigned to each centroid using LSH
            cluster_indices = [np.array(list(self.lsh.get_similar_items(centroid))).astype(int) for centroid in self.centroids]

            # Copy the centroids from the last iteration for the convergence criterion
            previous_centroids = self.centroids.copy()

            # Compute new centroids
            # self.centroids = [np.mean(X[cluster_indices[i]], axis=0) for i in range(self.num_clusters)]
            for i in range(self.num_clusters):
                centroid_mean = np.mean(X[cluster_indices[i]], axis=0)
                np.append(self.centroids,centroid_mean )
                self.distance_computations += len(X) * self.num_clusters
            self.centroids_history.append(self.centroids)            

            if np.all(np.equal(self.centroids, previous_centroids)):
                break
        
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Fit the model before making predictions")

        distances = distance.cdist(X, self.centroids, metric=self.distance_metric)
        labels = np.argmin(distances, axis=1)

        return labels
