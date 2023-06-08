import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, n_clusters, max_iter=100, atol=.1, method='lyod', **kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.atol = atol
        self.method = method
        self.centroids = None
        self.centroids_history = []
        self.distance_computations = 0
        for kw, kwarg in kwargs.items():
            setattr(self, kw, kwarg)

    def fit(self, X):
        if self.method in ('lyod', 'coresets'):

            if self.method == 'coresets':
                X = self._construct_coreset(X)

            self.centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iter):
                self.centroids_history.append(self.centroids)
                labels = self._assign_labels(X, self.centroids)
                new_centroids = self._update_centroids(X, labels)

                if np.allclose(new_centroids, self.centroids, atol=self.atol):
                    break

                self.centroids = new_centroids
        elif self.method == 'lsh':
            pass
        elif self.method == 'coresets':
            pass     

    def predict(self, X):
        if self.method in ('lyod', 'coresets'):
            distances = cdist(X, self.centroids)
            labels = np.argmin(distances, axis=1)
        elif self.method == 'lsh':
            pass
        elif self.method == 'coresets':
            pass      
        return labels

    def _construct_coreset(self, X):
        coreset_indices = np.random.choice(
            X.shape[0], size=self.samples,
            replace=False
        )
        coreset_points = X[coreset_indices]
        return coreset_points

    def _initialize_centroids(self, X):
        random_indices = np.random.choice(range(X.shape[0]), size=self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids

    def _assign_labels(self, X, centroids):
        distances = cdist(X, centroids)
        self.distance_computations += len(X) * self.n_clusters
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centroids(self, X, labels):
        unique_labels = np.unique(labels)
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            centroids[i] = np.mean(cluster_points, axis=0)

        return centroids