import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time


diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans(X, k):
    """Return cluster centroids and labels from k-means clustering."""
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


def kmeans_diamonds(n, k):
    """Run k-means on the first n rows of the numeric diamonds data."""
    X = diamonds_numeric.iloc[:n].to_numpy()

    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """Return the average runtime of kmeans_diamonds in seconds."""
    runtimes = []

    for _ in range(n_iter):
        start = time()

        kmeans_diamonds(n, k)

        runtime = time() - start
        runtimes.append(runtime)

    return np.mean(runtimes)
