import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

def centroid(cluster):
    s = np.array([0., 0., 0., 0.])
    for c in cluster:
        s += c
    return s / len (cluster)

def kmeans(data, k, epochs = 10):
    clusters = [[] for i in range(k)]

    for entry in data:
        clusters[np.random.random_integers(0, k-1)].append(entry)    
            
    centroids = [centroid(c) for c in clusters]

    for x in xrange(epochs):
        centroids = [centroid(c) for c in clusters]

        clusters = [[] for i in range(k)]

        for entry in data:
            d = np.array([distance.euclidean(entry, c) for c in centroids])
            i = d.argmin()
            clusters[i].append(entry)

    return clusters

data = pd.read_csv('iris.csv', header = None)
data = data.drop(4, 1)

clusters = kmeans(data.as_matrix(), 4, epochs = 10)

