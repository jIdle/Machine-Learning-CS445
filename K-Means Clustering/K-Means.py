# Written in Python 3.8.1
# Version specific features (e.g. "final" syntax) may preclude execution on earlier versions

# Kobe Davis
# Prof. Doliotis
# CS 445
# 3 March 2020
#
# Assignment 4: K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
from typing import final

np.set_printoptions(threshold=np.inf, linewidth=500, suppress=True)

#Important constants
NUMBER_OF_CLUSTERS: final = 10
NUMBER_OF_ATTRIBUTES: final = 64
MAX_ACCURACY: final = 100
TRAINING_DATA: final = "optdigits.train"
TESTING_DATA: final = "optdigits.test"
MAX: final = np.finfo(float).max
MIN: final = np.finfo(float).min

def main():
    # Load in the training and test data from file
    inTrain, inTest = load()

    # Select the random initial seeds for the centroids
    initialSeeds = (inTrain[np.random.randint(inTrain.shape[0], size=(5, NUMBER_OF_CLUSTERS))])[:,:,:-1]
    record = []

    # Loop 5 times and choose the clustering resulting from the run with the best AMSE
    for i in range(5):
        centroids = initialSeeds[i]
        continueLoop = True
        while continueLoop:
            clusteringIndices = assignCluster(inTrain[:,:-1], centroids, NUMBER_OF_CLUSTERS)
            clustering = [(inTrain[cluster])[:,:-1] for cluster in clusteringIndices]
            centroids, continueLoop = updateCentroids(clustering, centroids)

        # Compute the average mean squared error and store it
        amse = averageMeanSquareError(clustering, centroids, NUMBER_OF_CLUSTERS)
        record.append((clustering, clusteringIndices, centroids, amse))
    

    # Sort stored clusterings by their average mean squared error, select the first one because it has lowest error
    record.sort(key=lambda x: x[3])
    record = record[0]
    bestClustering = {'Clustering': record[0], 'Indices': record[1], 'Centroids': record[2], 'AMSE': record[3]}

    # Compute more error terms
    bestClustering['MSS'] = meanSquareSeparation(bestClustering['Centroids'], NUMBER_OF_CLUSTERS)
    bestClustering['AE'] = averageEntropy([(inTrain[cluster]) for cluster in bestClustering['Indices']], NUMBER_OF_CLUSTERS)

    print(f'Average Mean Squared Error: {bestClustering["AMSE"]}\n'
          f'Mean Squared Separation: {bestClustering["MSS"]}\n'
          f'Mean Entropy: {bestClustering["AE"]}\n')

    # Compute modes for each cluster's vector's labels, highest frequency represents that cluster
    clusterModes = [(i, mode((inTrain[(bestClustering['Indices'])[i]])[:,-1])[0][0]) for i in range(NUMBER_OF_CLUSTERS)]
    clusterModes.sort(key=lambda x: x[1])

    # Associate cluster centroids with cluster labels
    centroidLabels = [(clusterModes[i][1], (bestClustering['Centroids'])[clusterModes[i][0]]) for i in range(NUMBER_OF_CLUSTERS)]

    # Determine accuracy of the clustering on the test data
    successRate, confMatrix = testAcc(inTest, centroidLabels)
    print(f'Success Rate: {successRate}%\n'
          f'Confusion Matrix:\n{confMatrix}')

    # Reshape centroids so that they can be displayed as images
    reshapedMeans = [bestClustering['Centroids'][i].reshape((8,8)) for i in range(NUMBER_OF_CLUSTERS)]

    for i in range(NUMBER_OF_CLUSTERS):
        plt.matshow(reshapedMeans[i])
        plt.show()

def testAcc(vectors, centroidLabels):
    """
    Determines accuracy of the chosen clustering on the test data.
    Output the accuracy as a percentage and a confusion matrix.
    """

    successRate = 0
    confPred = np.zeros(vectors.shape[0])
    confAct = np.zeros(vectors.shape[0])
    idx = 0
    for v in vectors:
        shortest = MAX
        label = None
        for cl in centroidLabels:
            dist = euclideanDistance(v[:-1], cl[1])
            if dist < shortest:
                shortest = dist
                label = cl[0]
        if label == v[-1]:
            successRate += 1
        confPred[idx] = label
        confAct[idx] = v[-1]
        idx += 1

    pdPred = pd.Series(confPred, name='Predicted')
    pdAct = pd.Series(confAct, name='Actual')
    return np.round((successRate / vectors.shape[0]) * 100, 2), pd.crosstab(pdPred, pdAct)

def classProbabilities(cluster, K):
    """
    Determines the probability that a digit will be inside a
    cluster given the data currently inside the cluster.
    This function is used by the entropy function.
    Returns a list of probabilities for each class's occurence within a cluster.
    """

    return np.bincount(cluster[:,-1:].ravel(), minlength=K)/cluster.shape[0]

def entropy(cluster, K):
    """
    Returns the entropy of a cluster.
    """

    entropySum = 0
    probs = classProbabilities(cluster, K)
    for j in range(K):
        if probs[j] < 5e-320:
            continue
        entropySum += probs[j]*np.log2(probs[j])
    return -entropySum

def averageEntropy(clustering, K):
    """
    Returns the average entropy of a cluster.
    """

    aeSum = 0
    totalSize = sum(cluster[:,:-1].size for cluster in clustering)
    for cluster in clustering:
        aeSum += (cluster[:,:-1].size/totalSize) * entropy(cluster, K)
    return aeSum

def euclideanDistance(vectorL, vectorR):
    """
    Takes two vectors as arguments.
    Returns the distance between two vectors in n dimensional space.
    """
    if vectorL.size == 0 or vectorR.size == 0:
        return None
    return np.linalg.norm(vectorL - vectorR)

def meanSquareSeparation(centroids, K):
    """
    Determines the mean squared separation of a group of clusters using their centroids.
    Takes set of centroids as argument.
    Returns the aforementioned computed term.
    """
    mssSum = 0
    offset = 0
    for centroidL in centroids:
        offset += 1
        if centroidL.size == 0:
            continue
        for centroidR in centroids[offset:]:
            if centroidR.size == 0:
                continue
            mssSum += np.square(euclideanDistance(centroidL, centroidR))
    return mssSum / (K*(K-1))/2

def meanSquareError(cluster, centroid):
    """
    Computes the mean square error of a cluster.
    Takes a cluster and its centroid as arguments.
    This function is used by the average mean squared error function.
    Returns the aforementioned computed term.
    """
    mseSum = 0
    for point in cluster:
        if point.size == 0:
            continue
        mseSum += np.square(euclideanDistance(point, centroid))
    return mseSum / cluster.size

def averageMeanSquareError(clustering, centroids, K):
    """
    Computes and returns the average mean squared error accross all clusters.
    Takes all centroids and clusters as arguments.
    """

    amseSum = 0
    for cluster, centroid in zip(clustering, centroids):
        amseSum += meanSquareError(cluster, centroid)
    return amseSum / K

def computeCentroid(cluster):
    """
    Determines and returns the centroid of a cluster.
    This function is used by the updateCentroids function.
    """

    return np.mean(cluster, axis=0)

def updateCentroids(clustering, centroids):
    """
    Receives a clustering as an argument and determines the centroids of all contained clusters.
    Returns a list of updated centroids.
    """
    updated = np.array([computeCentroid(cluster) for cluster in clustering])
    if np.array_equal(centroids, updated):
        return updated, False
    return updated, True

def closestCentroid(vector, centroids, K):
    """
    Takes a vector and all centroids.
    Determines the centroid that is closest to the received vector.
    Utilises the euclideanDistance function to accomplish this task.
    Returns the index of that centroid to its calling function (assignCluster).
    """

    idx = 0
    shortest = euclideanDistance(vector, centroids[0])
    for i in range(1, K):
        newDist = euclideanDistance(vector, centroids[i])
        if newDist < shortest:
            shortest = newDist
            idx = i
    return idx
    
def assignCluster(vectors, centroids, K):
    """
    Given a set of vectors and a set of centroids.
    This function determines the closest centroid to each vector and 
    assigns them a group associated to that centroid.
    This groups all vectors with attributes that place them closely
    to a centroid together in the same group (cluster).
    Returns a clustering. For performance sake only the indexes to these vectors are grouped.
    """

    clusteringIndices = []
    for i in range(K):
        clusteringIndices.append([])

    size = vectors.shape[0]
    for i in range(size):
        clusterIdx = closestCentroid(vectors[i], centroids, K)
        clusteringIndices[clusterIdx].append(i)

    return clusteringIndices

def load():
    """
    Loads the test and training data from file and into numpy arrays.
    Returns these numpy arrays back to main.
    """
    inTrain = np.array([[int(num) for num in lines.split(',')]
                        for lines in open(TRAINING_DATA).readlines()])
    inTest = np.array([[int(num) for num in lines.split(',')]
                       for lines in open(TESTING_DATA).readlines()])
    return inTrain, inTest

if __name__ == '__main__':
    main()