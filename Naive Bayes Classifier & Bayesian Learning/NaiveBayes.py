# Written in Python 3.8.1
# Version specific features (e.g. "final" syntax) may preclude execution on earlier versions

# Kobe Davis
# Prof. Doliotis
# CS 445
# 24 February 2020
#
# Assignment 3: Bayes Classifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from typing import final

np.set_printoptions(threshold=np.inf, linewidth=500, suppress=True)

# Check for command line arguments, though does not guarantee CORRECT arguments
if len(sys.argv) != 3:
    print("\nYou must provide the training and test file as command line arguments to the program.")
    print("The first and second arguments are the training and test files respectively.\n")

# Important Constants
TRAINING_FILE: final = sys.argv[1]
TEST_FILE: final = sys.argv[2]
TINY: final = np.finfo(float).tiny

"""
Use these if the command line arguments do not work for some reason.
Simply disable the command line argument length checker on line 19.

TRAINING_FILE: final = "Datasets/yeast_training.txt"
TEST_FILE: final = "Datasets/yeast_test.txt"
"""

def main():
    print("\nConfirm that the following parameters are correct.\n")
    print(f'Training file: {TRAINING_FILE}')
    print(f'Test file: {TEST_FILE}')
    input("Press ENTER to continue.\n")

    print("\nLoading and formatting data from files...")
    inTrain, inTest, low, high = formatLoad()

    nClasses: final = high-low+1        # How many classes there are
    lblRange: final = (low, high)       # The actual class label numbers
    szAttr: final = len(inTrain[0][0])  # Number of columns for each of the classes
    totalInputs: final = sum([len(lblGroup) for lblGroup in inTrain])
    classProbL: final = np.array([[len(inTrain[i])/totalInputs for i in range(nClasses)]])

    print(f'Number of class labels: {nClasses}')
    print(f'Number of attributes: {szAttr}')

    print("\nComputing mean and stdev...")
    mu = computeMu(inTrain, szAttr, nClasses)
    sigma = computeSigma(inTrain, mu, szAttr, nClasses)

    print("\nTraining Results:")
    for i in range(nClasses):
        for j in range(szAttr-1):
                print(f'\tClass {int(mu[i][-1])}, '
                      f'\tAttribute {j+1}, '
                      f'\tMean = {np.round(mu[i][j], 2)},'
                      f'\tStandard Deviation = {np.round(sigma[i][j], 2)}')

    print("\nPredicting input classes...")
    predictions = predict(inTest[:,:-1], classProbL, mu[:,:-1], sigma[:,:-1], szAttr-1, nClasses)

    # The returned predictions matrix is interpreted and converted into an array of the
    # highest prediction scores. Following this, the array of highest prediction scores
    # is compared to an array of the actual class labels to determine the percentage of
    # correct predictions.
    maxPreds = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1) + low
    labels = inTest[:,-1:].T[0].astype(int)
    check = predictions == labels

    print("\nTest Results:")
    for i in range(len(inTest)):
        print(f'\tID={i},   '
              f'\tPredicted={predictions[i]}, '
              f'\tProbability={np.round(maxPreds[i],4)}, '
              f'\tTrue={labels[i]}, '
              f'\tAccuracy={1*check[i]}')

    check = np.mean(check)
    print(f'\nTotal Accuracy: {np.round(check*100, 2)}%')

    print("\nProgram completed successfully.\nExiting program.")

def predict(inputs, classProb, mu, sigma, cols, nClasses):
    """
    Takes test data, class probabilities, training means and stdevs, and number of classes.
    The function will use the stdevs and means to determine which class a particular input
    (from the test data) will fit into.
    A matrix of predictions is returned. Predictions are still in a format such that they
    are tied to their respective class. The highest prediction must be taken as the actual
    predictions for that input. In other words, the predictions matrix will have to be
    interpreted.
    """
    pdf = np.zeros((nClasses, len(inputs), cols))
    for i in range(nClasses):
        pdf[i] = np.exp(-np.square(inputs - mu[i])/(2*np.square(sigma[i]))) / (np.sqrt(2*np.pi) * sigma[i])

    pdf[pdf == 0] = TINY
    predictions = classProb * np.prod(pdf, axis=2, keepdims=True).T[0]
    predictions = np.log(predictions)
    predictions[predictions == np.NINF] = -745
    print("Divide by zero corrected by approximating for NINF.")

    return predictions

def computeMu(inputs, cols, nClasses):
    """
    Takes training data, number of columns, and number of class labels are
    taken as arguments. The function will compute a mean for each corresponding
    column of training inputs. This will be repeated for each class.
    Hence a matrix of means is returned.
    """
    mu = np.zeros((nClasses, cols))
    for i in range(nClasses):
        if inputs[i].size == 0:
            mu[i][-1] = mu[i-1][-1]+1
            continue
        mu[i] = np.sum(inputs[i], axis=0) / len(inputs[i])

    return mu

def computeSigma(inputs, mu, cols, nClasses):
    """
    Takes training data, training means, number of columns, and number of class
    labels taken as arguments. The function will compute a stdev for each corresponding
    mean. A row of stdevs corresponding to a mean will be genereated for each class.
    A numpy matrix of stdevs are returned.
    """
    sigma = np.zeros((nClasses, cols))
    for i in range(nClasses):
        if inputs[i].size == 0:
            continue
        sigma[i] = np.sqrt(np.sum((mu[i]-inputs[i])**2, axis=0) / len(inputs[i]))
    sigma[:,-1:] = mu[:,-1:]

    temp = sigma[:,:-1]
    temp[temp < 0.01] = 0.01
    
    return sigma

def formatLoad():
    """
    Loads training and test data from files. Data is extracted into numpy arrays.
    Test data is left in its default state, while the training data is formatted
    in such a way that the rows are sorted and grouped by their class labels.
    Sorting and grouping training data allows the program to batch compute parameters
    such as mean and stdev, this is more efficient given larger training data.
    The range of class labels are record for future use.
    The test data, training data, and range of class labels are returned from this function.
    """
    inTrain = np.array([[float(num) for num in lines.split()]
                        for lines in open(TRAINING_FILE).readlines()])
    inTest = np.array([[float(num) for num in lines.split()]
                        for lines in open(TEST_FILE).readlines()])

    inTrain = inTrain[inTrain[:,-1].argsort()]

    low = int(np.min(inTrain[:,-1]))
    high = int(np.max(inTrain[:,-1]))

    inTrain = [inTrain[inTrain[:,-1] == label, :] for label in range(low, high+1)]

    return inTrain, inTest, low, high

if __name__ == '__main__':
    main()
