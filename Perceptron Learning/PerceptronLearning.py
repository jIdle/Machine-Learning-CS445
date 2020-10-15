# Kobe Davis (901199860)
# Prof. Doliotis
# CS 445
# 1/24/2020

import numpy
import matplotlib.pyplot as plt

def main():
    print("\nLoading data from files...")

# Open file and extract lines into list
    trainLines = open("mnist_train.csv").readlines()
    testLines = open("mnist_test.csv").readlines()

# Image data is extracted from 'train/testLines' into numpy arrays 'train/testInputs'
# Variables 'train/testInputs' are 2D numpy arrays where each row is comprised of image pixels
    trainInputs = numpy.array([[int(num) for num in lines.split(',')] for lines in trainLines])
    numpy.random.shuffle(trainInputs)
    testInputs = numpy.array([[int(num) for num in lines.split(',')] for lines in testLines])
    numpy.random.shuffle(testInputs)

# Set important constant dimensions for perceptron learning algorithm
    cols = len(trainInputs[0])  # Number of trainInputs to the NN
    rows = 10                   # Number of outputs from the NN
    trainSet = len(trainInputs) # Number of training examples
    testSet = len(testInputs)   # Number of test examples
    numEpochs = 50              # Number of epochs to run
    lRate = 0.001                # How fast the NN should learn

# Initialize a 2D numpy array with random weights between -0.05 and 0.05
    weights = numpy.random.uniform(-0.05, 0.05, (rows, cols))

# 'trainDesired' and 'testDesired' are the vector forms of the ground truth NN outputs
# Both variables are 2D numpy arrays where each row is a NN vector output
    trainDesired = numpy.zeros((trainSet, rows))
    for i in range(trainSet):
        trainDesired[i][trainInputs[i][0]] = 1

    testDesired = numpy.zeros((testSet, rows))
    for i in range(testSet):
        testDesired[i][testInputs[i][0]] = 1

    print("Preprocessing...")

# Set first input (pixel) of each image to 1 so that bias (weight_0) works properly
# Also scale each input (pixel) down to be between 0 and 1
    for img in trainInputs:
        img = img/255
        img[0] = 1
    for img in testInputs:
        img = img/255
        img[0] = 1

    print("Beginning epochs...")

    trainAcc = []
    testAcc = []
    for i in range(numEpochs):
        print("Epoch", i)
        print("\tTraining network...")
        order = numpy.arange(trainSet)
        numpy.random.shuffle(order)
        trainInputs = trainInputs[order]
        trainDesired = trainDesired[order]

        epoch(trainSet, rows, lRate, weights, trainInputs, trainDesired)

        print("\tComputing accuracy...")
        order = numpy.arange(testSet)
        numpy.random.shuffle(order)
        testInputs = testInputs[order]
        testDesired = testDesired[order]

        trainAcc.append(test(trainSet, rows, weights, trainInputs, trainDesired))
        testAcc.append(test(testSet, rows, weights, testInputs, testDesired))
    
    xAxis = list(range(0, 50))
    plt.figure()
    plt.suptitle('Handwritten Digit Classification')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.xlim(0, 50)
    plt.xlabel('Epochs')
    plt.plot(xAxis, trainAcc, label='Training')
    plt.plot(xAxis, testAcc, label='Test')
    plt.legend()
    plt.savefig('Plot_001')

    print("\nResults:\n")
    for i in range(numEpochs):
        print("Epoch", i)
        print("\tTraining Accuracy: ", trainAcc[i])
        print("\tTest Accuracy: ", testAcc[i])

    confMatrix = numpy.zeros((10,10))
    confusion(testSet, weights, testInputs, testDesired, confMatrix)
    print("\nConfusion Matrix:")
    print(confMatrix)
   
# Function for generating the confusion matrix
def confusion(dataset, weights, inputs, desired, matrix):
    for i in range(dataset):
        outputs = numpy.array([numpy.dot(inputs[i], x) for x in weights])
        predicted = numpy.argmax(outputs)
        actual = numpy.argmax(desired[i])
        matrix[actual][predicted] += 1 

# Function for computing accuracy
def test(dataset, rows, weights, inputs, desired): 
    correct = 0
    for i in range(dataset):
        outputs = numpy.array([numpy.dot(inputs[i], x) for x in weights])
        index = numpy.argmax(outputs)
        outputs = numpy.zeros(rows)
        outputs[index] = 1

        if numpy.array_equal(desired[i], outputs) == True:
            correct += 1
 
    accuracy = (correct/dataset)*100
    return accuracy

# Function for updating nn weights
def epoch(dataset, rows, lRate, weights, inputs, desired):
    for i in range(dataset):
        outputs = numpy.array([numpy.dot(inputs[i], x) for x in weights])
        index = numpy.argmax(outputs)
        outputs = numpy.zeros(rows)
        outputs[index] = 1

        inner = lRate*(desired[i] - outputs)
        for j in range(rows):
            weights[j] = weights[j] + (inner[j]*inputs[i])
    
if __name__ == '__main__':
    main()
