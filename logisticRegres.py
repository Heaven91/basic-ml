# this script contains some  functions inplemented for logistic regression.
# coding-utf-8
from numpy import *


def loadData(fileName):
    fr = open(fileName)
    dataArr = []
    label = []
    for line in fr.readlines():
        lineStr = line.strip().split('\t')
        dataArr.append([1., float(lineStr[0]), float(lineStr[1])])
        label.append(int(lineStr[-1]))
    label = array(label)
    return dataArr, label


def sigmoid(inX):
    return 1. / (1 + exp(-inX))


def gradDecent(dataArr, target, stepsize=0.001):
    dataMat = mat(dataArr)
    # when an one dimentional list is transfromed to a matrix, usually get a
    # row vector
    targetVal = mat(target).T
    m, n = shape(dataMat)
    weights = mat(ones((n, 1)))
    maxCycles = 500
    for i in xrange(maxCycles):
        inX = dataMat * weights
        predict = sigmoid(inX)
        error = targetVal - predict
        weights = weights + stepsize * dataMat.T * error
    return weights


def stocasGrad(dataArr, target, stepSize=0.001):
    dataArr = array(dataArr)
    m, n = shape(dataArr)
    weights = ones((n))
    maxCycles = 500
    for i in xrange(maxCycles):
        for j in xrange(m):
            inX = sum(dataArr[j, :] * weights)
            predict = sigmoid(inX)
            error = target[j] - predict
            weights += stepSize * error * dataArr[j, :]
    return weights


'''
To prevent fluctuation of weights resulting from sequential choose of samples, stocasGrad0() will choose
samples randomly
'''


def stocasGrad0(dataArr, target, stepSize=0.001):
    dataArr = array(dataArr)
    m, n = shape(dataArr)
    weights = ones((n))
    maxCycles = 50000
    for i in xrange(maxCycles):
        j = random.randint(m)
        inX = sum(dataArr[j, :] * weights)
        predict = sigmoid(inX)
        error = target[j] - predict
        weights += stepSize * error * dataArr[j, :]
    return weights


def splitLinePlot(dataArr, target, weights):
    import matplotlib.pyplot as plt
    coordinateX0 = []
    coordinateY0 = []
    coordinateX1 = []
    coordinateY1 = []
    dataArr = array(dataArr)
    m = shape(dataArr)[0]
    for i in xrange(m):
        if target[i] == 1:
            coordinateX0.append(dataArr[i, 1])
            coordinateY0.append(dataArr[i, 2])
        else:
            coordinateX1.append(dataArr[i, 1])
            coordinateY1.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coordinateX0, coordinateY0, s=30, c='red')
    ax.scatter(coordinateX1, coordinateY1, s=30, c='green')
    plt.show()

# =================================test===================
dataArr, label = loadData('data/logisTestSet.txt')
weights = stocasGrad0(dataArr, label)
print "weights are:", weights
splitLinePlot(dataArr, label, weights)
