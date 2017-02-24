# unsuperviesd classification

from numpy import *


def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(line[0]), float(line[1])])
    return dataMat


def eculdDistance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCentor(dataSet, k):
    dataArr = array(dataSet)
    n = shape(dataArr)[1]
    centorArr = zeros((k, n))
    maxVal = dataArr.max(0)
    minVal = dataArr.min(0)
    valRange = maxVal - minVal
    centorArr = tile(minVal, (k, 1)) + \
        tile(valRange, (k, 1)) * random.rand(k, n)
    return centorArr


def kmeansCluster(dataArr, k):
    dataArr = array(dataArr)
    ptClassChanged = True
    m = shape(dataArr)[0]
    clusterIndex = zeros((m, 2)) - 1
    centor = randCentor(dataArr, k)
    while ptClassChanged:
        ptClassChanged = False
        for i in xrange(m):
            minDist = inf
            minIndex = -1
            for j in xrange(k):
                retDist = eculdDistance(dataArr[i, :], centor[j, :])
            if retDist < minDist:
                minDist = retDist
                minIndex = j
                if clusterIndex[i, 0] != minIndex:
                    ptClassChanged = True
            clusterIndex[i, :] = array([minIndex, minDist])
            print centor

        for cluster in xrange(k):
            kclusterData = dataArr[nonzero(clusterIndex[:, 0] == cluster)[0]]
            centor[cluster, :] = kclusterData.mean(0)
    return centor, clusterIndex


def clusterPlot(dataArr, centor, clusterindex):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[:, 0], dataArr[:, 1], c=20 *
               clusterindex[:, 0], s=50, edgecolors='white')
    ax.scatter(centor[:, 0], centor[:, 1], s=70, marker='s', c='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kmeans cluster')
    plt.legend()
    plt.show()


# =============================TEST===========================
dataMat = loadDataSet('data/kmeans_testSet.txt')
# centor = randCentor(dataMat, 4)
a, b = kmeansCluster(dataMat, 4)
print a, '\n', b
clusterPlot(array(dataMat), a, b)
