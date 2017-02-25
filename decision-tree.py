

from math import log


def calShannonEnt(dataset):
    numEnties = len(dataset)
    classCounts = {}
    for entry in dataset:
        classLabel = entry[-1]
        if classLabel not in classCounts.keys():
            classCounts[classLabel] = 0
        classCounts[classLabel] += 1

    shannonEnt = 0
    for keys in classCounts:
        prob = float(classCounts[keys]) / numEnties
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDateSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# spilt data according to slected features in each column


def spiltDataSet(dataSet, feature, value):
    subDataSet = []
    for featVec in dataSet:
        if featVec[feature] == value:
            newFeatVec = featVec[:feature]
            newFeatVec.extend(featVec[feature + 1:])
            subDataSet.append(newFeatVec)
    return subDataSet


def chooseBestSpiltFeat(dataSet):
    numFeat = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeat = -1
    for i in xrange(numFeat):
        valueList = [example[i] for example in dataSet]
        uniqueValue = set(valueList)
        newEntropy = 0
        for value in uniqueValue:
            subDataSet = spiltDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat


# -----------------------------------test------------------------------
if __name__ == '__main__':
    myData, labels = createDateSet()
    myData
    print calShannonEnt(myData)
    print spiltDataSet(myData, 0, 0)
    print 'the best spilt feature is', chooseBestSpiltFeat(myData)
