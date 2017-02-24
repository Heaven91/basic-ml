from numpy import *
import operator
from os import listdir
# listdir function is used for get the file name under a directory. a list
# contain string elements


def createDataSet():
    sample = array([[1., 1.1], [1., 1], [0, 0], [0, 0.1]])
    label = ['a', 'a', 'b', 'b']
    return sample, label


def knn(testpt, samples, labels, k):
    m = shape(samples)[0]
    testArr = tile(testpt, (m, 1))
    diffArr = testArr - samples
    distanceArr = diffArr ** 2
    distanceVec = sum(distanceArr, 1)
    sortedDistanceIndex = distanceVec.argsort()
    labelCount = {}
    for i in xrange(k):
        currentLabel = labels[sortedDistanceIndex[i]]
        if not labelCount.has_key(currentLabel):
            labelCount[currentLabel] = 1
            continue
        labelCount[currentLabel] += 1
    sortedLabelCount = sorted(labelCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    # print "sorted class is:", '\n', sortedLabelCount
    return sortedLabelCount[0][0]


def file2Arr(fileName):
    fr = open(fileName)
    lineText = fr.readlines()
    m = len(lineText)
    sampleArr = zeros((m, 3))
    labels = []
    index = 0
    for line in lineText:
        lineData = line.strip().split('\t')
        sampleArr[index, :] = lineData[0:3]
        labels.append(lineData[-1])
        index += 1
    return sampleArr, labels


'''
this funciton is used for normalize a dataset, for a dataset, usually we have many features, which each column in an
array represents a feature. while, different feature have different units and value ranges. So if we treat each feature
equally, features which have a higher magnititude will bias our result. To solve this problem, we need to scale all of
our features so that they contribute equally to our cost error.cost
'''


def autoNorm(dataSet):
    tempArr = dataSet.copy()
    m, n = shape(tempArr)
    maxVal = tempArr.max(0)
    minVal = tempArr.min(0)
    featureRange = maxVal - minVal
    diffArr = tempArr - tile(minVal, (m, 1))
    retArr = diffArr / tile(featureRange, (m, 1))
    return retArr, maxVal, minVal


'''
this functions is used for testing our knn algorithm. only one input parameter is given, that's the file name which contain
our training data and test data. Using dating data as an example.
'''


def testDating(fileName):
    validataRate = .1
    dataArr, labelArr = file2Arr(fileName)
    normalData, maxVal, minVal = autoNorm(dataArr)
    print normalData
    m = shape(normalData)[0]  # total number of data
    testNum = int(m * validataRate)
    errorNum = 0
    for i in xrange(testNum):
        estimateLabel = knn(normalData[i], normalData[
                            testNum:m, :], labelArr[testNum:m], 3)
        print "the estimate label for test point %d is: %s, the real label is: %s" \
            % (i, estimateLabel, labelArr[i])
        if estimateLabel != labelArr[i]:
            errorNum += 1
            print "error"
    print "uncorrectly labeled number of is: %d, the total error rate is: %.4f" \
        % (errorNum, errorNum / float(testNum))


'''
Application used for dating match based on functions defined before. Most of part is the same as the "testDating" function,
the difference is that user is required to input the target person informations
'''


def testPerson(trainDataFileName):
        # input information
    playGameTime = float(
        raw_input("input the time spend on playing video games per year:"))
    flyMiles = float(raw_input("input the flying miles per year:"))
    iceConsume = float(
        raw_input("input the amount of ice cream consumed per year:"))
    trainArr, labelArr = file2Arr(trainDataFileName)
    normalTrainData, maxVal, minVal = autoNorm(trainArr)
    testPt = array([playGameTime, flyMiles, iceConsume])
    normalTest = (testPt - minVal) / (maxVal - minVal)
    estimateLabel = knn(normalTest, normalTrainData, labelArr, 3)
    print "the estimate label for the person is: %s" % (estimateLabel)


# ================================knn used for handwritten digits recognit

'''
for a given picture, we have a 32*32 txt file. To use our function, we need to transform it to vector. Every pixel is a feature,
every txt file is a sample.
'''


def img2vec(fileName):
    fr = open(fileName)
    retArr = zeros((1, 1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retArr[0, j + 32 * i] = int(lineStr[j])
    return retArr


def handDigitsRecognition():
    trainLabel = []
    listTrainFileName = listdir('data/trainingDigits')
    m = len(listTrainFileName)
    trainDataArr = zeros((m, 1024))
    for i in xrange(m):
        filePath = 'data/trainingDigits/' + str(listTrainFileName[i])
        trainDataArr[i, :] = img2vec(filePath)
        # extract label from file name
        fileName = listTrainFileName[i].split('.')[0]
        label = fileName.split('_')[0]
        trainLabel.append(label)
    listTestFileName = listdir('data/testDigits')
    n = len(listTestFileName)
    errorNum = 0
    for i in xrange(n):
        filePath = 'data/testDigits/' + listTestFileName[i]
        testDataArr = img2vec(filePath)
        # extract label from file name
        fileName = listTestFileName[i].split('.')[0]
        testLabel = fileName.split('_')[0]
        estimateLabel = knn(testDataArr, trainDataArr, trainLabel, 3)
        print "the estimate label for test point %d is: %s, the real label is: %s" \
            % (i, estimateLabel, testLabel)
        if estimateLabel != testLabel:
            errorNum += 1
            print "error"
    print "uncorrected recognized number is: %d, the error rate is: %.4f" %(errorNum, float(errorNum) / n)

# --------------------------------------test-----------------------------
# sample, label = createDataSet()
# print knn([0, 0], sample, label, 3)
# sample, label = file2Arr('datingTestSet2.txt')
# print "sample is:", sample
# print "label is :", label
# print autoNorm(sample)
# testDating('data/datingTestSet.txt')
# testPerson('data/datingTestSet.txt')
# handDigitsRecognition()
