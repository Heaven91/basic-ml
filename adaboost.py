# this script is used fot testing adaboost algorithm.
# adaboost use simple classifers as base classifer, then build a better
# result upon it
from numpy import *
from math import *

# used for create trainning data


def createData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1., 1., -1., -1., 1.]
    return datMat, classLabels

# a classifier, given data and the selected feature as well as threshval
# of feature, the function returns the estimated class for the testing
# data


def stumpClassify(dataMatrix, dim, threshVal, threshIeq):
    retArray = ones((shape(dataMatrix)[0], 1))
    # shape return the demensiton of matrix, and index 0 represent the rows
    # number
    if threshIeq == 'lt':
        retArray[dataMatrix[:, dim] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dim] > threshVal] = -1.0
    return retArray

# used for get each classifier


def buildStump(dataMatrix, classLabels, D):
        # dataMatrix is matrix, classLabels is a row vector, D is
        # column vector
    dataMat = mat(dataMatrix)
    labelMat = mat(classLabels)
    m, n = shape(dataMat)
    numSteps = 10.
    bestStump = {}
    bestLabelEst = mat(zeros((m, 1)))
    minErr = inf
    for i in range(n):
        minVal = dataMat[:, i].min()
        maxVal = dataMat[:, i].max()
        stepSize = (maxVal - minVal) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inEqual in ['lt', 'gt']:
                threshVal = minVal + float(j) * stepSize
                predictLabel = stumpClassify(dataMat, i, threshVal, inEqual)
                # predictLabel is a column vector
                errArray = mat(ones((m, 1)))
                # print predictLabel , labelMat
                errArray[predictLabel == labelMat.T] = 0
                # if predict is equal to given labels, then the sample is
                # labeled correctly, so the error of this sample is set to zero
                # print "sample weight is:" , D, '\n' ,"wrong classification
                # array is :" , errArray
                weightErr = D.T * errArray
                # print "spilt feature: dim %d, thresh value: %.2f, thresh inequal: %s, the weighted error is: %.4f" \
                # % (i, threshVal, inEqual, weightErr)
                if weightErr < minErr:
                    minErr = weightErr
                    bestLabelEst = predictLabel.copy()
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['ineq'] = inEqual
    return minErr, bestLabelEst, bestStump

# tiain a set of classifiers, then estimate the result class according to
# all the classifiers


def trainAdaBoost(dataArr, classLabels, numIt=40):
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # initial weight for every sample
    classLabels = mat(classLabels)
    stumpArr = []
    aggClassEst = mat(zeros((m, 1)))
    for i in xrange(numIt):
        print "the %d iteritation:" % (i)
        print "----------------------------------------------------------"
        currentErr, classEst, stump = buildStump(dataArr, classLabels, D)
        print "the current error is: %.4f" % (currentErr)
        print "the estimated classes is:", classEst.T
        print "the weight D is :", D.T
        # stumpArr.append(stump)
        # if currentErr == 0:
        # 	break
        # errArr = mat(ones((m, 1)))
        # errArr[classEst == classLabels] = 0
        # epsilon = errArr.sum() / m
        alpha = float(0.5 * log((1 - currentErr) / max(currentErr, 1e-16)))
        stump['alpah'] = alpha
        stumpArr.append(stump)
        expon = multiply(-1. * alpha * mat(classLabels).T, classEst)
        print "the index is :", expon.T
        D = multiply(D, exp(expon))     # something wrong happened here, will be removed later
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print "the weighted classifier class estimate is :", aggClassEst.T
        aggErr = multiply(sign(aggClassEst) != classLabels, ones(m, 1))
        errRate = aggErr.sum() / m
        if errRate == 0:
            break
    return stumpArr


def testAdaBoost(dataArr, setofClassifiers):
    dataMat = mat(dataArr)
    m = shape(dataArr)[0]
    aggClassEst = zeros((m, 1))
    for i in range(len(setofClassifiers)):
        classEst = stumpClassify(dataMat, setofClassifiers[i]['dim'], setofClassifiers[
                                 i]['threshVal'], setofClassifiers[i][ineq])
        aggClassEst += setofClassifiers[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)


# --------------------------test---------------
if __name__ == '__main__':
    myData, labels = createData()
    print myData, labels
    D = mat(ones((5, 1)) / 5)
    # buildStump(myData, labels, D)  # test for single step classification
    stumpArr = trainAdaBoost(myData, labels)
