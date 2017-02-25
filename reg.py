# this demo is used for build soome regression algorithms, which including
# OLS, ordinary least quare. LWLR, ridge regression, lasso regression

# ordinary least square regression
from numpy import *


def loadDataSet(fileName):
    print "load the trainning data hand test points:"
    # get the number of features, the last columns is target value
    row = open(fileName).readline().split('\t')
    nunFeat = len(row) - 1
    fr = open(fileName)
    dataMat = []
    target = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(len(curLine) - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        target.append(float(curLine[-1]))
    return dataMat, target


def standardLR(xArr, yArr):
    print "ordinay least square algorithm:"
    xMat = mat(xArr)
    yMat = mat(yArr)
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0:
        print "xTx is singular, inverse cannot be calcualted"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


'''LWLR, locally weightted linear regression. To give different weights to different samples when do prediction. We introduce
the weigthed version. Usually a gaussian kernal is choosed to decide the weights. Intutively understanding, if the prediction
point is near to one of the trained sample, when do regression, this sample should have a bigger impact on the prediction
value , so a bigger weights is given. w(i) = exp((xi - x) / sigma)'''


def lwlr(testpoint, xArr, targetArr, standardSigma=1.):
    xMat = mat(xArr)
    yMat = mat(targetArr)
    m = shape(xMat)[0]
    W = mat(eye(m))
    for i in range(m):
        W[i, i] = exp(- (xMat[i, :] - testpoint) *
                      (xMat[i, :] - testpoint).T / (2 * standardSigma ** 2))
    xWx = xMat.T * W * xMat
    if linalg.det(xWx) == 0:
        print "xWs is singular, inverse cannot be calculated"
        return
    ws = xWx.I * xMat.T * W * yMat
    return testpoint * ws


def testlwlr(testpts, xArr, yArr, standardSigma):
    print "test the locally weighted linear regression"
    m = shape(testpts)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testpts[i], xArr, yArr, standardSigma)
    return yHat

'''When we do regression, sometimes the weights coeffcients is pretty samll, which means the corresponding traind dataMat
has minor impact on the testpts. To make these weights as small as possible, ridge regression is introduced.'''


def ridgeRegres(xArr, yArr, lam=1.):
    print "ridge regression:"
    xMat = mat(xArr)
    yMat = mat(yArr)
    m = shape(xMat)[1]
    I = mat(eye(m))
    xTx = xMat.T * xMat + lam * I
    if linalg.det(xTx) == 0:
        print "ridged version xTx is singular, invese cannot be calculated"
        return
    ws = xTx.I * xMat.T * yMat
    return ws


def ridgeTest(xArr, yArr):
    print "given different lambda, return weights, besides features is normalized "
    xMat = mat(xArr)
    yMat = mat(yMat)
    dim = shape(xMat)[1]
    numTestLam = 30
    # normalization
    yMat = yMat - mean(yMat, 0)
    xmeans = mean(xMat, 0)
    xvar = var(xMat, 0)
    xMat = (xMat - xmeans) / xvar
    wMat = zeros(numTestLam, dim)
    for i in range(numTestLam):
        lam = exp(i - 10)
        ws = ridgeRegres(xMat, yMat, lam)
        wMat[i, :] = ws.T
    return wMat


def mseErr(yMat, yHat):
    diffArr = array(yMat - yHat)
    return (diffArr ** 2).sum()


def regulize(xMat):
    inMat = xMat.copy()  # note:don't change the original xmat
    xmeans = mean(inMat, 0)
    xvar = var(inMat, 0)
    return array((inMat - xmeans)) / array(xvar)

# stage regression algorithm: given weights a intinal value, then + or - a step size for ecah feature, if loss error is reduced, then update
# weights to the new value


def stageRegres(xArr, yArr, stepsize=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr)
    # m,n represent the number of samples and features respectively
    m, n = shape(xMat)
    # normalization------------
    xMat = regulize(xMat)
    # xmeans = mean(xMat, 0)
    # xvar = var(xMat, 0)
    # xMat = (xMat - xmeans) / xvar
    # print xvar, '\n', xmeans
    yMat -= mean(yMat, 0)
    retWsMat = zeros((numIt, n))

    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsBest = ws.copy()
    for i in range(numIt):
        lowestErr = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += stepsize * sign
                yTest = xMat * mat(wsTest)
                rse = mseErr(yTest, yMat)
                if rse < lowestErr:
                    lowestErr = rse
                    wsBest = wsTest
            ws = wsBest.copy()
        retWsMat[i, :] = ws.T
    return retWsMat


# ==============================test=====================================
if __name__ == '__main__':
    print "test algorithms..."
    xArr, yArr = loadDataSet('ex0.txt')
    yArr = reshape(yArr, (len(yArr), 1))
    # ws = standardLR(xArr, yArr)
    # print ws
    retMat = stageRegres(xArr, yArr)
    print retMat
