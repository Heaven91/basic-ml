from numpy import *

def loadDdataSet(fileName):
	fr = open(fileName)
	dataSet = []
	for curLine in fr.readlines():
		curLine = curLine.strip().split('\t')
		dataSet.append(curLine)

	return dataSet


def binarySplitData(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
