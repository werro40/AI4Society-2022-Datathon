# This script implements the earthquake classifier and presents the 
# results on the test data set
import numpy as np
from itertools import product
import pandas as pd
from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def matrixDims():
    numSim = 24
    numRec = 127
    numPoints = 30000
    return numSim, numRec, numPoints

def timeSliceSetup(numPoints):
    numTimeSlices = 100
    timeSliceLength = int(numPoints/numTimeSlices)
    return numTimeSlices, timeSliceLength

def getMat(fName):
    readKwargs = {
        "dtype": np.single,
        "mode": "c",
        "shape": matrixDims(),
        "order":'C'
    }
    return np.memmap(fName,**readKwargs)

def getMatrices(xFile, yFile, zFile):
    return tuple([ getMat(fName) for fName in [xFile, yFile, zFile]])

def genEventChecker(df):
    @lru_cache
    def isEvent(timeStart, timeStop, sInd):
        cond1 = df["EventTime"] >= timeStart
        cond2 = df["EventTime"] < timeStop
        cond3 = df["Sim"] == sInd + 1
        filterFrame = df[(cond1) & (cond2) & (cond3)] 
        return 1 if np.shape(filterFrame)[0] > 0 else 0
    return isEvent

def getStats(xMat, yMat, zMat, eventDF):
    numSim, numRec, numPoints = matrixDims()
    numTimeSlices, timeSliceLength = timeSliceSetup(numPoints)
    simId = []
    timeStart = []
    timeStop = []
    eventOccur = []
    dataDict = lambda: {"min":[], "max":[], "mean":[], "std":[]} 
    xStats = dataDict()
    yStats = dataDict()
    zStats = dataDict()
    eventChecker = genEventChecker(eventDF)
    for sInd, timeSlice in product(range(numSim),range(numTimeSlices)):
        startSlice = timeSlice*timeSliceLength
        endSlice = startSlice + timeSliceLength
        sliceInd = (sInd, slice(None), slice(startSlice, endSlice))
        eventOccur.append(eventChecker(startSlice, endSlice, sInd))

        for dataMat, statDict in zip([xMat, yMat, zMat], [xStats, yStats, zStats]):
            dataSlice = dataMat[sliceInd]
            simId.append(sInd+1)
            timeStart.append(startSlice)
            timeStop.append(endSlice)
            statDict["min"].append(np.min(dataSlice))
            statDict["max"].append(np.max(dataSlice))
            statDict["mean"].append(np.mean(dataSlice))
            statDict["std"].append(np.std(dataSlice))
    
    inputData = np.zeros([numSim*numTimeSlices, 12])
    for ind, dictVal in enumerate([xStats, yStats, zStats]):
        inputData[:,4*ind] = dictVal['min']
        inputData[:,4*ind+1] = dictVal['max']
        inputData[:,4*ind+2] = dictVal['mean']
        inputData[:,4*ind+3] = dictVal['std']
    outputdata = np.array(eventOccur)
    return inputData, outputdata

def genTestStats(xMat, yMat, zMat):
    _, _, numPoints = matrixDims()
    numTimeSlices, timeSliceLength = timeSliceSetup(numPoints)

    dataDict = lambda: {"min":[], "max":[], "mean":[], "std":[]} 
    xStats = dataDict()
    yStats = dataDict()
    zStats = dataDict()
    for timeSlice in range(numTimeSlices):
        startSlice = timeSlice*timeSliceLength
        endSlice = startSlice + timeSliceLength
        sliceInd = (slice(None), slice(startSlice, endSlice))

        for dataMat, statDict in zip([xMat, yMat, zMat], [xStats, yStats, zStats]):
            dataSlice = dataMat[sliceInd]
            statDict["min"].append(np.min(dataSlice))
            statDict["max"].append(np.max(dataSlice))
            statDict["mean"].append(np.mean(dataSlice))
            statDict["std"].append(np.std(dataSlice))

    inputData = np.zeros([numTimeSlices, 12])
    for ind, dictVal in enumerate([xStats, yStats, zStats]):
        inputData[:,4*ind] = dictVal['min']
        inputData[:,4*ind+1] = dictVal['max']
        inputData[:,4*ind+2] = dictVal['mean']
        inputData[:,4*ind+3] = dictVal['std']
    return inputData

def genTestData(fDir):
    _, numRec, numPoints = matrixDims()
    xData = np.zeros([numRec, numPoints])
    yData = np.zeros([numRec, numPoints])
    zData = np.zeros([numRec, numPoints])
    for i in range(numRec):
        tdf = pd.read_csv(f"{fDir}/receiver_{i}.csv")
        xData[i,:] = np.array(tdf['X'].values)
        yData[i,:] = np.array(tdf['Y'].values)
        zData[i,:] = np.array(tdf['Z'].values)
    return  xData, yData, zData

def main():
    xFile = "../cleanData/xMat"
    yFile = "../cleanData/yMat"
    zFile = "../cleanData/zMat"
    eventFile = "../cleanData/eventMotion.csv"
    testDir = "../data/test/simulation_0"

    xMat, yMat, zMat = getMatrices(xFile, yFile, zFile)
    eventDF = pd.read_csv(eventFile)
    
    xMatTest, yMatTest, zMatTest = genTestData(testDir)
    trainData = genTestStats(xMatTest, yMatTest, zMatTest)
    
    iData, oData = getStats(xMat, yMat, zMat, eventDF)

    
    iTrain, iTest, oTrain, oTest = train_test_split(iData, oData)
    clf = LogisticRegression(random_state=0).fit(iTrain, oTrain)
    print(f"Model Training Evaluation: {clf.score(iTest, oTest)}")

    numTimeSlices, timeSliceLength = timeSliceSetup(30000)
    timeVals = [timeSliceLength*(float(i)+0.5) for i in range(numTimeSlices)]
    predVals = clf.predict(trainData)
    for t,v in zip(timeVals, predVals):
        if v > 0.5:
            print(f"Possible event at {t}")
    plt.plot(timeVals, predVals)
    plt.title("Earthquake Prediction")
    plt.xlabel("Time")
    plt.show()
    print(clf.coef_)

if __name__ == "__main__":
    main()
