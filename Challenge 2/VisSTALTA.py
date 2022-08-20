# This script visualizes the STA/LTA on the test dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def matrixDims():
    numSim = 24
    numRec = 127
    numPoints = 30000
    return numSim, numRec, numPoints

def testDataDir(recNum):
    return f"../data/test/simulation_0/receiver_{recNum}.csv"

def genMatrix():
    _, numRec, numPoints = matrixDims()
    return np.zeros([numRec, numPoints])

def genSimData():
    sMatrix = genMatrix()
    for rec in range(127):
        tempFrame = pd.read_csv(testDataDir(rec))

        #xData = np.array(tempFrame['X'].values)
        #yData = np.array(tempFrame['Y'].values)
        zData = np.array(tempFrame['Z'].values)
        #sData = np.sqrt(xData**2 + yData**2 + zData**2)
        #sMatrix[rec,:] = sData
        sMatrix[rec,:] = zData
    return sMatrix

def transformDataRec(window, inArr, outArr, rInd):
    _, _, numTime = matrixDims()
    staTerm = lambda ind: np.sum(inArr[rInd, ind:ind+window]**2)
    ltaTerm = lambda ind: np.sum(inArr[rInd, ind-window:ind]**2)
    enRatio = lambda ind: staTerm(ind)/ltaTerm(ind) 
    modEnRatio = lambda ind: (np.abs(inArr[rInd, ind])*enRatio(ind))**3
    
    timeSlice= slice(window-1,-window)
    timePoints = range(numTime)[timeSlice]
    outArr[rInd, timeSlice] = np.array([enRatio(ind) for ind in timePoints])

def transformData(window, inArr):
    _, nRec, nTime = matrixDims()
    outputMatrix = np.zeros([nRec, nTime])
    for i in range(nRec):
        transformDataRec(window, inArr, outputMatrix, i)
    return outputMatrix

def genPlotter(data, window):
    newData = transformData(window, data)
    def plotSim(startVal,endVal):
        plt.imshow(newData[:, startVal:endVal],aspect="auto")
        plt.title("STA/LTA of z data")
        plt.xlabel("Time")
        plt.ylabel("Receiver")
        plt.show()
    return plotSim

def main():
    rawData = genSimData()
    plotterFunc = genPlotter(rawData, 100)
    while True:
        startNum = int(input("Enter the start index:"))
        endNum = int(input("Enter the end index:"))
        plotterFunc(startNum, endNum)

if __name__ == "__main__":
    main()