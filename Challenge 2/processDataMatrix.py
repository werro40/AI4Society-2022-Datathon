# This script processes the x, y, z data into 3 numpy binary files
import numpy as np
from itertools import product
import pandas as pd

def matrixDims():
    numSim = 24
    numRec = 127
    numPoints = 30000
    return numSim, numRec, numPoints

def generateMat(fName):
    memmapKwargs = {
        "dtype": np.single,
        "mode": "w+",
        "shape": matrixDims(),
        "order":'C'
    }
    return np.memmap(fName,**memmapKwargs)

def simDataDir(simNum, recNum):
    return f"../data/train/simulation_{simNum}/receiver_{recNum}.csv"

def eventDataDir(simNum):
    return f"../data/train/simulation_{simNum}_output.csv"

def loadMatrix(fName, dataHeader):
    mmat = generateMat(fName)
    numSim, numRec, _ = matrixDims()

    for sInd,rInd in product(range(numSim),range(numRec)):
        print(f"{fName}: Processing Sim {sInd + 1}, Receiver {rInd}")
        tempFrame = pd.read_csv(simDataDir(sInd+1,rInd))
        mmat[sInd,rInd,:] = tempFrame[dataHeader].values
    return mmat

def loadMatrixS(fName):
    mmat = generateMat(fName)
    numSim, numRec, _ = matrixDims()

    for sInd,rInd in product(range(numSim),range(numRec)):
        print(f"{fName}: Processing Sim {sInd + 1}, Receiver {rInd}")
        tempFrame = pd.read_csv(simDataDir(sInd+1,rInd))
        mmat[sInd,rInd,:] = np.sqrt(tempFrame['X'].values**2 +
                                    tempFrame['Y'].values**2 +
                                    tempFrame['Z'].values**2)
    return mmat

def main():
    loadMatrixS("../cleanData/sMat")
    loadMatrix("../cleanData/xMat","X")
    loadMatrix("../cleanData/yMat","Y")
    loadMatrix("../cleanData/zMat","Z")

if __name__ == "__main__":
    main()