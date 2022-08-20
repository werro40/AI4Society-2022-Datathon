# This script processes the provided data into a pandas dataframe
import pandas as pd
import numpy as np
import sqlite3 as db

from functools import reduce
from itertools import product

def simDataDir(simNum, recNum):
    return f"../data/train/simulation_{simNum}/receiver_{recNum}.csv"

def eventDataDir(simNum):
    return f"../data/train/simulation_{simNum}_output.csv"

def genEventFrame(simNum):
    return pd.read_csv(eventDataDir(simNum))

def modEventFrameSim(simNum):
    initFrame = genEventFrame(simNum)
    eventTimes = lambda ind: initFrame[f"Receiver_{ind}"].values
    realTimes = lambda ind: [x for x in eventTimes(ind) if x != -1]

    realArray = reduce(lambda x,y: x+realTimes(y), range(127), [] )
    recArray = reduce(lambda x,y: x+[y]*len(realTimes(y)), range(127), [])
    simArray = [simNum]*len(recArray)
    dataStruct = {"simulation": simArray,
                  "receiver": recArray,
                  "eventTime": realArray}
    return pd.DataFrame(dataStruct, dtype=int)

def modEventFrame():
    dataStruct = {"simulation": [],
                  "receiver": [],
                  "eventTime": []}
    emptyFrame = pd.DataFrame(dataStruct, dtype=int)
    pdjoin = lambda x,y: pd.concat([x, modEventFrameSim(y)])
    return reduce(pdjoin, range(1,25), emptyFrame)

def addMagFrame():
    initFrame = modEventFrame()
    cond1 = lambda recNum: initFrame['receiver']==recNum
    cond2 = lambda simNum: initFrame['simulation']==simNum
    
    simList = []
    recList = []
    timeList = []
    xSeries = []
    ySeries = []
    zSeries = []
    for sim,rec in product(range(1,25), range(127)):
        print(f"Processing Sim {sim}, Receiver {rec}")
        tVals = initFrame[(cond1(rec)) & (cond2(sim))]["eventTime"].values
        tempFrame = pd.read_csv(simDataDir(sim,rec))
        cond3 = lambda timeVal: tempFrame['Timestep']==timeVal
        for tval in tVals:
            selFrame = tempFrame[cond3(tval)]
            simList += [sim]
            recList += [rec]
            timeList += [tval]
            xSeries += [selFrame['X'].values[0]]
            ySeries += [selFrame['Y'].values[0]]
            zSeries += [selFrame['Z'].values[0]]
    sSeries = np.sqrt(np.array(xSeries)**2 + np.array(ySeries)**2 + np.array(zSeries)**2)

    
    return pd.DataFrame({
        "Sim": simList,
        "Rec": recList,
        "EventTime": timeList,
        "X": xSeries,
        "Y": ySeries,
        "Z": zSeries,
        "S": sSeries
    })
        
def main():
    finalFrame = addMagFrame()
    finalFrame.to_csv("../cleanData/eventMotion.csv")

    with db.connect('../cleanData/eventMotion.db') as con:
        finalFrame.to_sql("eventMotion",con)

if __name__ == "__main__":
    main()

