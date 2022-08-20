# This script linearly interpolates the visually inspected data and
# generates the kaggle submission script

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

detSeries = [0,20,40,60,80,100,126]

# visually determined values from STALTA plot
tSeries = [
    [477,477,539,671,793,912,1065],
    [5905, 5846, 5694, 5504, 5456, 5424, 5543],
    [11089, 11012, 10802, 10732, 10626, 10516, 10538],
    [22898, 22840, 22828, 22862, 22951, 23097, 23231]
]
def genInterp(xSet, ySet):
    interpFunc = lambda x: np.interp(x, xSet, ySet)
    return interpFunc
funcSeries = [genInterp(detSeries, tSet) for tSet in tSeries]
detectors = list(range(127))
dataSeries = [funcS(np.array(detectors)) for funcS in funcSeries]

for dataS in dataSeries:
    plt.plot(detectors, dataS)
plt.show()

completeData = -1*np.ones([127, 5])
for ind in range(4):
    completeData[:,ind] = dataSeries[ind]

id = []
pred = []
for recVal, eventInd in product(range(127), range(5)):
    id.append(f"R{recVal}-E{eventInd}")
    predVals = [int(val) for val in completeData[recVal,:]]
    pred.append(predVals[eventInd])

output = pd.DataFrame({
    "Id": id,
    "Predicted": pred
})

output.to_csv("submit15.csv",index=False)