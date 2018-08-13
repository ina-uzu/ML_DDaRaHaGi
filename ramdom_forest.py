# -*- coding: ms949 -*-
import numpy
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

data = open("winequality-red.csv", "r")

xList = []
labels = []
names = []

firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        # Split with ";"
        row = line.strip().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

# print(xList);print(labels)

nrows = len(xList)
ncols = len(xList[0])

X = numpy.array(xList)
Y = numpy.array(labels)
wineNames = numpy.array(names)
# print(X);print(Y);print(wineNames)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)
# print(xTrain);print(xTest);print(yTrain);print(yTest)

mseOos = []
nTreeList = range(50, 500, 10)
for iTrees in nTreeList:
    depth = None
    maxFeat = 4  # 조정해볼 것
    wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
                                                 max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)
    wineRFModel.fit(xTrain, yTrain)
    # 데이터 세트에 대한 MSE 누적
    prediction = wineRFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))

print("MSE")
print(mseOos[-1])

plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
# plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()

featureImportance = wineRFModel.feature_importances_

featureImportance = featureImportance / featureImportance.max()
sorted_idx = numpy.argsort(featureImportance)
barPos = numpy.arange(sorted_idx.shape[0]) + .5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, wineNames[sorted_idx])
plot.xlabel('Variable Importance')

plot.show()
