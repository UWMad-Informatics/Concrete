# Create by Michael Hibbard November 4, 2016

# import necessary libraries
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random as random

# Do a linear regression in which there are 2 predictors for a single variable

# Initialize the predictors and the dependent variable
iValues = np.array([])
jValues = np.array([])
yValues = np.array([])

# For ease, have user input for size of data
dataSize = int(input("How large should the data set be? Use a factor of 10."))

# Add 100 terms to each of the predictors with an error term of +-4
for q in range(0, dataSize):

    iValue = int(random.random() * 1000)
    jValue = int(random.random() * 1000)
    iValues = np.append(iValues, [iValue])
    jValues = np.append(jValues, [jValue])

    Error = -1000 + 2000 * random.random()

    # It doesn't really matter, so let's just say y = 3*i + 5*j + 4
    yValue = 3*iValue + 5*jValue + 4 + Error
    yValues = np.append(yValues, [yValue])

# Repeat mean and standard deviation of the y values
print("")
print("-----GENERATING DATA SET-----")
print("The mean of the data is", yValues.mean(), "and the standard deviation is", yValues.std(), ".")
print("-----------------------------")
print("")

# Make a quick histogram of the data:
yBins = np.array([])
numberOfBins = 25
binRange = (yValues.max() - yValues.min())/25
yBinMarker = yValues.min()
for yIndex in range(0, numberOfBins):
    yBins = np.append(yBins, [yBinMarker])
    yBinMarker += binRange
dependentYHist = plt.figure(0)
plt.hist(yValues, yBins, histtype='bar')
dependentYHist.suptitle('Histogram of Y Values')

numberOfTests = int(input("How many CV tests should be done on the data set?"))

# Initialize an array to store the RMSE values in.
RMSEValues = np.array([])
# Initialize values for the best and worst RMSE
bestRMSE = 1000
bestRMSEData = None
bestRMSEPredicted = None
bestCoef = None
bestIntercept = None
worstRMSE = 0
worstRMSEData = None
worstRMSEPredicted = None
worstCoef = None
worstIntercept = None

# Declare the regression to be a dictionary

print("-----Running CV tests. Please wait-----")
for z in range(0, numberOfTests):

    if z/numberOfTests == 0.0:
        print("[                ]  0% completion")
    elif z/numberOfTests == 0.25:
        print("[////            ] 25% completion")
    elif z/numberOfTests == 0.50:
        print("[////////        ] 50% completion")
    elif z/numberOfTests == 0.75:
        print("[////////////    ] 75% completion")
    elif z/(numberOfTests-1) == 1.00:
        print("[////////////////] 100% completion")

    # Randomly break the data up into training and testing. Will use 80% for training, 20% for testing.
    TestIndices = np.array([])
    while TestIndices.__len__() < int(0.8*dataSize):
        # Randomly select an index value and store it. If it has already been chosen, pick again.
        index = int(random.random() * dataSize)
        if not TestIndices.__contains__(index):
            TestIndices = np.append(TestIndices, [index])
    # For aesthetic purposes:
    TestIndices = np.sort(TestIndices)

    # For the desired training indices, add the values to the training arrays
    predictorTrainValues = np.array([])
    yTrainValues = np.array([])

    for q in range(0, TestIndices.__len__()):
        predictorTrainValues = np.append(predictorTrainValues, [iValues.take(q), jValues.take(q)])
        yTrainValues = np.append(yTrainValues, yValues.take(q))

    # Reshape the numpy arrays into a usable format
    predictorTrainValues = predictorTrainValues.reshape((int(0.8 * dataSize), 2))
    yTrainValues = yTrainValues.reshape((int(0.8 * dataSize), 1))

    # Run a linear regression on the training data
    regr = linear_model.LinearRegression()
    regr.fit(predictorTrainValues, yTrainValues)

    # Now, create the testing arrays:
    predictorTestValues = np.array([])
    yTestValues = np.array([])

    for p in range(0, dataSize):
        if not TestIndices.__contains__(p):
            predictorTestValues = np.append(predictorTestValues, [iValues.take(p), jValues.take(p)])
            yTestValues = np.append(yTestValues, [yValues.take(p)])
    predictorTestValues = predictorTestValues.reshape((int(0.2*dataSize), 2))
    yTestValues = yTestValues.reshape((int(0.2*dataSize), 1))

    # Predict the values
    predictedYValues = regr.predict(predictorTestValues)

    # Find the RMSE value and add it to the RMSE array:
    RMSE = sqrt(mean_squared_error(yTestValues, predictedYValues))
    RMSEValues = np.append(RMSEValues, [RMSE])

    # Test whether or not the RMSE is the best or the worst
    if RMSE < bestRMSE:
        bestRMSE = RMSE
        bestRMSEData = yTestValues
        bestRMSEPredicted = predictedYValues
        bestCoef = regr.coef_
        bestIntercept = regr.intercept_
    elif RMSE > worstRMSE:
        worstRMSE = RMSE
        worstRMSEData = yTestValues
        worstRMSEPredicted = predictedYValues
        worstCoef = regr.coef_
        worstIntercept = regr.intercept_

# Report mean and standard deviation of RMSE values
print("The mean of the RMSE values is", RMSEValues.mean(), "and the standard deviation is", RMSEValues.std(), ".")
print("The best RMSE value was", bestRMSE, "which had a corresponding regression equation of y =", bestCoef.take(0),
      "i +", bestCoef.take(1), "j +", bestIntercept.take(0), ".")
print("The worst RMSE value was", worstRMSE, "which had a corresponding regression equation of y =", worstCoef.take(0),
      "i +", worstCoef.take(1), "j +", worstIntercept.take(0), ".")

# Plot a scatter plot of data for best RMSE value:
xOneBest = bestRMSEData.min()
xTwoBest = bestRMSEData.max()
scatterBest = plt.figure(1)
plt.scatter(bestRMSEData, bestRMSEPredicted, marker="o", color="b")
plt.plot([xOneBest, xTwoBest], [xOneBest, xTwoBest], color="k")
scatterBest.suptitle("Actual Values vs. Predicted Values for best RMSE")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

# Plot a scatter plot of data for worst RMSE value
xOneWorst = worstRMSEData.min()
xTwoWorst = worstRMSEData.max()
scatterWorst = plt.figure(2)
plt.scatter(worstRMSEData, worstRMSEPredicted, marker="o", color="b")
plt.plot([xOneWorst, xTwoWorst], [xOneWorst, xTwoWorst], color="k")
scatterWorst.suptitle("Actual Values vs. Predicted Values for worst RMSE")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

# Plot a histogram of the RMSE values
RMSEbins = np.array([])
numberRMSEBins = 25
RMSEbinrange = (worstRMSE - bestRMSE)/25
binMarker = bestRMSE
RMSEindex = 0
for i in range(0, numberRMSEBins):
    RMSEbins = np.append(RMSEbins, [binMarker])
    binMarker += RMSEbinrange
RMSEhistogram = plt.figure(3)
plt.hist(RMSEValues, RMSEbins, histtype='bar')
RMSEhistogram.suptitle('Histogram of RMSE Values')

print("")
print("One thing to figure out is why the best and worst RMSE equations are the same. I can't figure it out for the",
      " life of me.")

plt.show()
