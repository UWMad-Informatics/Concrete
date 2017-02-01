import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def readcsv():

    """ Reads the data from a specified .csv file and returns it as a numpy array. Also returns a pandas dataframe to
    retrieve the column headers from."""

    filename = input("What is the name of the .csv file to read from? ")

    # import the file:
    completearray = pd.read_csv(filename, delimiter=',')
    completenumpyarray = np.transpose(completearray.as_matrix())

    return completearray, completenumpyarray


def generaterandomindices(dataSize, percentTest):

    """ For the given datasize from the .csv file generates a set of random indices to use for the training set. Uses a
    user defined percentage of the data for training and testing. """

    TrainIndices = np.array([])
    while TrainIndices.__len__() < int(percentTest * dataSize):
        # Randomly select an index value and store it. If it has already been chosen, pick again.
        index = int(random.random() * dataSize)
        if not TrainIndices.__contains__(index):
            TrainIndices = np.append(TrainIndices, [index])

    # For aesthetic purposes:
    TrainIndices = np.sort(TrainIndices)

    return TrainIndices


def createtrainingarrays(dataSize, xVariables, yVariable, TrainIndices):

    """ Creates a random training array of x and y values from the previously generated random training indices."""

    # For the desired training indices, add the values to the training arrays
    xTrainValues = np.array([])
    yTrainValues = np.array([])
    indexCounter = 0
    for q in range(0, dataSize):
        if TrainIndices.__contains__(q):

            if indexCounter is 0:
                xTrainValues = xVariables[q]
                indexCounter = -1
            else:
                xTrainValues = np.vstack((xTrainValues, xVariables[q]))

            yTrainValues = np.append(yTrainValues, yVariable[0][q])

    # Reshape the data to proper dimensions so that a linear regression may be performed
    length = yTrainValues.size
    yTrainValues = yTrainValues.reshape(length, 1)

    return xTrainValues, yTrainValues


def createtestarrays(dataSize, xVariables, yVariable, TrainIndices):

    """ Uses the remaining values from the dataset (those not included in the training data) to form a set of testing
    arrays. """

    xTestValues = np.array([])
    yTestValues = np.array([])
    indexCounter1 = 0
    for p in range(0, dataSize):
        if not TrainIndices.__contains__(p):
            if indexCounter1 is 0:
                xTestValues = xVariables[p]
                indexCounter1 = -1
            else:
                xTestValues = np.vstack((xTestValues, xVariables[p]))
            yTestValues = np.append(yTestValues, yVariable[0][p])

    # Reshape array to proper size
    length = yTestValues.size
    yTestValues = yTestValues.reshape(length, 1)

    return xTestValues, yTestValues


def createstring(yVariable, RMSEValues, bestRMSE, bestCoef, bestIntercept, worstRMSE, worstCoef, worstIntercept,
                 currentYVariable, xVariableNames):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "Results for " + str(currentYVariable) + ":\n\n" + "The mean of the y-variable was "\
                   + str(yVariable.mean()) + " and the standard deviation was " + str(yVariable.std()) + ".\n"\
                   + "The mean of the RMSE values" + " is " + str(RMSEValues.mean())\
                   + " and the standard deviation is " + str(RMSEValues.std()) + ".\n"

    best_regression_string = "The worst RMSE was " + str(bestRMSE) + " with a regression equation of: "
    for m in range(0,bestCoef[0].__len__()):
        best_regression_string = best_regression_string + str(bestCoef[0][m]) + "*" + xVariableNames[m] + " + "
    best_regression_string = best_regression_string + str(bestIntercept[0]) + ".\n"

    worst_regression_string = "The worst RMSE was " + str(worstRMSE) + " with a regression equation of: "
    for n in range(0,worstCoef[0].__len__()):
        worst_regression_string = worst_regression_string + str(worstCoef[0][n]) + "*" + xVariableNames[n]\
                                  + " + "
    worst_regression_string = worst_regression_string + str(worstIntercept[0]) + ".\n\n\n"

    outputstring = outputstring + best_regression_string + worst_regression_string

    return outputstring


def main():

    completearray, completeNumpyArray = readcsv()

    # For now, do not assume that the data we are using starts with 31 y-variable columns
    numYvariables = int(input("How many y variables are there? NOTE: Add 1 to the total to account for the column labels. "))

    # get the names of he variables
    variableNames = list(completearray.columns.values)
    yVariableNames = variableNames[:numYvariables]
    xVariableNames = variableNames[numYvariables:]

    # For simplicity, move the x data into a separate array. All of this will be used no matter which y-variable is
    # being tested.
    xVariables = np.transpose(completeNumpyArray[numYvariables:])

    # Prompt for number of CV tests to run on each y variable:
    numberOfTests = int(input("How many CV tests should be done for each y variable? "))

    # Prompt for the percentage of data to use for training. The remaining amount will be used for testing.
    percentTest = int(input("What percentage of data do you want to use for testing? ")) / 100

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt",'w')
    numberOfTestsstring = "For this run, the total number of CV tests done on each y-variable was "\
        + str(numberOfTests) + ".\n" + str(percentTest*100) + "% of the data was used for testing.\n\n"
    output_file.write(numberOfTestsstring)

    # For each of the Y variables, want to run a linear regression:
    for i in range(1, numYvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        dataSize = np.size(completeNumpyArray[i])
        yVariable = completeNumpyArray[i].reshape(1, dataSize)
        currentYVariable = yVariableNames[i]

        # Create a figure that will store the subplots (for this particular y variable)
        figure = plt.figure(i)
        plotTitleString = "Results for " + str(currentYVariable)
        figure.suptitle(plotTitleString)

        # Create a histogram of the current y variable
        histogramOfY = figure.add_subplot(2, 2, 1)
        plt.hist(np.transpose(yVariable),histtype="bar")
        histogramOfY.title.set_text('Histogram of Y-Values')

        # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
        RMSEValues = np.array([])

        # Initialize values for the best and worst RMSE
        bestRMSE = 1000000
        bestRMSEData = None
        bestRMSEPredicted = None
        bestCoef = None
        bestIntercept = None
        worstRMSE = 0
        worstRMSEData = None
        worstRMSEPredicted = None
        worstCoef = None
        worstIntercept = None

        # Perform a specified number of CV tests on the data:
        for z in range(0, numberOfTests):

            # Randomly break the data up into training and testing. Will use input percentage for training,
            #  20% for testing.
            TrainIndices = generaterandomindices(dataSize, percentTest)
            xTrainValues, yTrainValues = createtrainingarrays(dataSize, xVariables, yVariable, TrainIndices)

            # Run a linear regression on the current y variable and the x variables:
            regr = linear_model.LinearRegression()
            regr.fit(xTrainValues, yTrainValues)

            # Now, want to run cross validation test

            # Start by creating the testing arrays:
            xTestValues, yTestValues = createtestarrays(dataSize, xVariables, yVariable, TrainIndices)

            # Predict the values
            predictedYValues = regr.predict(xTestValues)

            # Calculate the RMSE value and add it to the current array.
            RMSE = sqrt(mean_squared_error(yTestValues, predictedYValues))
            RMSEValues = np.append(RMSEValues, [RMSE])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
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

        # add the plots for best, worst fits as well as a plot of the RMSE values from the tests.
        bestRMSEplot = figure.add_subplot(2,2,2)
        plt.scatter(bestRMSEData, bestRMSEPredicted, marker="o", color="b")
        bestRMSEplot.title.set_text('Actual vs. Predicted Values for Best RMSE')

        worstRMSEplot = figure.add_subplot(2,2,3)
        plt.scatter(worstRMSEData, worstRMSEPredicted, marker="o", color="b")
        worstRMSEplot.title.set_text('Actual vs. Predicted Values for Worst RMSE')

        RMSEhist = figure.add_subplot(2,2,4)
        plt.hist(RMSEValues,histtype="bar")
        RMSEhist.title.set_text('Histogram of RMSE Values')

        # Store the information from the run in an output .txt file
        outputstring = createstring(yVariable, RMSEValues, bestRMSE, bestCoef, bestIntercept, worstRMSE, worstCoef,
                                    worstIntercept, currentYVariable, xVariableNames)
        output_file.write(outputstring)

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
