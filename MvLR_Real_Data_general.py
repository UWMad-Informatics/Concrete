import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

    return completearray, completenumpyarray, filename


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

    outputstring = "-------------------------\n" \
                   "Results for " + str(currentYVariable) + ":" \
                   "\n-------------------------\n"\
                   + "- The mean of the y-variable was " + "{0:.4e}".format(yVariable.mean()) + " and the standard" \
                            " deviation was " + "{0:.4e}".format(yVariable.std()) + ".\n\n"\
                   + "- The mean of the RMSE values was " + "{0:.4e}".format(RMSEValues.mean()) + " and the standard" \
                            " deviation was " + "{0:.4e}".format(RMSEValues.std()) + ".\n\n"

    best_regression_string = "- The best RMSE was " + "{0:.4e}".format(bestRMSE) + " with a regression equation of:\n    "

    for m in range(0,bestCoef[0].__len__()):
        best_regression_string = best_regression_string + "{0:.4e}".format(bestCoef[0][m]) + "*" + "(" + xVariableNames[m] + ")" + " + "
    best_regression_string = best_regression_string + "{0:.4e}".format(bestIntercept[0]) + ".\n\n"

    worst_regression_string = "- The worst RMSE was " + "{0:.4e}".format(worstRMSE) + " with a regression equation of:\n   "

    for n in range(0,worstCoef[0].__len__()):
        worst_regression_string = worst_regression_string + "{0:.4e}".format(worstCoef[0][n]) + "*" + "(" + xVariableNames[n] + ")"\
                                  + " + "
    worst_regression_string = worst_regression_string + "{0:.4e}".format(worstIntercept[0]) + ".\n\n\n"

    outputstring = outputstring + best_regression_string + worst_regression_string

    return outputstring


def main():

    completearray, completeNumpyArray, filename = readcsv()

    # For now, do not assume that the data we are using starts with 31 y-variable columns
    numYvariables = int(input("How many y variables are there? NOTE: Add 1 to the total to account for the column labels. "))

    # get the names of the variables
    variableNames = list(completearray.columns.values)
    yVariableNames = variableNames[:numYvariables]
    xVariableNames = variableNames[numYvariables:]

    # For simplicity, move the x data into a separate array. All of this will be used no matter which y-variable is
    # being tested.
    xVariables = np.transpose(completeNumpyArray[numYvariables:])

    # Prompt for number of CV tests to run on each y variable:
    numberOfTests = int(input("How many CV tests should be done for each y variable? "))

    # Prompt for the percentage of data to use for training. The remaining amount will be used for testing.
    percentTest = int(input("What percentage of data do you want to use for training? ")) / 100

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt",'w')
    numberOfTestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + "Total number of y - variables: " + str(numYvariables-1) + "\n" \
                          + "For this run, the total number of CV tests done on each y-variable was " + str(numberOfTests)\
                          + ".\nFurthermore, " + str(percentTest*100) + "% of the data was used for training.\n" \
                          "---------------------------------------------------------------------------\n\n\n"
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
        histogramOfY.locator_params(axis='x',nbins=5)
        histogramOfY.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(np.transpose(yVariable),histtype="bar")
        histogramOfY.set_title('Histogram of Y-Values')
        histogramOfY.set_xlabel('Y-Values')
        histogramOfY.set_ylabel('Number in Range')

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
            # 20% for testing.
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
        bestRMSEplot.locator_params(axis='x',nbins=5)
        bestRMSEplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        bestRMSEplot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.scatter(bestRMSEData, bestRMSEPredicted, marker="o", color="b")
        bestRMSEplot.set_title('Actual vs. Predicted Values for Best RMSE')
        bestRMSEplot.set_xlabel('Actual Values')
        bestRMSEplot.set_ylabel('Predicted Values')
        ymin = np.amin(bestRMSEPredicted) - np.std(bestRMSEPredicted)
        ymax = np.amax(bestRMSEPredicted) + np.std(bestRMSEPredicted)
        xmin = np.amin(bestRMSEData) - np.std(bestRMSEData)
        xmax = np.amax(bestRMSEData) + np.std(bestRMSEData)
        bestRMSEplot.set_ylim([ymin,ymax])
        bestRMSEplot.set_xlim([xmin,xmax])

        worstRMSEplot = figure.add_subplot(2,2,3)
        worstRMSEplot.locator_params(axis='x',nbins=5)
        worstRMSEplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        worstRMSEplot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.scatter(worstRMSEData, worstRMSEPredicted, marker="o", color="b")
        worstRMSEplot.set_title('Actual vs. Predicted Values for Worst RMSE')
        worstRMSEplot.set_xlabel('Actual Values')
        worstRMSEplot.set_ylabel('Predicted Values')
        ymin = np.amin(worstRMSEPredicted) - np.std(worstRMSEPredicted)
        ymax = np.amax(worstRMSEPredicted) + np.std(worstRMSEPredicted)
        xmin = np.amin(worstRMSEData) - np.std(worstRMSEData)
        xmax = np.amax(worstRMSEData) + np.std(worstRMSEData)
        worstRMSEplot.set_ylim([ymin, ymax])
        worstRMSEplot.set_xlim([xmin, xmax])

        RMSEhist = figure.add_subplot(2,2,4)
        RMSEhist.locator_params(axis='x',nbins=5)
        RMSEhist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(RMSEValues,histtype="bar")
        RMSEhist.set_title('Histogram of RMSE Values')
        RMSEhist.set_xlabel('RMSE Value')
        RMSEhist.set_ylabel('Number in Range')

        plt.tight_layout()

        # Store the information from the run in an output .txt file
        outputstring = createstring(yVariable, RMSEValues, bestRMSE, bestCoef, bestIntercept, worstRMSE, worstCoef,
                                    worstIntercept, currentYVariable, xVariableNames)
        output_file.write(outputstring)

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
