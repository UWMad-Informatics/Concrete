import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sklearn as sklearn
import time
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def initializeData():

    """ Initializes the data taken from the completeData.csv and the formattedXValues.csv. Note that these must be
    the names of the arrays in your folder."""

    # Read in the CSV
    allX = pd.read_csv('completeRawData.csv', keep_default_na=False)
    xValues = pd.read_csv('formattedXValues.csv')
    filename = "completeData.csv and formattedXValues.csv"

    # Separate the CSV columns into array variables and numpy vars to store new categorical variables
    mixNum = allX['Mix Number']
    mixP = allX['Mix Proportion']
    mixPFinal = np.empty(len(mixP))
    scm = allX['SCM']
    scmFinal = np.empty(len(scm))
    fineA = allX['Fine Aggregate']
    fineAFinal = np.empty(len(fineA))
    coarseA = allX['Coarse Aggregate']
    coarseAFinal = np.empty(len(coarseA))

    # Loop through every mix in the csv file
    # Not sure how to do 3 different variables
    for y in range(0, len(mixNum)):
        # Sort Mix Proportions
        if mixP[y] == "A-F":
            mixPFinal[y] = 2
        elif mixP[y] == "A-S":
            mixPFinal[y] = 1
        elif mixP[y] == "A":
            mixPFinal[y] = 0
        else:
            print('Unidentified Variable in mixP: ')
            print(mixP[y])

        # Sort SCM into slag or fly ash
        if scm[y] == 'N/A':
            scmFinal[y] = 1000
        elif scm[y] == 'Slag 1':
            scmFinal[y] = 0
        elif scm[y] == 'Slag 2':
            scmFinal[y] = 0
        elif scm[y] == 'Fly Ash 1':
            scmFinal[y] = 1
        elif scm[y] == 'Fly Ash 2':
            scmFinal[y] = 1
        elif scm[y] == 'Fly Ash 3':
            scmFinal[y] = 1
        else:
            print('Unidentified Variable in scm: ')
            print(scm[y])

        # Sort the fine aggregate
        if fineA[y] == 'Sand A':
            fineAFinal[y] = 0
        elif fineA[y] == 'Sand B':
            fineAFinal[y] = 1
        else:
            print('Unidentified Variable in fineA: ')
            print(fineA[y])

        # Sort the coarse aggregate
        if coarseA[y] == 'GG1':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'GG2':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'GG3':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'GG4':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'GG5':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'GG6':
            coarseAFinal[y] = 0
        elif coarseA[y] == 'CS1':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS2':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS3':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS4':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS5':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS6':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS7':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS8':
            coarseAFinal[y] = 1
        elif coarseA[y] == 'CS9':
            coarseAFinal[y] = 1
        else:
            print('Unidentified Variable in coarseA: ')
            print(coarseA[y])

    # One Hot Encode the sorted variables
    encodedMixP = pd.get_dummies(mixPFinal)
    encodedSCM = pd.get_dummies(scmFinal)
    encodedFineA = pd.get_dummies(fineAFinal)
    encodedCoarseA = pd.get_dummies(coarseAFinal)

    # Update the headers for onehotencoded variables
    # Get the current variable names
    encodedSCMlist = list(encodedSCM.columns.values)
    encodedFineAlist = list(encodedFineA.columns.values)
    encodedCoarseAlist = list(encodedCoarseA.columns.values)
    encodedMixPlist = list(encodedMixP.columns.values)
    # go through and replace the current names with the updated ones
    encodedSCM.rename(columns={encodedSCMlist[0]: 'SCM_0', encodedSCMlist[1]: 'SCM_1', encodedSCMlist[2]: 'SCM_1000'},
        inplace=True)
    encodedFineA.rename(columns={encodedFineAlist[0]: 'FineA_0', encodedFineAlist[1]: 'FineA_1'}, inplace=True)
    encodedCoarseA.rename(columns={encodedCoarseAlist[0]: 'CoarseA_0', encodedCoarseAlist[1]: 'CoarseA_1'},
                          inplace=True)
    encodedMixP.rename(columns={encodedMixPlist[0]: 'MixP_0', encodedMixPlist[1]: 'MixP_1', encodedMixPlist[2]: 'MixP_2'},
        inplace=True)

    # Remake the dataframe to include the onehotencoded columns instead of the regular columns.
    firstHalf = allX.ix[:, :22]
    oneHotEncodedframe = pd.concat([encodedMixP, encodedSCM, encodedFineA, encodedCoarseA], axis=1)
    secondHalf = xValues.ix[:, 6:]
    completearray = pd.concat([firstHalf, oneHotEncodedframe, secondHalf], axis=1)
    variablenames = list(completearray.columns.values)
    # convert to numpy array
    completenumpyarray = completearray.as_matrix()

    # remove the first 15 rows in the array to clear the NaN entries
    completenumpyarray = completenumpyarray[15:, :]
    # Also, remove the columns that include mix A as well as SCM_1000

    #####
    # Now, Ask whether or not to run decision trees on batch A data or batch B
    batch = input("which batch to run tests on (A or B)? ")

    if batch == "A":

        # break up the data into the batch A values
        batchAYcolumns = [0, 5, 6, 7, 8, 21]
        yvariables = np.transpose(completenumpyarray[:, batchAYcolumns])
        numyvariables = 6
        yvariablenames = [variablenames[x] for x in batchAYcolumns]
        batchAXcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 32, 35, 38, 41]
        # normalize the x variables. Will normalize y variables in the main body
        # after a histogram of the data is created.
        xvariables = completenumpyarray[:, batchAXcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        xVariablesShape = xvariables.shape
        # index through each of the columns and find the l2 norm
        for p in range(0, xVariablesShape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            # index through each value of the column (thus, go through each row) and divide by the l2 norm
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variablenames[x] for x in batchAXcolumns]

    elif batch == "B":

        # break up the data into the batch B values
        batchBYcolumns = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        yvariables = np.transpose(completenumpyarray[:, batchBYcolumns])
        numyvariables = 17
        yvariablenames = [variablenames[x] for x in batchBYcolumns]
        batchBXcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 33, 36, 39, 42]
        # normalize the x variables. Will normalize y variables in the main body
        # after a histogram of the data is created.
        xvariables = completenumpyarray[:, batchBXcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        xVariablesShape = xvariables.shape
        # index through each of the columns and find the l2 norm
        for p in range(0, xVariablesShape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            # index through each value of the column (thus, go through each row) and divide by the l2 norm
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variablenames[x] for x in batchBXcolumns]

    else:
        print("Invalid Input.")
        exit(0)

    return completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch


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
                 currentYVariable, xVariableNames, r2values, bestr2, worstr2):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "-------------------------\n" \
                   "Results for " + str(currentYVariable) + ":" \
                   "\n-------------------------\n"\
                   + "- The mean of the y-variable was " + "{0:.4e}".format(yVariable.mean()) + " and the standard" \
                            " deviation was " + "{0:.4e}".format(yVariable.std()) + ".\n\n" \
                   + "- The mean of the RMSE values was " + "{0:.4e}".format(RMSEValues.mean()) + " and the standard" \
                            " deviation was " + "{0:.4e}".format(RMSEValues.std()) + ".\n\n" \
                   + "- The mean of the R^2 values was " + "{0:.4}".format(r2values.mean()) + " and the standard " \
                            " deviation was " + "{0:.4}".format(r2values.std()) + ".\n\n" \
                   + "- The best R^2 value was " + "{0:.4}".format(r2values.max()) + " and the worst was "\
                   + "{0:.4}".format(r2values.min()) + ". \n\n"

    best_regression_string = "- The best RMSE was " + "{0:.4e}".format(bestRMSE) + " with an R^2 value of "\
                             + "{0:.4}".format(bestr2) + " and a regression equation of:\n "

    for m in range(0,bestCoef[0].__len__()):
        best_regression_string = best_regression_string + "{0:.4e}".format(bestCoef[0][m]) + "*" + "(" + xVariableNames[m] + ")" + " + "
    best_regression_string = best_regression_string + "{0:.4e}".format(bestIntercept) + ".\n\n"

    worst_regression_string = "- The worst RMSE was " + "{0:.4e}".format(worstRMSE) + "with an R^2 value of "\
                              + "{0:.4}".format(worstr2) + " and a regression equation of:\n   "

    for n in range(0,worstCoef[0].__len__()):
        worst_regression_string = worst_regression_string + "{0:.4e}".format(worstCoef[0][n]) + "*" + "(" + xVariableNames[n] + ")"\
                                  + " + "
    worst_regression_string = worst_regression_string + "{0:.4e}".format(worstIntercept) + ".\n\n\n"

    outputstring = outputstring + best_regression_string + worst_regression_string

    return outputstring


def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch = initializeData()

    # Prompt for number of CV tests to run on each y variable:
    numberOfTests = int(input("How many CV tests should be done for each y variable? "))

    # Prompt for the percentage of data to use for training. The remaining amount will be used for testing.
    percentTest = int(input("What percentage of data do you want to use for training? ")) / 100

    # Prompt whether or not creating plots is desired
    makeplotsquestion = input("Would you like to create and save plots for this run? (True / False)")
    if makeplotsquestion == "True":
        makeplots = True
    elif makeplotsquestion == "False":
        makeplots = False

    # Create a string that will be added to each of the plots. This will include information such as the specific
    #  type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Multivariate Linear Regression run on " + dateandtime

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt",'w')
    numberOfTestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + additionalinfo + "\n" \
                          + "Total number of y - variables: " + str(numyvariables-1) + "\n" \
                          + "For this run, the total number of CV tests done on each y-variable was " + str(numberOfTests)\
                          + ".\nFurthermore, " + str(100 - percentTest*100) + "% of the data was used for testing.\n" \
                          "---------------------------------------------------------------------------\n\n\n"
    output_file.write(numberOfTestsstring)

    # Create two arrays. These will be arrays of arrays. One will store all of the RMSE data (normalized by std) and the
    # other will contain all of the r^2 data. These will then be used to create box and whisker plots of all the data
    # side by side for comparison of fit.
    allrmsedata = list()
    allrsquareddata = list()

    # For each of the Y variables, want to run a linear regression:
    for i in range(1, numyvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        dataSize = np.size(yvariables[i])
        yVariable = yvariables[i].reshape(1, dataSize)
        currentYVariable = yvariablenames[i]

        # Create a histogram of the current y variable
        # histogramOfY = figure.add_subplot(2, 2, 1)
        # histogramOfY.locator_params(axis='x',nbins=5)
        # histogramOfY.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        # plt.hist(np.transpose(yVariable),histtype="bar")
        # histogramOfY.set_title('Histogram of Y-Values')
        # histogramOfY.set_xlabel('Y-Values')
        # histogramOfY.set_ylabel('Number in Range')

        # Want to normalize the y variable
        y_mean = yVariable.mean()
        y_std = yVariable.std()
        y_normalized = (yVariable - y_mean) / y_std

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
        # Initialize values for R^2 values that correspond to the best and worst RMSE
        bestr2 = None
        worstr2 = None

        # Initialize an array to store the R^2 values in
        r2values = np.array([])

        # Perform a specified number of CV tests on the data:
        for z in range(0, numberOfTests):

            # Randomly break the data up into training and testing. Will use input percentage for training,
            # 20% for testing.
            TrainIndices = generaterandomindices(dataSize, percentTest)
            xTrainValues, yTrainValues = createtrainingarrays(dataSize, xvariables, y_normalized, TrainIndices)

            # Run a linear regression on the current y variable and the x variables:
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(xTrainValues, yTrainValues)

            # Now, want to run cross validation test

            # Start by creating the testing arrays:
            xTestValues, yTestValues_normalized = createtestarrays(dataSize, xvariables, y_normalized, TrainIndices)

            # Predict the values
            predictedYValues_normalized = regr.predict(xTestValues)

            # De-normalize the data
            yTestValues = yTestValues_normalized * y_std + y_mean
            predictedYValues = predictedYValues_normalized * y_std + y_mean

            # Calculate the RMSE value and add it to the current array.
            RMSE = sqrt(mean_squared_error(yTestValues, predictedYValues))
            RMSEValues = np.append(RMSEValues, [RMSE])

            # Calculate the R^2 value and add it to the array
            r2 = r2_score(yTestValues, predictedYValues)
            r2values = np.append(r2values, [r2])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
            if RMSE < bestRMSE:
                bestRMSE = RMSE
                bestRMSEData = yTestValues
                bestRMSEPredicted = predictedYValues
                bestCoef = regr.coef_
                bestIntercept = regr.intercept_
                bestr2 = r2
            elif RMSE > worstRMSE:
                worstRMSE = RMSE
                worstRMSEData = yTestValues
                worstRMSEPredicted = predictedYValues
                worstCoef = regr.coef_
                worstIntercept = regr.intercept_
                worstr2 = r2

        # Depending on whether or not makeplots is true or false
        if makeplots is True:
            # Create a figure that will store the subplots (for this particular y variable)
            figure = plt.figure(i)
            plotTitleString = "Results for " + str(currentYVariable)
            figure.suptitle(plotTitleString)
            figure.text(0, 0, additionalinfo)

            # add the plots for best, worst fits as well as a plot of the RMSE and r^2 values from the tests.
            rsquaredhist = figure.add_subplot(2, 2, 1)
            rsquaredhist.locator_params(axis='x', nbins=10)
            rsquaredhist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            plt.hist(r2values, histtype="bar")
            rsquaredhist.set_title('Histogram of r^2 Values')
            rsquaredhist.set_xlabel('r^2 Value')
            rsquaredhist.set_ylabel('Number in Range')

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

            # save the figure
            titlestring = yvariablenames[i] + "_" + batch + str(100 - percentTest*100) + ".png"
            # make the figure more readable
            figure.set_size_inches(14.2, 8)
            figure.savefig(titlestring)

        # Add the normalized RMSE and r^2 values to the complete arrays
        normalizedrmsedata = RMSEValues / yVariable.std()
        normalizedrmsedata = normalizedrmsedata.reshape(1, np.size(normalizedrmsedata))
        allrmsedata.append(normalizedrmsedata)
        r2values = r2values.reshape(1, np.size(r2values))
        allrsquareddata.append(r2values)

        # Store the information from the run in an output .txt file
        outputstring = createstring(yVariable, RMSEValues, bestRMSE, bestCoef, bestIntercept, worstRMSE, worstCoef,
                                    worstIntercept, currentYVariable, xvariablenames, r2values, bestr2, worstr2)
        output_file.write(outputstring)

    # If make plots is true, create the side-by-side plots
    if makeplots is True:

        # RMSE plot
        rmseboxplot = plt.figure()
        rmsebp = rmseboxplot.add_subplot(1, 1, 1)
        rmsebp.boxplot(allrmsedata)
        rmsebp.set_title('Normalized RMSE Data for Each Y Variable')
        x = range(1,len(yvariablenames))
        plt.xticks(x, yvariablenames[1:len(yvariablenames)], rotation='vertical')
        rmsebp.set_xlabel('Y Variables')
        rmsebp.set_ylabel('RMSE / std')
        plt.tight_layout()

        # R^2 plot
        rsquaredboxplot = plt.figure()
        rsquaredbp = rsquaredboxplot.add_subplot(1, 1, 1)
        rsquaredbp.boxplot(allrsquareddata)
        rsquaredbp.set_title('R^2 Values for Each Y Variable')
        x = range(1,len(yvariablenames))
        plt.xticks(x, yvariablenames[1:len(yvariablenames)], rotation='vertical')
        rsquaredbp.set_xlabel('Y Variables')
        rsquaredbp.set_ylabel('R^2 Value')
        plt.tight_layout()

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
