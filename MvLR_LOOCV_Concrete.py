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
    allX = pd.read_csv('completeData.csv', keep_default_na=False)
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
    firstHalf = allX.ix[:, :21]
    cte = allX.ix[:, 25]
    oneHotEncodedframe = pd.concat([encodedMixP, encodedSCM, encodedFineA, encodedCoarseA], axis=1)
    secondHalf = xValues.ix[:, 6:]
    completearray = pd.concat([firstHalf, cte, oneHotEncodedframe, secondHalf], axis=1)
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


def createstring(yVariable, currentYVariable, RMSE, r2):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "-------------------------\n" \
                   "Results for " + str(currentYVariable) + ":" \
                   "\n-------------------------\n"\
                   + "- The mean of the y-variable was " + "{0:.4e}".format(yVariable.mean()) + " and the standard" \
                            " deviation was " + "{0:.4e}".format(yVariable.std()) + ".\n\n" \
                   + "- Using LOOCV, the RMSE of the y-variable was" + "{0:.4e}".format(RMSE) + " and the R^2 value" \
                     " was " + "{0:.4e}".format(r2) + ".\n\n" \

    return outputstring


def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch = initializeData()

    # Create a string that will be added to each of the plots. This will include information such as the specific
    #  type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Multivariate Linear Regression run on " + dateandtime + " LOOCV"

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt",'w')
    numberOfTestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + additionalinfo + "\n" \
                          + "Total number of y - variables: " + str(numyvariables-1) + "\n" \
                          "---------------------------------------------------------------------------\n\n\n"
    output_file.write(numberOfTestsstring)

    # For each of the Y variables, want to run a linear regression:
    for i in range(1, numyvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        dataSize = np.size(yvariables[i])
        yVariable = np.transpose(yvariables[i].reshape(1, dataSize))
        currentYVariable = yvariablenames[i]

        # Create a figure that will store the subplots (for this particular y variable)
        figure = plt.figure(i)
        plotTitleString = "Results for " + str(currentYVariable)
        figure.suptitle(plotTitleString)
        figure.text(0, 0, additionalinfo)

        # Create a histogram of the current y variable
        histogramOfY = figure.add_subplot(1, 2, 1)
        histogramOfY.locator_params(axis='x',nbins=5)
        histogramOfY.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(yVariable,histtype="bar")
        histogramOfY.set_title('Histogram of Y-Values')
        histogramOfY.set_xlabel('Y-Values')
        histogramOfY.set_ylabel('Number in Range')

        # Now, want to normalize the y variable
        y_mean = yVariable.mean()
        y_std = yVariable.std()
        y_normalized = (yVariable - y_mean) / y_std

        # Want to run a cross validation test leaving one of the data points out at a time:
        # Initialize an array to store the individual predicted y variables in
        predictedyarray = np.array([])
        for z in range(0, np.size(yVariable)):

            # Define the training set to be all but the z index of the y variable (remove it, as well as the
            # corresponding x data.
            yTrainValues = np.delete(y_normalized, (z), axis=0)
            xTrainValues = np.delete(xvariables, (z), axis=0)

            # Run a linear regression on the current y variable and the x variables:
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(xTrainValues, yTrainValues)

            # Now, want to run cross validation test

            # Start by creating the testing array - only use the zth element (the one that was previously left out)
            xTestValues = xvariables[z, :]
            xTestValues = np.reshape(xTestValues, ([1, np.size(xTestValues)]))

            # Predict the values
            predictedYValues_normalized = regr.predict(xTestValues)

            # De-normalize the data
            predictedYValue = predictedYValues_normalized * y_std + y_mean

            # add the predicted y value to an array. These will later be plotted against the actual y values.
            predictedyarray = np.append(predictedyarray, [predictedYValue])

        # Add a plot of the entire LOOCV fit to the figure.
        loocvplot = figure.add_subplot(1, 2, 2)
        loocvplot.locator_params(axis='x',nbins=5)
        loocvplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        loocvplot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.scatter(yVariable, predictedyarray, marker="o", color="b")
        loocvplot.set_title('LOOCV Plot of Concrete Data')
        loocvplot.set_xlabel('Actual Values')
        loocvplot.set_ylabel('Predicted Values')
        ymin = np.amin(predictedyarray) - np.std(predictedyarray)
        ymax = np.amax(predictedyarray) + np.std(predictedyarray)
        xmin = np.amin(yVariable) - np.std(yVariable)
        xmax = np.amax(yVariable) + np.std(yVariable)
        loocvplot.set_ylim([ymin,ymax])
        loocvplot.set_xlim([xmin,xmax])

        # Find the RMSE of the LOOCV fit
        RMSE = sqrt(mean_squared_error(yVariable, predictedyarray))
        # Find the R^2 value of the LOOCV fit
        r2 = r2_score(yVariable, predictedyarray)

        plt.tight_layout()

        # save the figure
        titlestring = yvariablenames[i] + "_LOOCVplot.png"
        # make the figure more readable
        figure.set_size_inches(14.2, 8)
        figure.savefig(titlestring)

        # Store the information from the run in an output .txt file
        outputstring = createstring(yVariable, currentYVariable, RMSE, r2)
        output_file.write(outputstring)

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
