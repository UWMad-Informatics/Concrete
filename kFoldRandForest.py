import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# TODO initializeData()
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

    # Now, Ask whether or not to run decision trees on batch A data or batch B
    batch = input("which batch to run tests on (A or B)? ")

    if batch == "A":

        # break up the data into the batch A values
        batchAYcolumns = [0, 5, 6, 7, 8, 21]
        yvariables = np.transpose(completenumpyarray[:, batchAYcolumns])
        numyvariables = 6
        yvariablenames = [variablenames[x] for x in batchAYcolumns]
        batchAXcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 32, 35, 38, 41]
        xvariables = completenumpyarray[:, batchAXcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        xVariablesShape = xvariables.shape
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, xVariablesShape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
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
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, xVariablesShape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variablenames[x] for x in batchBXcolumns]

    else:
        print("Invalid Input.")
        exit(0)

    return completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch


# TODO: createstring
# CHANGE: added avg_rmse as a parameter.
def createstring(rmsevalues, bestrmse, worstrmse, avg_rmse, currentyvariable, yvariable, r2values, bestr2, worstr2):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "-------------------------------------------------------\n" \
                   "Results for " + str(currentyvariable) + " using 100 Trees per Forest" +\
                                                            "\n-------------------------------------------------------\n" \
                    "- The mean of the yvariable was " + "{0:.4e}".format(yvariable.mean()) + " and the standard" \
                    " deviation was " + "{0:.4e}".format(yvariable.std()) + "\n\n" \
                   + "- The mean of the RMSE values was " + "{0:.4e}".format(rmsevalues.mean()) + " and the standard" \
                   + " deviation was " + "{0:.4e}".format(rmsevalues.std()) + ".\n\n"\
                   + "- The mean of the R^2 values was " + "{0:.4}".format(r2values.mean()) + " and the standard " \
                        " deviation was " + "{0:.4}".format(r2values.std()) + ".\n\n" \
                   + "- The best R^2 value was " + "{0:.4}".format(r2values.max()) + " and the worst was " \
                   + "{0:.4}".format(r2values.min()) + ". \n\n"

    best_string = "- The best RMSE was " + "{0:.4e}".format(bestrmse) + " with an R^2 value of "\
                  + "{0:.4}".format(bestr2) + "\n\n"

    worst_string = "- The worst RMSE was " + "{0:.4e}".format(worstrmse) + "with an R^2 value of "\
                              + "{0:.4}".format(worstr2) + "\n\n\n"

    # CHANGE: added avg_string
    avg_string = "- The mean RMSE was " + str(avg_rmse)

    outputstring = outputstring + best_string + worst_string + avg_string

    return outputstring


# TODO: Main Method
def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch = initializeData()

    # Prompt whether or not creating plots is desired
    makeplotsquestion = input("Would you like to create and save plots for this run? (True / False)")
    if makeplotsquestion == "True":
        makeplots = True
    elif makeplotsquestion == "False":
        makeplots = False

    # Prompt for number of CV tests to run on each y variable:
    # CHANGE: Change from # CV tests to # folds used for CV
    num_folds = int(input("How many folds should be used for cross validation? "))

    # Create a string that will be added to each of the plots. This will include information such as the specific
    #  type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Run on " + dateandtime

    # Create a .txt file to store the output in:
    # CHANGE: Output file content changed for number of folds
    output_file = open("output_file.txt", 'w')
    numberoftestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + "Total number of y - variables: " + str(numyvariables - 1) + "\n" \
                          + "For this run, " + str(num_folds) + "CV was used. "\
                          + "---------------------------------------------------------------------------\n\n\n"
    output_file.write(numberoftestsstring)

    # For each of the y variables, run a random forest using all of the x variables
    for i in range(1, numyvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        datasize = np.size(yvariables[i])
        yvariable = yvariables[i]
        currentyvariable = yvariablenames[i]

        # Now, want to normalize the y variable
        y_mean = yvariable.mean()
        y_std = yvariable.std()
        y_normalized = (yvariable - y_mean) / y_std
        y_normalized = np.reshape(y_normalized, newshape=(len(y_normalized), 1))

        # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
        rmsevalues = np.array([])
        # Initialize values for the best and worst RMSE
        bestrmse = 1000000
        bestrmsedata = None
        bestrmsepredicted = None
        worstrmse = 0
        worstrmsedata = None
        worstrmsepredicted = None
        # Initialize values for R^2 values that correspond to the best and worst RMSE
        bestr2 = None
        worstr2 = None

        # Initialize an array to store the R^2 values in
        r2values = np.array([])

        # Break the data into folds to be used for k-fold CV.
        rows_add = int(len(xvariables)/num_folds)
        print("Rows add: " + str(rows_add))
        # Create train and test to hold data later
        print("Y Normalized Shape: " + str(y_normalized.shape))
        xtrainvalues = np.ndarray(shape=(len(xvariables) - rows_add, len(xvariables[0, :])))
        ytrainvalues = np.ndarray(shape=(rows_add*(num_folds - 1), 1))
        print("X Train shape, expected: " + str(xtrainvalues.shape))
        print("Y Train Shape, Expected: " + str(ytrainvalues.shape))

        xtestvalues = np.ndarray(shape=(rows_add, len(xvariables[:, 0])))
        ytestvalues_normalized = np.ndarray(shape=(rows_add, len(y_normalized)))

        # CHANGE: Create the model here so we use the same model on all sets for a given x
        rand_forest = RandomForestRegressor()

        # CHANGE: average RMSE calculation initialization
        # This will be used to sum the RMSE for all folds, which we will then average.
        avg_rmse_sum = 0

        # Perform a specified number of CV tests on the data:
        for z in range(0, num_folds - 1):
            # Use different folds for train and test each run.
            # Edge case: last fold. If not at last fold, grab all rows needed. Otherwise, grab i to end of the rows.
            if z != z - 1:
                xtestvalues = xvariables[z:z + rows_add, :]
                ytestvalues_normalized = y_normalized[z:z + rows_add]
            else:
                xtestvalues = xvariables[z:, :]
                ytestvalues_normalized = y_normalized[z:z + rows_add, :]

            # Edge case: first fold
            if z != 0:
                xtrainvalues = xvariables[:z*rows_add, :]
                xtrainvalues = np.concatenate((xtrainvalues, xvariables[(z*rows_add) + rows_add:, :]))
                ytrainvalues = y_normalized[:z*rows_add, :]
                ytrainvalues = np.concatenate((ytrainvalues, y_normalized[(z*rows_add) + rows_add:, :]))
            else:
                xtrainvalues = xvariables[rows_add:, :]
                ytrainvalues = y_normalized[rows_add:, :]

            # Fit the model
            rand_forest.fit(xtrainvalues, ytrainvalues)

            # Now, want to run cross validation test
            # Predict the values
            predictedyvalues_normalized = rand_forest.predict(xtestvalues)

            # De-normalize the data
            ytestvalues = (ytestvalues_normalized * y_std) + y_mean
            predictedyvalues = (predictedyvalues_normalized * y_std) + y_mean

            # Calculate the RMSE value and add it to the current array.
            rmse = sqrt(mean_squared_error(ytestvalues, predictedyvalues))
            rmsevalues = np.append(rmsevalues, [rmse])

            # Calculate the R^2 value and add it to the array
            r2 = r2_score(ytestvalues, predictedyvalues)
            r2values = np.append(r2values, [r2])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
            if rmse < bestrmse:
                bestrmse = rmse
                bestrmsedata = ytestvalues
                bestrmsepredicted = predictedyvalues
                bestr2 = r2
            elif rmse > worstrmse:
                worstrmse = rmse
                worstrmsedata = ytestvalues
                worstrmsepredicted = predictedyvalues
                worstr2 = r2

                # If we want tomake plots, create a figure that will store the subplots (for this particular y variable)
                if makeplots is True:
                    figure = plt.figure(i)
                    plottitle = "Results for " + str(currentyvariable) + " Using 10 Fold CV"
                    figure.suptitle(plottitle)
                    figure.text(0, 0, additionalinfo)

                    # add the plots for best, worst fits as well as a plot of the RMSE and r^2 values from the tests.
                    rsquaredhist = figure.add_subplot(2, 2, 1)
                    rsquaredhist.locator_params(axis='x', nbins=10)
                    rsquaredhist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                    plt.hist(r2values, histtype="bar")
                    rsquaredhist.set_title('Histogram of r^2 Values')
                    rsquaredhist.set_xlabel('r^2 Value')
                    rsquaredhist.set_ylabel('Number in Range')

                    # add the plots for best, worst fits as well as a plot of the RMSE values from the tests.
                    bestrmseplot = figure.add_subplot(2, 2, 2)
                    bestrmseplot.locator_params(axis='x', nbins=5)
                    bestrmseplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    bestrmseplot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    plt.scatter(bestrmsedata, bestrmsepredicted, marker="o", color="b")
                    bestrmseplot.set_title('Actual vs. Predicted Values for Best RMSE')
                    bestrmseplot.set_xlabel('Actual Values')
                    bestrmseplot.set_ylabel('Predicted Values')
                    ymin = np.amin(bestrmsepredicted) - np.std(bestrmsepredicted)
                    ymax = np.amax(bestrmsepredicted) + np.std(bestrmsepredicted)
                    xmin = np.amin(bestrmsedata) - np.std(bestrmsedata)
                    xmax = np.amax(bestrmsedata) + np.std(bestrmsedata)
                    bestrmseplot.set_ylim([ymin, ymax])
                    bestrmseplot.set_xlim([xmin, xmax])

                    worstrmseplot = figure.add_subplot(2, 2, 3)
                    worstrmseplot.locator_params(axis='x', nbins=5)
                    worstrmseplot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    worstrmseplot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    plt.scatter(worstrmsedata, worstrmsepredicted, marker="o", color="b")
                    worstrmseplot.set_title('Actual vs. Predicted Values for Worst RMSE')
                    worstrmseplot.set_xlabel('Actual Values')
                    worstrmseplot.set_ylabel('Predicted Values')
                    ymin = np.amin(worstrmsepredicted) - np.std(worstrmsepredicted)
                    ymax = np.amax(worstrmsepredicted) + np.std(worstrmsepredicted)
                    xmin = np.amin(worstrmsedata) - np.std(worstrmsedata)
                    xmax = np.amax(worstrmsedata) + np.std(worstrmsedata)
                    worstrmseplot.set_ylim([ymin, ymax])
                    worstrmseplot.set_xlim([xmin, xmax])

                    rmsehist = figure.add_subplot(2, 2, 4)
                    rmsehist.locator_params(axis='x', nbins=5)
                    rmsehist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                    plt.hist(rmsevalues, histtype="bar")
                    rmsehist.set_title('Histogram of RMSE Values')
                    rmsehist.set_xlabel('RMSE Value')
                    rmsehist.set_ylabel('Number in Range')

                    plt.tight_layout()

                    # save the figure
                    # CHANGE: changed plot title to be kFold.
                    titlestring = yvariablenames[i] + "_" + batch + "_kFold.png"
                    # make the figure more readable
                    figure.set_size_inches(14.2, 8)
                    # CHANGE: Save plots to a folder in my computer
                    figure.savefig("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + titlestring)

        # CHANGE: Calculate the average RMSE for this run and added them as input param for createstring method
        avg_rmse = avg_rmse_sum / num_folds
        outputstring = createstring(rmsevalues, bestrmse, worstrmse, avg_rmse, currentyvariable, yvariable, r2values,
                                    bestr2, worstr2)
        output_file.write(outputstring)

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()