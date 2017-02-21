import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# TODO readcsv()
def readcsv():

    """ Reads the data from a specified .csv file and returns it as a numpy array. Also returns a pandas dataframe to
    retrieve the column headers from."""

    print("Note that data must be in the form: \"Sample number / y-variables / categorical x-variables / numerical"
          " x-variables\"")
    filename = input("What is the name of the .csv file to read from? ")

    # import the file:
    completearray = pd.read_csv(filename, delimiter=',')

    # Prompt for how many y variables the data contains. The rest are considered x variables.
    numyvariables = int(input("How many y variables are there? NOTE: Add 1 to the total to account for the column"
                              " labels. "))

    # get the names of the variables
    variablenames = list(completearray.columns.values)
    yvariablenames = variablenames[:numyvariables]

    # Separate the x data into a separate array
    xvariables = completearray.iloc[:, numyvariables:len(completearray.columns)]

    # Prompt how many categorical values there are. These will be onehotencoded into binary.
    numcatvariables = int(input("How many of the x variables are categorical? "))

    # Separate the x data into categorical and numerical values
    xcategoricalarray = xvariables.iloc[:, 0:numcatvariables]
    xnumericalarray = xvariables.iloc[:, numcatvariables:len(xvariables.columns)]

    # Onehotencode the categorical x variables
    xcategoricalarray = pd.get_dummies(xcategoricalarray)

    # combine the binarized categorical variables with the numerical variables
    xvariables = pd.concat([xcategoricalarray, xnumericalarray], axis=1)
    xvariablenames = list(xvariables.columns.values)

    # Turn the complete array and the xvariables array into numpy arrays so that may be more easily used.
    completenumpyarray = np.transpose(completearray.as_matrix())
    xvariables = xvariables.as_matrix()

    return completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables


# TODO: generaterandomindices
def generaterandomindices(datasize, percenttest):

    """ For the given datasize from the .csv file generates a set of random indices to use for the training set. Uses a
    user defined percentage of the data for training and testing. """

    trainindices = np.array([])
    while trainindices.__len__() < int(percenttest * datasize):
        # Randomly select an index value and store it. If it has already been chosen, pick again.
        index = int(random.random() * datasize)
        if not trainindices.__contains__(index):
            trainindices = np.append(trainindices, [index])

    # For aesthetic purposes:
    trainindices = np.sort(trainindices)

    return trainindices


# TODO: createtrainingarrays
def createtrainingarrays(datasize, xvariables, yvariable, trainindices):

    """ Creates a random training array of x and y values from the previously generated random training indices."""

    # For the desired training indices, add the values to the training arrays
    xtrainvalues = np.array([])
    ytrainvalues = np.array([])
    indexcounter = 0
    for q in range(0, datasize):
        if trainindices.__contains__(q):

            if indexcounter is 0:
                xtrainvalues = xvariables[q]
                indexcounter = -1
            else:
                xtrainvalues = np.vstack((xtrainvalues, xvariables[q]))

            ytrainvalues = np.append(ytrainvalues, yvariable[0][q])

    # Reshape the data to proper dimensions so that a linear regression may be performed
    length = ytrainvalues.size
    ytrainvalues = ytrainvalues.reshape(length, 1)

    return xtrainvalues, ytrainvalues


# TODO: createtestarrays
def createtestarrays(datasize, xvariables, yvariable, trainindices):

    """ Uses the remaining values from the dataset (those not included in the training data) to form a set of testing
    arrays. """

    xtestvalues = np.array([])
    ytestvalues = np.array([])
    indexcounter1 = 0
    for p in range(0, datasize):
        if not trainindices.__contains__(p):
            if indexcounter1 is 0:
                xtestvalues = xvariables[p]
                indexcounter1 = -1
            else:
                xtestvalues = np.vstack((xtestvalues, xvariables[p]))
            ytestvalues = np.append(ytestvalues, yvariable[0][p])

    # Reshape array to proper size
    length = ytestvalues.size
    ytestvalues = ytestvalues.reshape(length, 1)

    return xtestvalues, ytestvalues


# TODO: createstring
def createstring(rmsevalues, bestrmse, worstrmse, currentyvariable,yvariable):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "-------------------------\n" \
                   "Results for " + str(currentyvariable) + ":" \
                                                            "\n-------------------------\n" \
                   + "- The mean of the y-variable was " + "{0:.4e}".format(yvariable.mean()) + " and the standard" \
                   + " deviation was " + "{0:.4e}".format(yvariable.std()) + ".\n\n" \
                   + "- The mean of the RMSE values was " + "{0:.4e}".format(rmsevalues.mean()) + " and the standard" \
                   + " deviation was " + "{0:.4e}".format(rmsevalues.std()) + ".\n\n"

    best_string = "- The best RMSE was " + "{0:.4e}".format(bestrmse) + "\n\n"

    worst_string = "- The worst RMSE was " + "{0:.4e}".format(worstrmse) + "\n\n\n"

    outputstring = outputstring + best_string + worst_string

    return outputstring


# TODO: Update this to run a decision tree (classification / regression?) instead of a linear regression
def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables = readcsv()

    # Prompt for number of CV tests to run on each y variable:
    numberoftests = int(input("How many CV tests should be done for each y variable? "))

    # Prompt for the percentage of data to use for training. The remaining amount will be used for testing.
    percenttest = int(input("What percentage of data do you want to use for training? ")) / 100

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt", 'w')
    numberoftestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + "Total number of y - variables: " + str(numyvariables - 1) + "\n" \
                          + "For this run, the total number of CV tests done on each y-variable was "\
                          + str(numberoftests)\
                          + ".\nFurthermore, " + str(percenttest*100) + "% of the data was used for training.\n" \
                          "---------------------------------------------------------------------------\n\n\n"
    output_file.write(numberoftestsstring)

    # For each of the y variables, want to run a decision tree using all of the x variables
    for i in range(1, numyvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        datasize = np.size(completenumpyarray[i])
        yvariable = completenumpyarray[i].reshape(1, datasize)
        currentyvariable = yvariablenames[i]

        # Create a figure that will store the subplots (for this particular y variable)
        figure = plt.figure(i)
        plottitle = "Results for " + str(currentyvariable)
        figure.suptitle(plottitle)

        # Create a histogram of the current y variable
        histogramofy = figure.add_subplot(2, 2, 1)
        histogramofy.locator_params(axis='x', nbins=5)
        histogramofy.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(np.transpose(yvariable), histtype="bar")
        histogramofy.set_title('Histogram of Y-Values')
        histogramofy.set_xlabel('Y-Values')
        histogramofy.set_ylabel('Number in Range')

        # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
        rmsevalues = np.array([])
        # Initialize values for the best and worst RMSE
        bestrmse = 1000000
        bestrmsedata = None
        bestrmsepredicted = None
        worstrmse = 0
        worstrmsedata = None
        worstrmsepredicted = None

        # Perform a specified number of CV tests on the data:
        for z in range(0, numberoftests):

            # Randomly break the data up into training and testing. Will use input percentage for training,
            # 20% for testing.
            trainindices = generaterandomindices(datasize, percenttest)
            xtrainvalues, ytrainvalues = createtrainingarrays(datasize, xvariables, yvariable, trainindices)

            # TODO: this is the main area that would need to be reworked. I believe that the rest of the training /
            # TODO: testing random selection should translate just fine to this.
            dtree = DecisionTreeRegressor()
            dtree.fit(xtrainvalues, ytrainvalues)

            # Now, want to run cross validation test

            # Start by creating the testing arrays:
            xtestvalues, ytestvalues = createtestarrays(datasize, xvariables, yvariable, trainindices)

            # Predict the values
            predictedyvalues = dtree.predict(xtestvalues)

            # Calculate the RMSE value and add it to the current array.
            rmse = sqrt(mean_squared_error(ytestvalues, predictedyvalues))
            rmsevalues = np.append(rmsevalues, [rmse])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
            if rmse < bestrmse:
                bestrmse = rmse
                bestrmsedata = ytestvalues
                bestrmsepredicted = predictedyvalues
            elif rmse > worstrmse:
                worstrmse = rmse
                worstrmsedata = ytestvalues
                worstrmsepredicted = predictedyvalues

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

        outputstring = createstring(rmsevalues, bestrmse, worstrmse, currentyvariable,yvariable)
        output_file.write(outputstring)

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
