import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def createstring(rmsevalues, bestrmse, worstrmse, avg_rmse, currentyvariable, yvariable, r2values, bestr2, worstr2):

    """Creates a formatted string to print to the .txt output file. Upper block creates text file for reading, lower
       block creates a csv used for making plots, etc in excel."""

    # Text document for reading about results
    # outputstring = "-------------------------------------------------------\n" \
    #               "Results for " + str(currentyvariable) + " using 10 Trees per Forest" \
    #               + "\n-------------------------------------------------------\n" \
    #               + "- The mean of the yvariable was " + str(yvariable.mean()) + " and the standard" \
    #               + " deviation was " + str(yvariable.std()) + "\n\n" \
    #               + "- The mean of the RMSE values was " + str(rmsevalues.mean()) + " and the standard" \
    #               + " deviation was " + str(rmsevalues.std()) + ".\n\n"\
    #               + "- The mean of the R^2 values was " + str(r2values.mean()) + " and the standard " \
    #               + " deviation was " + str(r2values.std()) + ".\n\n" \
    #               + "- The best R^2 value was " + str(r2values.max()) + " and the worst was " \
    #               + str(r2values.min()) + ". \n\n"
    # best_string = "- The best RMSE was " + str(bestrmse) + " with an R^2 value of "\
    #              + str(bestr2) + "\n\n"
    # worst_string = "- The worst RMSE was " + str(worstrmse) + "with an R^2 value of "\
    #               + str(worstr2) + "\n\n"
    # outputstring = outputstring + best_string + worst_string +

    # CSV of output for use in excel
    output_string = str(currentyvariable) + "," + str(yvariable.mean()) + "," + str(yvariable.std()) + "," \
        + str(rmsevalues.mean()) + "," + str(rmsevalues.std()) + "\n"

    return output_string


def main():

    # Initialize the data here so that we do not need to onehotencode during every single run
    batchtouse = "batch A"

    if batchtouse == "batch A":

        # Store the name of the batch we are using
        batch = 'batch A'
        # Store the name of the input file
        filename = "encodedBatchAData.csv"
        # Read the input file
        encodedbatchadata = pd.read_csv("encodedBatchAData.csv")
        # separate out the x and y variables
        xvariables = encodedbatchadata.ix[:, 1:16]
        yvariables = encodedbatchadata.ix[:, 16:]
        # Store the names of the x and y variables
        xvariablenames = list(xvariables.columns.values)
        yvariablenames = list(yvariables.columns.values)
        # Store the number of y variables (will be used for later for loops)
        numyvariables = len(yvariablenames)
        # Convert the x and y variables from pandas dataframes to numpy arrays
        xvariables = xvariables.as_matrix()
        yvariables = yvariables.as_matrix()
        yvariables = yvariables.transpose()

        # Normalize the xvariables here. y variables will be later normalized
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std

    else:

        # Store the name of the batch we are using
        batch = 'batch B'
        # Store the name of the input file
        filename = "encodedBatchBData.csv"
        # Read the input file
        encodedbatchbdata = pd.read_csv("encodedBatchBData.csv")
        # separate out the x and y variables
        xvariables = encodedbatchbdata.ix[:, 1:18]
        yvariables = encodedbatchbdata.ix[:, 18:]
        # Store the names of the x and y variables
        xvariablenames = list(xvariables.columns.values)
        print(xvariablenames)
        yvariablenames = list(yvariables.columns.values)
        print(yvariablenames)
        # Store the number of y variables (will be used for later for loops)
        numyvariables = len(yvariablenames)
        # Convert the x and y variables from pandas dataframes to numpy arrays
        xvariables = xvariables.as_matrix()
        yvariables = yvariables.as_matrix()
        yvariables = yvariables.transpose()

        # Normalize the xvariables here. y variables will be later normalized
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std

    # Prompt whether or not creating plots is desired
    makeplots = True
    # Number of folds for CV Tests
    num_folds = 10

    # Create a string that will be added to each of the plots. This will include information such as the specific
    # type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Run on " + dateandtime

    # Parameter values for the random forest regressor
    num_trees = 90
    warm_start_val = True
    criterion_val = "mae"
    bootstrap_val = True

    # Create a .txt file to store the output in:
    #output_file = open("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\output_csv.txt", 'w')
    output_file = open("C:\\Users\\Michael\\PycharmProjects\\Concrete\\output_csv.txt", 'w')

    numberoftestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          + "Total number of y - variables: " + str(numyvariables - 1) + "\n" \
                          + "For this run, " + str(num_folds) + "-fold CV was used and parameter values of: \n"\
                          + "n_estimators: " + str(num_trees) +"\n"  + "warm_start: " + str(warm_start_val) +"\n" \
                          + "criterion: " + criterion_val + "\nbootstrap: " + str(bootstrap_val) \
                          + "\n---------------------------------------------------------------------------\n\n\n" \
                          + "Y variable mean, Y variable std, Mean RMSE, RMSE STD\n"
    output_file.write(numberoftestsstring)

    allrmsedata = list()

    # For each of the y variables, run a random forest using all of the x variables
    for i in range(0, numyvariables):

        # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
        rmsevalues = list()
        # Initialize an array to store the R^2 values in
        r2values = np.array([])

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        yvariable = yvariables[i]
        currentyvariable = yvariablenames[i]

        print(currentyvariable)
        print(yvariable.std())

        # Now, want to normalize the y variable
        y_mean = yvariable.mean()
        y_std = yvariable.std()
        y_normalized = (yvariable - y_mean) / y_std
        y_normalized = np.reshape(y_normalized, newshape=(len(y_normalized), 1))

        # Initialize values for the best and worst RMSE
        bestrmse = 1000000
        bestrmsedata = None
        bestrmsepredicted = None
        worstrmse = 0
        worstrmsedata = np.ones(shape=1)
        worstrmsepredicted = None
        avg_rmse_sum = 0
        # Initialize values for R^2 values that correspond to the best and worst RMSE
        bestr2 = None
        worstr2 = None

        for z in range(0, 10):

            # Create the model here so we use the same model on all sets for a given x
            rand_forest = RandomForestRegressor(n_estimators=num_trees, warm_start=warm_start_val, criterion=criterion_val,
                                            bootstrap=bootstrap_val)

            # Break the data into folds to be used for k-fold CV.
            kf = KFold(n_splits=10, shuffle=True, random_state=None)

            for j in range(0, 10):

                allFoldsPredicted = np.array([])
                allFoldsActual = np.array([])

                # Perform a specified number of CV tests on the data:
                for train_index, test_index in kf.split(yvariable):
                    xtrainvalues, xtestvalues = xvariables[train_index], xvariables[test_index]
                    ytrainvalues_normalized, ytestvalues_normalized = y_normalized[train_index], y_normalized[test_index]

                    # Fit the model
                    rand_forest.fit(xtrainvalues, ytrainvalues_normalized.ravel())
                    # Predict the y values
                    predictedyvalues_normalized = rand_forest.predict(xtestvalues)

                    # De-normalize the data
                    ytestvalues = (ytestvalues_normalized * y_std) + y_mean
                    predictedyvalues = (predictedyvalues_normalized * y_std) + y_mean

                    # Add the data to the complete fold array
                    allFoldsPredicted = np.append(allFoldsPredicted, predictedyvalues)
                    allFoldsActual = np.append(allFoldsActual, ytestvalues)

            # Calculate the RMSE value and add it to the current array.
            rmse = sqrt(mean_squared_error(allFoldsActual, allFoldsPredicted))
            rmsevalues = np.append(rmsevalues, [rmse])

            # Calculate the R^2 value and add it to the array
            r2 = r2_score(allFoldsActual, allFoldsPredicted)
            r2values = np.append(r2values, [r2])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
            if rmse < bestrmse:
                bestrmse = rmse
                bestrmsedata = allFoldsActual
                bestrmsepredicted = allFoldsPredicted
                bestr2 = r2
            elif rmse > worstrmse:
                worstrmse = rmse
                worstrmsedata = allFoldsActual
                worstrmsepredicted = allFoldsPredicted
                worstr2 = r2

        # If we want to make plots, create a figure that will store the subplots (for this particular y variable)
        if makeplots:
            figure = plt.figure(i)
            plottitle = "Results for " + str(currentyvariable) + " Using " + str(num_folds) + " Fold CV"
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
            titlestring = yvariablenames[i] + "_" + batch + "_kFold.png"
            # make the figure more readable
            figure.set_size_inches(14.2, 8)
            # CHANGE: Save plots to a folder in my computer
            #figure.savefig("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + titlestring, bbox_inches
            #              ='tight')
            figure.savefig("C:\\Users\\Michael\\PycharmProjects\\Concrete\\" + titlestring, bbox_inches='tight')

            # Update all variables lists
            normalizedrmsedata = rmsevalues / y_std
            normalizedrmsedata = normalizedrmsedata.reshape(1, np.size(normalizedrmsedata))
            allrmsedata.append(normalizedrmsedata)

        # Check out the trees
        # rand_forest_trees = rand_forest.estimators_
        # i = 1
        # tree_depths = list()
        # tree_info_doc = open("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + currentyvariable
        #                     + " tree_info.txt", 'w')
        # tree_info_doc.write("Max Depth of Trees \n\n")

        # for my_tree in rand_forest_trees:
            # Create a .dot file to visualize the tree using graphviz (I prefer webgraphviz)
            # tree.export_graphviz(my_tree, out_file="C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\Tree " +
            #                     str(i) + ".dot")
        #    curr_depth = my_tree.tree_.max_depth
        #    tree_info_doc.write("Tree " + str(i) + ": " + str(curr_depth) + "\n")
        #    tree_depths.append(curr_depth)
        #    i += 1

        # tree_info_doc.close()

        # Make a bar plot of tree depths
        # tree_depth_plot = plt.figure()
        # tdp = tree_depth_plot.add_subplot(1, 1, 1)
        # tdp.bar(range(1, i), tree_depths)
        # plt.ylabel("Maximum Depth")
        # plt.xlim(xmin=0, xmax=i)
        # plt.title("Maximum Tree Depth for " + str(currentyvariable))
        # tree_depth_plot.savefig("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + currentyvariable
        #                        + " tree_depth_plot.png", bbox_inches='tight')
        # Clear the plot to make sure plots don't overlap.
        # tree_depth_plot.clf()

        # Calculate the average RMSE for this run and added them as input param for createstring method
        avg_rmse = avg_rmse_sum / float(num_folds)
        outputstring = createstring(rmsevalues, bestrmse, worstrmse, avg_rmse, currentyvariable, yvariable,
                                    r2values, bestr2, worstr2)
        output_file.write(outputstring)

    # RMSE boxplot comparing all y variables
    rmseboxplot = plt.figure()
    rmsebp = rmseboxplot.add_subplot(1, 1, 1)
    rmsebp.boxplot(allrmsedata, sym="")
    rmsebp.set_title('Normalized RMSE Data for Each Y Variable Using ' + str(num_folds) + "-fold CV")
    x = range(1, len(yvariablenames))
    plt.xticks(x, yvariablenames[0:len(yvariablenames)], rotation='vertical')
    rmsebp.set_xlabel('Y Variables')
    rmsebp.set_ylabel('RMSE / Y Variable Standard Deviation')
    rmseboxplot.tight_layout()
    #rmseboxplot.savefig("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\RMSE Box and Whisker")
    rmseboxplot.savefig("C:\\Users\\Michael\\PycharmProjects\\Concrete\\RMSE Box and Whisker")

    output_file.close()

# Run the script:
if __name__ == '__main__':
    main()
