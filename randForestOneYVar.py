import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score


# TODO initializeData()
def initializeData():

    """ Initializes the data taken from the completeData.csv and the formattedXValues.csv. Note that these must be
    the names of the arrays in your folder."""

    # Read in the CSV
    all_variables = pd.read_csv('completeData.csv', keep_default_na=False)
    x_values = pd.read_csv('formattedXValues.csv')
    filename = "completeData.csv and formattedXValues.csv"
    # Separate the CSV columns into array variables and numpy vars to store new categorical variables
    mix_num = all_variables['Mix Number']
    mix_p = all_variables['Mix Proportion']
    mix_p_final = np.empty(len(mix_p))
    scm = all_variables['SCM']
    scm_final = np.empty(len(scm))
    fine_a = all_variables['Fine Aggregate']
    fine_a_final = np.empty(len(fine_a))
    coarse_a = all_variables['Coarse Aggregate']
    coarse_a_final = np.empty(len(coarse_a))

    # Loop through every mix in the csv file
    for y in range(0, len(mix_num)):
        # Sort Mix Proportions
        if mix_p[y] == "A-F":
            mix_p_final[y] = 2
        elif mix_p[y] == "A-S":
            mix_p_final[y] = 1
        elif mix_p[y] == "A":
            mix_p_final[y] = 0
        else:
            print('Unidentified Variable in mixP: ')
            print(mix_p[y])

        # Sort SCM into slag or fly ash
        if scm[y] == 'N/A':
            scm_final[y] = 1000
        elif scm[y] == 'Slag 1':
            scm_final[y] = 0
        elif scm[y] == 'Slag 2':
            scm_final[y] = 0
        elif scm[y] == 'Fly Ash 1':
            scm_final[y] = 1
        elif scm[y] == 'Fly Ash 2':
            scm_final[y] = 1
        elif scm[y] == 'Fly Ash 3':
            scm_final[y] = 1
        else:
            print('Unidentified Variable in scm: ')
            print(scm[y])

        # Sort the fine aggregate
        if fine_a[y] == 'Sand A':
            fine_a_final[y] = 0
        elif fine_a[y] == 'Sand B':
            fine_a_final[y] = 1
        else:
            print('Unidentified Variable in fineA: ')
            print(fine_a[y])

        # Sort the coarse aggregate
        if coarse_a[y] == 'GG1':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'GG2':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'GG3':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'GG4':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'GG5':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'GG6':
            coarse_a_final[y] = 0
        elif coarse_a[y] == 'CS1':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS2':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS3':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS4':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS5':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS6':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS7':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS8':
            coarse_a_final[y] = 1
        elif coarse_a[y] == 'CS9':
            coarse_a_final[y] = 1
        else:
            print('Unidentified Variable in coarseA: ')
            print(coarse_a[y])

    # One Hot Encode the sorted variables
    encoded_mix_p = pd.get_dummies(mix_p_final)
    encoded_scm = pd.get_dummies(scm_final)
    encoded_fine_a = pd.get_dummies(fine_a_final)
    encoded_coarse_a = pd.get_dummies(coarse_a_final)

    # Update the headers for onehotencoded variables
    # Get the current variable names
    encoded_scm_list = list(encoded_scm.columns.values)
    encoded_finea_list = list(encoded_fine_a.columns.values)
    encoded_coarsea_list = list(encoded_coarse_a.columns.values)
    encoded_mixp_list = list(encoded_mix_p.columns.values)
    # go through and replace the current names with the updated ones
    encoded_scm.rename(columns={encoded_scm_list[0]: 'SCM_0', encoded_scm_list[1]: 'SCM_1', encoded_scm_list[2]:
                                'SCM_1000'}, inplace=True)
    encoded_fine_a.rename(columns={encoded_finea_list[0]: 'FineA_0', encoded_finea_list[1]: 'FineA_1'}, inplace=True)
    encoded_coarse_a.rename(columns={encoded_coarsea_list[0]: 'CoarseA_0', encoded_coarsea_list[1]: 'CoarseA_1'},
                            inplace=True)
    encoded_mix_p.rename(columns={encoded_mixp_list[0]: 'MixP_0', encoded_mixp_list[1]: 'MixP_1', encoded_mixp_list[2]:
                                  'MixP_2'}, inplace=True)

    # Remake the dataframe to include the onehotencoded columns instead of the regular columns.
    first_half = all_variables.ix[:, :21]
    cte = all_variables.ix[:, 25]
    onehot_encoded_frame = pd.concat([encoded_mix_p, encoded_scm, encoded_fine_a, encoded_coarse_a], axis=1)
    second_half = x_values.ix[:, 6:]

    completearray = pd.concat([first_half, cte, onehot_encoded_frame, second_half], axis=1)
    variablenames = list(completearray.columns.values)
    # convert to numpy array
    completenumpyarray = completearray.as_matrix()

    # Prompt which batch of data to use
    batch = input("Which batch would you like to run on (A or B)? \n Batch A: CTE, MoR \n Batch B: " +
                  "CS, MoE, Poisson's Ratio, Splitting Tensile Strength \n")

    # Extract values from Batch A
    if batch == "A":
        y_var = input("Which y variable would you like to test: \n 1) MoR 7 Day\n 2) MoR 14 Day\n 3) MoR 28 Day\n" +
                      " 4) MoR 90 Day\n 5) CTE 28 Day\n")
        batch_a_ycolumns = [5, 6, 7, 8, 21]
        yvariable = np.transpose(completenumpyarray[:, batch_a_ycolumns[int(y_var) - 1]])
        yvariablename = variablenames[batch_a_ycolumns[int(y_var) - 1]]
        batch_a_xcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 32, 35, 38, 41]
        xvariables = completenumpyarray[:, batch_a_xcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variablenames[x] for x in batch_a_xcolumns]

    # Extract values from Batch B
    elif batch == "B":
        y_var = input("Which y variable would you like to test: \n 1) CS 7 Day\n 2) CS 14 Day\n 3) CS 28 Day\n" +
                      " 4) CS 90 Day\n 5) MoE 7 Day\n 6) MoE 14 Day\n 7) MoE 28 Day\n 8) MoE 90 Day\n 9) PR 7 Day\n " +
                      "10) PR 14 Day\n 11) PR 28 Day\n 12) PR 90 Day\n 13) STS 7 Day\n 14) STS 14 Day\n " +
                      "15) STS 28 Day\n 16) STS 90 Day\n")
        batch_b_ycolumns = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        yvariable = np.transpose(completenumpyarray[:, batch_b_ycolumns[int(y_var) - 1]])
        yvariablename = variablenames[batch_b_ycolumns[int(y_var) - 1]]
        batch_b_xcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 33, 36, 39, 42]

        # Normalize the x variables.
        xvariables = completenumpyarray[:, batch_b_xcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and normalize data by subtracting mean, dividing standard dev.
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variablenames[x] for x in batch_b_xcolumns]

    else:
        print("Invalid Input.")
        exit(0)

    return completenumpyarray, xvariables, filename, xvariablenames, yvariablename, yvariable, batch


# TODO: createstring
def createstring(rmsevalues, bestrmse, worstrmse, currentyvariable, yvariable, r2values, bestr2, worstr2,
                 num_trees, tree_depth):

    """Creates a formatted string to print to the .txt output file"""

    outputstring = "-------------------------------------------------------\n" \
                   "Results for " + str(currentyvariable) + " using " + str(num_trees) + " Trees per Forest and\n" \
                   + "a max tree depth of " + str(tree_depth) \
                   + "\n-------------------------------------------------------\n" \
                   + "- The mean of the yvariable was " + str(yvariable.mean()) + " and the standard" \
                   + " deviation was " + str(yvariable.std()) + "\n\n" \
                   + "- The mean of the RMSE values was " + str(rmsevalues.mean()) + " and the standard" \
                   + " deviation was " + str(rmsevalues.std()) + ".\n\n"\
                   + "- The mean of the R^2 values was " + str(r2values.mean()) + " and the standard " \
                   + " deviation was " + str(r2values.std()) + ".\n\n" \
                   + "- The best R^2 value was " + str(r2values.max()) + " and the worst was " \
                   + str(r2values.min()) + ". \n\n"

    best_string = "- The best RMSE was " + str(bestrmse) + " with an R^2 value of "\
                  + str(bestr2) + "\n\n"

    worst_string = "- The worst RMSE was " + str(worstrmse) + "with an R^2 value of "\
                   + str(worstr2) + "\n\n"

    outputstring = outputstring + best_string + worst_string
    return outputstring


def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablename, yvariable, batch = initializeData()

    # Prompt whether or not creating plots is desired
    make_plots = True

    # Number of folds for CV Tests
    # num_folds = 10
    kf = KFold(n_splits=10, shuffle=True, random_state=None)

    # Create a string that will be added to each of the plots. This will include information such as the specific
    #  type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Run on " + dateandtime

    # Create a .txt file to store the output in:
    output_file = open("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + yvariablename
                       + " output_file.txt", 'w')

    numberoftestsstring = "\n---------------------------------------------------------------------------\n" \
                          "Name of input file: " + filename + "\n" \
                          "\nFor this run, " + str(kf.get_n_splits()) + "-fold CV was used. \n"\
                          + "---------------------------------------------------------------------------\n\n\n"
    output_file.write(numberoftestsstring)

    all_rmse_data = list()

    # Normalize the y variable
    y_mean = yvariable.mean()
    y_std = yvariable.std()
    y_normalized = (yvariable - y_mean) / y_std
    y_normalized = np.reshape(y_normalized, newshape=(len(y_normalized), 1))

    # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
    rmsevalues = list()
    # Initialize an array to store the R^2 values in
    r2values = np.array([])
    # Initialize values for the best and worst RMSE
    bestrmse = 1000000
    bestrmsedata = None
    bestrmsepredicted = None
    worstrmse = 0
    worstrmsedata = np.ones(shape=1)
    worstrmsepredicted = None
    # Initialize values for R^2 values that correspond to the best and worst RMSE
    bestr2 = None
    worstr2 = None


    rand_forest = RandomForestRegressor(n_estimators=90, criterion="mae", warm_start=True, bootstrap=True)

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

    # If we want to make plots, create a figure that will store the subplots
    if make_plots:
        figure = plt.figure()
        plottitle = "Results for " + str(yvariablename) + " Using " + str(kf.get_n_splits()) + " Fold CV"
        figure.suptitle(plottitle, fontsize=16)
        figure.text(0, 0, additionalinfo)

        # add the plots for best, worst fits as well as a plot of the RMSE and r^2 values from the tests.
        rsquaredhist = figure.add_subplot(2, 2, 1)
        rsquaredhist.locator_params(axis='x', nbins=10)
        plt.hist(r2values, histtype="bar")
        rsquaredhist.set_title('Histogram of r^2 Values', fontsize=16)
        rsquaredhist.set_xlabel('r^2 Value', fontsize=16)
        rsquaredhist.set_ylabel('Number in Range', fontsize=16)

        # add the plots for best, worst fits as well as a plot of the RMSE values from the tests.
        bestrmseplot = figure.add_subplot(2, 2, 2)
        bestrmseplot.locator_params(axis='x', nbins=5)
        plt.scatter(bestrmsedata, bestrmsepredicted, marker="o", color="b")
        bestrmseplot.set_title('Actual vs. Predicted Values for Best RMSE', fontsize=16)
        bestrmseplot.set_xlabel('Actual Values', fontsize=16)
        bestrmseplot.set_ylabel('Predicted Values', fontsize=16)
        ymin = np.amin(bestrmsepredicted) - np.std(bestrmsepredicted)
        ymax = np.amax(bestrmsepredicted) + np.std(bestrmsepredicted)
        xmin = np.amin(bestrmsedata) - np.std(bestrmsedata)
        xmax = np.amax(bestrmsedata) + np.std(bestrmsedata)
        bestrmseplot.set_ylim([ymin, ymax])
        bestrmseplot.set_xlim([xmin, xmax])

        worstrmseplot = figure.add_subplot(2, 2, 3)
        worstrmseplot.locator_params(axis='x', nbins=5)
        plt.scatter(worstrmsedata, worstrmsepredicted, marker="o", color="b")
        worstrmseplot.set_title('Actual vs. Predicted Values for Worst RMSE', fontsize=16)
        worstrmseplot.set_xlabel('Actual Values', fontsize=16)
        worstrmseplot.set_ylabel('Predicted Values', fontsize=16)
        ymin = np.amin(worstrmsepredicted) - np.std(worstrmsepredicted)
        ymax = np.amax(worstrmsepredicted) + np.std(worstrmsepredicted)
        xmin = np.amin(worstrmsedata) - np.std(worstrmsedata)
        xmax = np.amax(worstrmsedata) + np.std(worstrmsedata)
        worstrmseplot.set_ylim([ymin, ymax])
        worstrmseplot.set_xlim([xmin, xmax])

        rmsehist = figure.add_subplot(2, 2, 4)
        rmsehist.locator_params(axis='x', nbins=5)
        plt.hist(rmsevalues, histtype="bar")
        rmsehist.set_title('Histogram of RMSE Values', fontsize=16)
        rmsehist.set_xlabel('RMSE Value', fontsize=16)
        rmsehist.set_ylabel('Number in Range', fontsize=16)

        plt.tight_layout()
        # save the figure
        titlestring = str(yvariablename) + "_MaxTreeDepth" + str(tree_depth) + "_kFold.png"
        # make the figure more readable
        figure.set_size_inches(14.2, 8)
        figure.savefig("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + titlestring)

    # Update all variables lists
    normalizedrmsedata = rmsevalues / y_std
    normalizedrmsedata = normalizedrmsedata.reshape(1, np.size(normalizedrmsedata))
    all_rmse_data.append(normalizedrmsedata)

    # Calculate the average RMSE for this run and added them as input param for createstring method
    outputstring = createstring(rmsevalues, bestrmse, worstrmse, yvariablename, yvariable, r2values, bestr2,
                                worstr2, num_trees=90, tree_depth=tree_depth)
    output_file.write(outputstring)

    # Check out the trees
    rand_forest_trees = rand_forest.estimators_
    i = 1
    tree_depths = list()
    tree_info_doc = open("C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\" + yvariablename
                         + " tree_info.txt", 'w')
    tree_info_doc.write("Max Depth of trees \n\n")

    for my_tree in rand_forest_trees:
        # tree.export_graphviz(my_tree, out_file="C:\\Users\\mvane\\Documents\\Skunkworks\\Random Forest Results\\Tree " +
        #                     str(i) + ".dot")
        curr_depth = my_tree.tree_.max_depth
        tree_info_doc.write("Tree " + str(i) + ": " + str(curr_depth) + "\n")
        tree_depths.append(curr_depth)
        i += 1

    output_file.close()
    tree_info_doc.close()

# Run the script:
if __name__ == '__main__':
    main()
