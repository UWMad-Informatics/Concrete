import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold



def initialize_data():

    """ Initializes the data taken from the completeData.csv and the formattedXValues.csv. Note that these must be
    the names of the arrays in your folder."""

    # Read in the CSV
    all_x = pd.read_csv('completeData.csv', keep_default_na=False)
    x_values = pd.read_csv('formattedXValues.csv')
    filename = "completeData.csv and formattedXValues.csv"

    # Separate the CSV columns into array variables and numpy vars to store new categorical variables
    mix_num = all_x['Mix Number']
    mix_p = all_x['Mix Proportion']
    mix_p_final = np.empty(len(mix_p))
    scm = all_x['SCM']
    scm_final = np.empty(len(scm))
    fine_a = all_x['Fine Aggregate']
    fine_a_final = np.empty(len(fine_a))
    coarse_a = all_x['Coarse Aggregate']
    coarse_a_final = np.empty(len(coarse_a))

    # Loop through every mix in the csv file
    # Not sure how to do 3 different variables
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
    encoded_fine_a_list = list(encoded_fine_a.columns.values)
    encoded_coarse_a_list = list(encoded_coarse_a.columns.values)
    encoded_mix_p_list = list(encoded_mix_p.columns.values)
    # go through and replace the current names with the updated ones
    encoded_scm.rename(columns={encoded_scm_list[0]: 'SCM_0', encoded_scm_list[1]: 'SCM_1', encoded_scm_list[2]:
                       'SCM_1000'}, inplace=True)
    encoded_fine_a.rename(columns={encoded_fine_a_list[0]: 'FineA_0', encoded_fine_a_list[1]: 'FineA_1'}, inplace=True)
    encoded_coarse_a.rename(columns={encoded_coarse_a_list[0]: 'CoarseA_0', encoded_coarse_a_list[1]: 'CoarseA_1'},
                            inplace=True)
    encoded_mix_p.rename(columns={encoded_mix_p_list[0]: 'MixP_0', encoded_mix_p_list[1]: 'MixP_1',
                                  encoded_mix_p_list[2]: 'MixP_2'}, inplace=True)

    # Remake the dataframe to include the onehotencoded columns instead of the regular columns.
    first_half = all_x.ix[:, :21]
    cte = all_x.ix[:, 25]
    onehot_encoded_frame = pd.concat([encoded_mix_p, encoded_scm, encoded_fine_a, encoded_coarse_a], axis=1)
    second_half = x_values.ix[:, 6:]
    complete_array = pd.concat([first_half, cte, onehot_encoded_frame, second_half], axis=1)
    variable_names = list(complete_array.columns.values)
    # convert to numpy array
    completenumpyarray = complete_array.as_matrix()

    # Now, Ask whether or not to run decision trees on batch A data or batch B
    batch = input("Which batch to run tests on (A or B)? ")

    if batch == "A":

        # break up the data into the batch A values
        batch_a_ycolumns = [0, 5, 6, 7, 8, 21]
        yvariables = np.transpose(completenumpyarray[:, batch_a_ycolumns])
        numyvariables = 6
        yvariablenames = [variable_names[x] for x in batch_a_ycolumns]
        batch_a_xcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 32, 35, 38, 41]
        # normalize the x variables. Will normalize y variables in the main body
        # after a histogram of the data is created.
        xvariables = completenumpyarray[:, batch_a_xcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and normalize
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            # index through each value of the column (thus, go through each row) and divide by the l2 norm
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variable_names[x] for x in batch_a_xcolumns]

    elif batch == "B":

        # break up the data into the batch B values
        batch_b_ycolumns = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        yvariables = np.transpose(completenumpyarray[:, batch_b_ycolumns])
        numyvariables = 17
        yvariablenames = [variable_names[x] for x in batch_b_ycolumns]
        batch_b_xcolumns = [23, 24, 25, 26, 28, 29, 30, 31, 33, 36, 39, 42]
        # normalize the x variables. Will normalize y variables in the main body
        # after a histogram of the data is created.
        xvariables = completenumpyarray[:, batch_b_xcolumns]
        # Normalize each of the x variables
        # get number of columns of x variables
        x_variables_shape = xvariables.shape
        # index through each of the columns and find the l2 norm
        for p in range(0, x_variables_shape[1]):
            x_mean = xvariables[:, p].mean()
            x_std = xvariables[:, p].std()
            # index through each value of the column (thus, go through each row) and divide by the l2 norm
            xvariables[:, p] = (xvariables[:, p] - x_mean) / x_std
        xvariablenames = [variable_names[x] for x in batch_b_xcolumns]

    else:
        print("Invalid Input.")
        exit(0)

    return completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch


def createstring(y_variable, rmse_values, best_rmse, best_coef, best_intercept, worst_rmse, worst_coef, worst_intercept,
                 current_y_variable, x_variable_names, r2values, bestr2, worstr2):

    """Creates a formatted string to print to the .txt output file"""

    # output_string = "-------------------------\n" \
    #               "Results for " + str(current_y_variable) + ":" \
    #               "\n-------------------------\n"\
    #               + "- The mean of the y-variable was " + "{0:.4e}".format(y_variable.mean()) + " and the standard" \
    #               + " deviation was " + "{0:.4e}".format(y_variable.std()) + ".\n\n" \
    #               + "- The mean of the RMSE values was " + "{0:.4e}".format(rmse_values.mean()) + " and the standard" \
    #              + " deviation was " + "{0:.4e}".format(rmse_values.std()) + ".\n\n" \
    #               + "- The mean of the R^2 values was " + "{0:.4}".format(r2values.mean()) + " and the standard " \
    #               + " deviation was " + "{0:.4}".format(r2values.std()) + ".\n\n" \
    #               + "- The best R^2 value was " + "{0:.4}".format(r2values.max()) + " and the worst was "\
    #               + "{0:.4}".format(r2values.min()) + ". \n\n"

    # best_regression_string = "- The best RMSE was " + "{0:.4e}".format(best_rmse) + " with an R^2 value of "\
    #                        + "{0:.4}".format(bestr2) + " and a regression equation of:\n "

    # for m in range(0, best_coef[0].__len__()):
    #    best_regression_string = best_regression_string + "{0:.4e}".format(best_coef[0][m]) + "*" + \
    #                             "(" + x_variable_names[m] + ")" + " + "
    #    best_regression_string = best_regression_string + "{0:.4e}".format(best_intercept) + ".\n\n"

    # worst_regression_string = "- The worst RMSE was " + "{0:.4e}".format(worst_rmse) + "with an R^2 value of "\
    #                         + "{0:.4}".format(worstr2) + " and a regression equation of:\n   "

    #for n in range(0, worst_coef[0].__len__()):
    #    worst_regression_string = worst_regression_string + "{0:.4e}".format(worst_coef[0][n]) + "*" + \
    #                              "(" + x_variable_names[n] + ")" + " + "
    # worst_regression_string = worst_regression_string + "{0:.4e}".format(worst_intercept) + ".\n\n\n"

    # outputstring = outputstring + best_regression_string + worst_regression_string

    # CSV of output for use in excel
    output_string = str(current_y_variable) + "," + str(rmse_values.mean()) + "," + str(rmse_values.std()) + "\n"

    return output_string


def main():

    completenumpyarray, xvariables, filename, xvariablenames, yvariablenames, numyvariables, yvariables, batch \
        = initialize_data()

    # Prompt whether or not creating plots is desired
    makeplots = True
    num_folds = 10

    # Create a string that will be added to each of the plots. This will include information such as the specific
    #  type of test as well as the date and time it was run. This will also be added to the output .txt file.
    dateandtime = time.strftime("%Y-%m-%d at %H:%M")
    additionalinfo = "Multivariate Linear Regression run on " + dateandtime

    # Create a .txt file to store the output in:
    output_file = open("output_file.txt", 'w')
    number_of_tests_string = "\n---------------------------------------------------------------------------\n" \
                             + "Results for Multivariate Linear Regression\n" \
                             + "Name of input file: " + filename + "\n" \
                             + additionalinfo + "\n" \
                             + "Total number of y - variables: " + str(numyvariables-1) + "\n" \
                             + ".\nFurthermore, " + str(num_folds) + " folds were used for cross validation.\n" \
                             + "---------------------------------------------------------------------------\n\n\n"
    output_file.write(number_of_tests_string)

    # Create two arrays. These will be arrays of arrays. One will store all of the RMSE data (normalized by std) and the
    # other will contain all of the r^2 data. These will then be used to create box and whisker plots of all the data
    # side by side for comparison of fit.
    allrmsedata = list()
    allrsquareddata = list()

    # For each of the Y variables, want to run a linear regression:
    for i in range(1, numyvariables):

        # Separate out the current y variable, shape it to appropriate dimensions so that it matches the x variables
        y_variable = yvariables[i]
        current_y_variable = yvariablenames[i]

        # Want to normalize the y variable
        y_mean = y_variable.mean()
        y_std = y_variable.std()
        y_normalized = (y_variable - y_mean) / y_std
        y_normalized = np.reshape(y_normalized, newshape=(len(y_normalized), 1))

        # Initialize an array to store the RMSE values in (these will be used later during cross validation tests).
        rmse_values = np.array([])
        # Initialize values for the best and worst RMSE values
        best_rmse = 1000000
        best_rmse_data = None
        best_rmse_predicted = None
        best_coef = None
        best_intercept = None
        worst_rmse = 0
        worst_rmse_data = None
        worst_rmse_predicted = None
        worst_coef = None
        worst_intercept = None
        # Initialize values for R^2 values that correspond to the best and worst RMSE
        bestr2 = None
        worstr2 = None

        # Initialize an array to store the R^2 values in
        r2values = np.array([])

        # Create the splitter for k-fold CV
        kf = KFold(n_splits=10, shuffle=True, random_state=None)
        regr = linear_model.LinearRegression(fit_intercept=False)

        # Perform a specified number of CV tests on the data:
        for train_index, test_index in kf.split(y_variable):
            xtrainvalues, xtestvalues = xvariables[train_index], xvariables[test_index]
            ytrainvalues_normalized, ytestvalues_normalized = y_normalized[train_index], y_normalized[test_index]

            # Fit the model
            regr.fit(xtrainvalues, ytrainvalues_normalized)

            # Now, want to run cross validation test
            # Predict the values
            predictedyvalues_normalized = regr.predict(xtestvalues)

            # De-normalize the data
            ytestvalues = ytestvalues_normalized * y_std + y_mean
            predicted_y_values = predictedyvalues_normalized * y_std + y_mean

            # Calculate the RMSE value and add it to the current array.
            rmse = sqrt(mean_squared_error(ytestvalues, predicted_y_values))
            rmse_values = np.append(rmse_values, [rmse])

            # Calculate the R^2 value and add it to the array
            r2 = r2_score(ytestvalues, predicted_y_values)
            r2values = np.append(r2values, [r2])

            # Check whether or not this RMSE is the best / worst RMSE of the current y variable
            if rmse < best_rmse:
                best_rmse = rmse
                best_rmse_data = ytestvalues
                best_rmse_predicted = predicted_y_values
                best_coef = regr.coef_
                best_intercept = regr.intercept_
                bestr2 = r2
            elif rmse > worst_rmse:
                worst_rmse = rmse
                worst_rmse_data = ytestvalues
                worst_rmse_predicted = predicted_y_values
                worst_coef = regr.coef_
                worst_intercept = regr.intercept_
                worstr2 = r2

        # Depending on whether or not makeplots is true or false
        if makeplots is True:

            # Create a figure that will store the subplots (for this particular y variable)
            figure = plt.figure(i)
            plot_title_string = "Results for " + str(current_y_variable)
            figure.suptitle(plot_title_string)
            figure.text(0, 0, additionalinfo)

            # add the plots for best, worst fits as well as a plot of the RMSE and r^2 values from the tests.
            rsquaredhist = figure.add_subplot(2, 2, 1)
            rsquaredhist.locator_params(axis='x', nbins=10)
            rsquaredhist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            plt.hist(r2values, histtype="bar")
            rsquaredhist.set_title('Histogram of r^2 Values')
            rsquaredhist.set_xlabel('r^2 Value')
            rsquaredhist.set_ylabel('Number in Range')

            best_rmse_plot = figure.add_subplot(2, 2, 2)
            best_rmse_plot.locator_params(axis='x', nbins=5)
            best_rmse_plot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            best_rmse_plot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.scatter(best_rmse_data, best_rmse_predicted, marker="o", color="b")
            best_rmse_plot.set_title('Actual vs. Predicted Values for Best RMSE')
            best_rmse_plot.set_xlabel('Actual Values')
            best_rmse_plot.set_ylabel('Predicted Values')
            ymin = np.amin(best_rmse_predicted) - np.std(best_rmse_predicted)
            ymax = np.amax(best_rmse_predicted) + np.std(best_rmse_predicted)
            xmin = np.amin(best_rmse_data) - np.std(best_rmse_data)
            xmax = np.amax(best_rmse_data) + np.std(best_rmse_data)
            best_rmse_plot.set_ylim([ymin, ymax])
            best_rmse_plot.set_xlim([xmin, xmax])

            worst_rmse_plot = figure.add_subplot(2, 2, 3)
            worst_rmse_plot.locator_params(axis='x', nbins=5)
            worst_rmse_plot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            worst_rmse_plot.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.scatter(worst_rmse_data, worst_rmse_predicted, marker="o", color="b")
            worst_rmse_plot.set_title('Actual vs. Predicted Values for Worst RMSE')
            worst_rmse_plot.set_xlabel('Actual Values')
            worst_rmse_plot.set_ylabel('Predicted Values')
            ymin = np.amin(worst_rmse_predicted) - np.std(worst_rmse_predicted)
            ymax = np.amax(worst_rmse_predicted) + np.std(worst_rmse_predicted)
            xmin = np.amin(worst_rmse_data) - np.std(worst_rmse_data)
            xmax = np.amax(worst_rmse_data) + np.std(worst_rmse_data)
            worst_rmse_plot.set_ylim([ymin, ymax])
            worst_rmse_plot.set_xlim([xmin, xmax])

            rmse_hist = figure.add_subplot(2, 2, 4)
            rmse_hist.locator_params(axis='x', nbins=5)
            rmse_hist.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.hist(rmse_values, histtype="bar")
            rmse_hist.set_title('Histogram of RMSE Values')
            rmse_hist.set_xlabel('RMSE Value')
            rmse_hist.set_ylabel('Number in Range')

            plt.tight_layout()

            # save the figure
            titlestring = yvariablenames[i] + "_" + batch + "_kFold.png"
            # make the figure more readable
            figure.set_size_inches(14.2, 8)
            figure.savefig(titlestring)

        # Add the normalized RMSE and r^2 values to the complete arrays
        normalizedrmsedata = rmse_values / y_variable.std()
        normalizedrmsedata = normalizedrmsedata.reshape(1, np.size(normalizedrmsedata))
        allrmsedata.append(normalizedrmsedata)
        r2values = r2values.reshape(1, np.size(r2values))
        allrsquareddata.append(r2values)

        # Store the information from the run in an output .txt file
        outputstring = createstring(y_variable, rmse_values, best_rmse, best_coef, best_intercept, worst_rmse,
                                    worst_coef, worst_intercept, current_y_variable, xvariablenames, r2values, bestr2,
                                    worstr2)
        output_file.write(outputstring)

    # If make plots is true, create the side-by-side plots
    if makeplots:

        # RMSE plot
        rmseboxplot = plt.figure()
        rmsebp = rmseboxplot.add_subplot(1, 1, 1)
        rmsebp.boxplot(allrmsedata)
        rmsebp.set_title('Normalized RMSE Data for Each Y Variable')
        x = range(1, len(yvariablenames))
        plt.xticks(x, yvariablenames[1:len(yvariablenames)], rotation='vertical')
        rmsebp.set_xlabel('Y Variables')
        rmsebp.set_ylabel('RMSE / std')
        plt.tight_layout()

        # R^2 plot
        rsquaredboxplot = plt.figure()
        rsquaredbp = rsquaredboxplot.add_subplot(1, 1, 1)
        rsquaredbp.boxplot(allrsquareddata)
        rsquaredbp.set_title('R^2 Values for Each Y Variable')
        x = range(1, len(yvariablenames))
        plt.xticks(x, yvariablenames[1:len(yvariablenames)], rotation='vertical')
        rsquaredbp.set_xlabel('Y Variables')
        rsquaredbp.set_ylabel('R^2 Value')
        plt.tight_layout()

    output_file.close()
    plt.show()

# Run the script:
if __name__ == '__main__':
    main()
