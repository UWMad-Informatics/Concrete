import numpy as np

# Number of folds
k = 10
# Fake data
# Note: Access a column: data[:,X])
x_vars = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])

# Determine the number of rows that will be used in each fold (# folds/# data points in 1 x variable)
# For example, if you're doing 10-fold CV with 100 instances per variable, there will be 10 rows per fold.
rows_add = int(k/len(x_vars[0]))
# Create train and test to hold data later
train = np.ndarray(shape=(len(x_vars[0]) - rows_add, len(x_vars[:, 0])))
test = np.ndarray(shape=(rows_add, len(x_vars[:, 0])))

# Use different folds for train and test each run.
for i in range(0, k - 1):
    if i != k - 1:
        test = x_vars[i:i+rows_add, :]
    else:
        test = x_vars[i:, :]

    if i != 0:
        train = x_vars[:i, :]
        train = np.concatenate((train, x_vars[(i+1):, :]))
    else:
        train = x_vars[(i+1):, :]
