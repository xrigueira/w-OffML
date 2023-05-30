
"""This custom metric attempts to compare the prediction
when the labels are only non-zero. This is interesting
because of the imabalanced nature of the data with far
more zeros than non-zero labels."""

import numpy as np

# Load the data
y_test = np.load('y_test.npy')
y_hat = np.load('y_hat.npy').flatten()

# Compute the overall distance including all zeros
## Use absolute difference
distance_abs = np.abs(y_test - y_hat)
print('Average global absolute distance', np.mean(distance_abs))

## Use Euclidean distance
distance_Euclid = np.linalg.norm(y_test - y_hat)
print('Average global Euclidean distance', np.mean(distance_Euclid))

# Find the indices of non-zero values in y_test
nonzero_indices = np.nonzero(y_test)[0]

# Compute the distance between non-zero values of y_test and the corresponding y_hat values
## Use absolute difference
distance_abs = np.abs(y_test[nonzero_indices] - y_hat[nonzero_indices])
print('Average absolute distance', np.mean(distance_abs))

## Use Euclidean distance
distance_Euclid = np.linalg.norm(y_test[nonzero_indices] - y_hat[nonzero_indices])
print('Average Euclidean distance', np.mean(distance_Euclid))

# Compute the R-squared metric
from sklearn.metrics import r2_score
r2_score_value = r2_score(y_test, y_hat)
print('Global R-squared measure', r2_score_value)

# Compute the R-squared metric with non-zero values of y_test and the corresponding y_hat values
r2_score_value = r2_score(y_test[nonzero_indices], y_hat[nonzero_indices])
print('R-squared measure', r2_score_value)

# Plot the labels and the prediction
import matplotlib.pyplot as plt

x_values = np.arange(0, len(y_test), 1)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(y_test[nonzero_indices])
ax.plot(y_hat[nonzero_indices])

plt.show()
