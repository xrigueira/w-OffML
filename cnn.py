# Deploy a simple NN for classification and prediction

import keras
import tensorflow as tf

# Make sure this works
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Read the data as in the other examples and convert into tensors
# Perform minmax scaling

# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/02_neural_network_classification_in_tensorflow.ipynb

# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/03_convolutional_neural_networks_in_tensorflow.ipynb

# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb

# Check ChatGPT answer

def reader(self):
    """This method read the data and splits it in training and testing sets.
    ----------
    Arguments:
    None.
    
    Returns:
    X_train: (np.array): variables train set.
    X_test: (np.array): variables test set.
    y_train: (np.array): target train set.
    y_test: (np.array): target test set."""
    
    # Read the data
    data = pd.read_csv(f'data/labeled_{self.station}_cle copy.csv', sep=',', encoding='utf-8')

    # Convert variable columns to np.ndarray
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # Split the data into test and train sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)


