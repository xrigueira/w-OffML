# Deploy a simple CNN for classification and prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

# Resources
# https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57
# https://www.mlq.ai/time-series-with-tensorflow-cnn/
# https://www.macnica.co.jp/en/business/ai/blog/142046/
# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/02_neural_network_classification_in_tensorflow.ipynb
# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/03_convolutional_neural_networks_in_tensorflow.ipynb
# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb

def reader(station):
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
    data = pd.read_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8')

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data.iloc[:, 1:-1] = scaler.fit_transform(data.iloc[:, 1:-1])
    
    # Convert variable columns to np.ndarray
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    # Get number of variables
    features = X.shape[1]

    # Split the data into test and train sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test, features

def windows(X_train, X_test, y_train, y_test):
    
    # Reshape data to satisfy (batch_size, sequence_length, num_features):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test

def cnn(X_train, X_test, y_train, y_test, features, num_epochs, tune_lr):

    # TODO: Implement mini-batching
    
    # Set random seed
    tf.random.set_seed(0)
    
    # Define the model
    # input_sahpe = (samples, time steps in each samples, feautres)
    model = keras.models.Sequential([
        keras.layers.Conv1D(64, kernel_size=1, strides=1, input_shape=(6, 1), activation='relu'),
        keras.layers.Conv1D(128, kernel_size=1, activation='relu'),
        keras.layers.Conv1D(256, kernel_size=1, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Get model's summary
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0035), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Train the model
    if tune_lr == True:
        print("Tunning learning rate")
        # Create a learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, callbacks=[lr_scheduler])
    else:
        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)

    # Test the performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Loss: %.2f, Accuracy: %.2f' % (loss*100, accuracy*100))
    
    # Perform prediction and get confusion matrix
    y_hat = model.predict(X_test)
    print(tf.math.confusion_matrix(y_test, y_hat))
    
    # Plot the loss and accuracy curves
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.title('Learning curves')
    plt.show()
    
    if tune_lr == True:
        print("Tunning learning rate")
        lrs = 1e-4 * (10**(np.arange(num_epochs)/20))
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()

if __name__ == '__main__':
    
    station = 901

    X_train, X_test, y_train, y_test, features = reader(station=station)
    
    # cnn(X_train, X_test, y_train, y_test, features, num_epochs=50, tune_lr=False)
