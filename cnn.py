# Deploy a simple CNN for classification and prediction

import keras
import tensorflow as tf

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
    import pandas as pd
    data = pd.read_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8')

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data.iloc[:, 1:-1] = scaler.fit_transform(data.iloc[:, 1:-1])

    # Convert variable columns to np.ndarray
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # Split the data into test and train sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def cnn(X_train, X_test, y_train, y_test):

    # Set random seed
    tf.random.set_seed(0)

    # Define the model
    model = keras.models.Sequential([
        keras.layers.Conv1D(32, kernel_size=3, strides=1, input_shape=(6, 1), activation='relu'),
        keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Get model's summary
    model.summary()
    
    # Reshape data to satisfy (batch_size, sequence_length, num_features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=4, batch_size=32)

    # # Test the performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Loss: %.2f, Accuracy: %.2f' % (loss*100, accuracy*100))
    
    # Perform prediction and get confusion matrix
    y_hat = model.predict(X_test)
    print(tf.math.confusion_matrix(y_test, y_hat))

if __name__ == '__main__':
    
    station = 901

    X_train, X_test, y_train, y_test = reader(station=station)

    cnn(X_train, X_test, y_train, y_test)

    # Implement mini-batching
    
    