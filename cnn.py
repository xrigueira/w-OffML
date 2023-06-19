# Deploy a simple CNN for classification and prediction

import numpy as np
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import keras
import tensorflow as tf

class CNN():
    
    def __init__(self, station, group_size, step_size, learning_rate, num_epochs, tune_lr) -> None:
        self.station = station
        self.group_size = group_size
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.tune_lr = tune_lr

    def custom_train_test_split(self, X, y, train_size):
        """This function split the data at the 'train size' length. The test set 
        would be all items which index is smaller then this number, while the test 
        size all those items with an index above.
        ----------
        Arguments:
        X (np.array): predictor variables.
        y (np.array): labels or target variable.
        train_size (float): defines the size of the train and test sets.
        
        Return:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set."""

        # Define train sets
        X_train = X[:int(train_size*len(X))]
        y_train = y[:int(train_size*len(y))]

        # Define test sets
        X_test = X[int(train_size*len(X)):]
        y_test = y[int(train_size*len(y)):]

        return X_train, X_test, y_train, y_test

    def reader(self):
        """This method reads the data and splits it in training and testing sets.
        ----------
        Arguments:
        station (int): the station number.

        Returns:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set."""

        # Read the data
        data = pd.read_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8')

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
        X_train, X_test, y_train, y_test = self.custom_train_test_split(X, y, train_size=0.8)

        return X_train, X_test, y_train, y_test, features

    def windows(self, array, data_type):
        """This method takes an array and generates overlapping windows.
        ----------
        Arguments:
        array (np.array): the array to process.
        group_size (int): the length of each window.
        step_size (int): the step used to create the windows.
        data_type (string): whether it is data or labels.

        Returns:
        groups (np.array): windowed array."""

        groups = []
        for i in range(0, array.shape[0] - self.group_size + 1, self.step_size):
            group = array[i:i + self.group_size]
            if data_type == 'X':
                groups.append(group)
            else:
                label = sum(group) / len(group)
                if label >= 0.5:
                    label = 1
                else:
                    label = 0
                groups.append(label)
        
        return np.array(groups)
    
    def class_weights(self, array):
        
        neg = np.count_nonzero(array==0)
        pos = np.count_nonzero(array==1)
        total = neg + pos
        
        weight_for_0 = (1 / neg) * (total / 2)
        weight_for_1 = (1 / pos) * (total / 2)
        
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        return class_weight
    
    def plot_metrics(self, history):
        
        metrics = ['loss', 'prc', 'precision', 'recall']
        
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                    color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])

    plt.legend()
    def cnn(self, X_train, X_test, y_train, y_test, features, class_weight):
        
        # Set random seed
        tf.random.set_seed(0)
        
        # Define the model
        # input_sahpe = (samples, time steps in each samples, feautres)
        model = keras.models.Sequential([
            keras.layers.Conv1D(32, kernel_size=1, strides=1, input_shape=(self.group_size, features), activation='relu'),
            keras.layers.Conv1D(64, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(128, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(256, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(512, kernel_size=1, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            # keras.layers.Conv1D(512, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(1024, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(1014, kernel_size=1, activation='relu'),
            # keras.layers.Conv1D(512, kernel_size=1, activation='relu'),
            # keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            # keras.layers.Dense(512, activation='relu'),
            # keras.layers.Dense(256, activation='relu'),
            # keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Get model's summary
        model.summary()
        
        # Define metrics
        metrics = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=keras.losses.BinaryCrossentropy(), metrics=metrics)

        # Train the model
        if self.tune_lr == True:
            print("Tunning learning rate")
            # Create a learning rate scheduler callback
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
            history = model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32, callbacks=[lr_scheduler], verbose=0)
        else:
            history = model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32, class_weight=class_weight, verbose=0)

        # Test the performance
        results = model.evaluate(X_test, y_test, verbose=1)
        for name, value in zip(model.metrics_names, results):
            print(name, ': ', value)
        print()
        
        # Plot metrics
        # self.plot_metrics(model)
        
        # Perform prediction and get confusion matrix
        y_hat = model.predict(X_test)
        
        # Plot the loss and accuracy curves
        pd.DataFrame(history.history).plot(figsize=(10, 7))
        plt.title('Learning curves')
        plt.show()
        
        if self.tune_lr == True:
            print("Tunning learning rate")
            lrs = 1e-4 * (10**(np.arange(self.num_epochs)/20))
            plt.figure(figsize=(10, 7))
            plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning rate vs. loss")
            plt.show()
        
        return y_hat
    
    def custom_metric(self, y_test, y_hat):
        """This custom metric attempts to compare the prediction
        when the labels are only non-zero. This is interesting
        because of the imabalanced nature of the data with far
        more zeros than non-zero labels."""
        
        # Flatten y_hat
        y_hat = y_hat.flatten()
        
        # Compute the overall distance including all zeros
        ## Use absolute difference
        distance_abs = np.abs(y_test - y_hat)
        print('Average global absolute distance', np.mean(distance_abs))

        # Find the indices of non-zero values in y_test
        nonzero_indices = np.nonzero(y_test)[0]
        
        # Compute the R-squared metric
        from sklearn.metrics import r2_score
        r2_score_value = r2_score(y_test, y_hat)
        print('Global R-squared measure', r2_score_value)

        # Plot the labels and the prediction
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(y_test, label='test')
        ax.plot(y_hat, label='prediction')
        # ax.plot(y_test[nonzero_indices])
        # ax.plot(y_hat[nonzero_indices])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    
    station = 901
    group_size = 32
    
    # Create an instance of the CNN class
    cnn_model = CNN(station=station, group_size=group_size, step_size=1, learning_rate=0.001, num_epochs=100, tune_lr=False)

    # Read and split the data
    X_train, X_test, y_train, y_test, features = cnn_model.reader()

    # Preprocess the data to group in overlapping windows
    X_train = cnn_model.windows(array=X_train, data_type='X')
    X_test = cnn_model.windows(array=X_test, data_type='X')
    y_train = cnn_model.windows(array=y_train, data_type='y')
    y_test = cnn_model.windows(array=y_test, data_type='y')
    
    # Calculate the class weights
    class_weight = cnn_model.class_weights(y_train)
    
    y_hat = cnn_model.cnn(X_train, X_test, y_train, y_test, features, class_weight)
    
    cnn_model.custom_metric(y_test, y_hat)
