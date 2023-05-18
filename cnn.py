# Deploy a simple NN for classification and prediction

import keras
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
