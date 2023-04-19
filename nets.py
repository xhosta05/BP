import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Input, Activation, MaxPool1D

from keras.optimizers import SGD

def Baseline_Model(input_dim, num_classes, OPT = 'adam', metrics=['accuracy'], loss = 'categorical_crossentropy'):
  # Create model
  The_Model = Sequential()
  The_Model.add(Dense(32, input_dim = input_dim, activation = 'relu'))
  The_Model.add(Dense(num_classes, activation = 'softmax'))
  # Compile model
  The_Model.compile(loss = 'categorical_crossentropy', optimizer = OPT, metrics=['accuracy'])
  return The_Model

def Conv_Model_1l(input_shape, num_classes, OPT = 'adam', time_steps = 1, metrics=['accuracy'], loss = 'categorical_crossentropy'):
  # OPT =keras.optimizers.Adam( learning_rate=0.001, weight_decay=True)
  # OPT = SGD(learning_rate = 0.01, momentum = 0.9)
  
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(input_shape), name="input"))

  model.add(layers.RepeatVector(time_steps))
  model.add(Dropout(0.5))
  model.add(Conv1D(64, 2, activation="relu", padding="same"))

  model.add(Dropout(0.5))  
  model.add(Dense(num_classes, activation = 'softmax'))
  
  model.add(Flatten())

  model.compile(loss = loss, optimizer = OPT, metrics = metrics)
  return model

