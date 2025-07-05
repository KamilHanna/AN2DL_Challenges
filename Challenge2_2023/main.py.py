from tqdm import tqdm
import cython, scipy, sklearn, skimage, yaml, imutils, cv2, psutil, h5py
import os

# Fix randomness and hide warnings
seed = 42
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
os.environ["PATH"] += os.pathsep + ('C:\Program Files\Graphviz\bin')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np

np.random.seed(seed)

import logging

import random

random.seed(seed)

# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Conv1D, MaxPooling1D, Flatten, Bidirectional
from keras.utils import plot_model
from keras import layers
from keras.optimizers import RMSprop

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
print("TensorFlow version:", tf.__version__)

import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''#########################################################################################'''
'''#################################### Data loading part:##################################'''

# Load the data
training_data = np.load('training_data.npy')
categories = np.load('categories.npy')
valid_periods = np.load('valid_periods.npy')

# Print the result
print("Training_data shape  :", training_data.shape, "type :",type(training_data))  # indexed rows range from 0 to 47999, COLS FROM 0 to 2275.
print("Categories shape     :", categories.shape, "type : ", type(categories))  # indexed rows range from 0 to 47999.
print("Valid_periods shape :", valid_periods.shape, "type :",type(valid_periods))  # indexed rows range from 0 to 47999, COLS FROM 0 to 1.
# Maybe : 9518411 (number of all elements of the 48000 time series...)

'''#########################################################################################'''
'''############################# Retrieve original series ##################################'''

# num_rows, num_cols = training_data.shape
# Initialize the original series; with 48000 rows... & a variable number of columns.
original_series = [[] for _ in range(48000)]
cont=0
# Here we are extracting the original series from the training_data without the bounding zeros.
for i in range(0, 48000):
    start_index = valid_periods[i][0]
    end_index = valid_periods[i][1]
    for j in range(start_index, end_index):
        original_series[i].append(training_data[i, j])
#original_series=original_series[:i]
zeta = len(original_series)
print(zeta)

'''#########################################################################################'''
'''############################# Creating data set array ###################################'''

# Create a series with 48000 rows & 218 columns... filled with zeros.
# Inserting i's index if the series is smaller than 218.
for i in range(0, cont):
    while (len(original_series[i]) < 218):
        original_series[i].insert(0, original_series[i,0])

# Specify the target length (218 in our case)
target_length = 218
new_series = []

# Iterate over each sublist and remove the last elements if needed
for i in range(len(original_series)):
    if len(original_series[i]) >= target_length:
        # Iterate in steps of target_length
        for j in range(0, len(original_series[i]), target_length):
            new_sublist = original_series[i][j : j + target_length]
            while len(new_sublist) < target_length:
                new_sublist.insert(0, 0 )
            new_series.append(new_sublist)
  #  else:
   #     while len(original_series[i]) < target_length:
    #        original_series[i].insert(0, 0)
     #   new_series.append(original_series[i])

# Check the length of the new series
cont = len(new_series)

original_series = np.array(new_series)

x_series = np.array(new_series)

np.round(x_series, decimals=3)

original_series = np.concatenate((original_series, x_series,x_series,x_series,x_series), axis=0)

print("original series shape" , original_series.shape)

# Select the first 200 columns
final_series = original_series[:, :200]

# Select the last 18 columns
val_set = original_series[:, -18:]

# Now you have two separate arrays

final_series = np.array(final_series)
print("final_series shape :", final_series.shape, "type : ", type(final_series))
print("val_set shape :", val_set.shape, "type : ", type(val_set))

'''#########################################################################################'''
'''################################## Splitting dataset ####################################'''

# 20% for testing, 80% for training.
test_size = 0.2
# Train_data : X_train
# Test_data :  X_test

# Split the data into training and testing sets without shuffling
# (since we are working with time series forecasting here, the order matters!!
X_train, X_test = train_test_split(final_series, test_size=test_size, shuffle=False, random_state=42)
Y_val, Z_val = train_test_split(val_set, test_size=0.2, shuffle=False, random_state=42)

print("W Shape :", X_train.shape, "W type :", type(X_train))
print("Z Shape :", X_test.shape, "Z type :", type(X_test))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Example data (replace this with your own data)
# Assume you have a 2D numpy array X with shape (num_samples, time_steps, num_features)
# and a 2D numpy array y with shape (num_samples, output_dim)
num_samples = cont
time_steps = 200
num_features = 1
output_dim = 18

y = Y_val
print("y Shape :", y.shape, "y type :", type(y))

# Build the LSTM model
model = Sequential()

# layer to ignore the padded 0's
model.add(Masking(mask_value=0, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(layers.Dense(64, activation='relu'))
model.add(Dense(units=output_dim))

# Compile the model
model.compile(optimizer='adam', loss='mse')  # You can use other optimizers and loss functions as needed

# Print the model summary
model.summary()

# Train the model
model.fit(X_train,y, epochs=8, batch_size=32, validation_split=0.2)  # Adjust the number of epochs and batch size

# Make predictions (replace this with your test data)
predictions = model.predict(X_test)

# Print the predictions
print(predictions)

# Variable that holds directory name.
save_model_dir = 'savemymodel'
# Create the directory if it doesn't exist
os.makedirs(save_model_dir, exist_ok=True)

model.save(os.path.join(save_model_dir, 'SubmissionModel'))
print("Model saved to:", save_model_dir)

# print(Z_val.shape, predictions.shape)
# Assuming 'actual_values' is your true target values for the test set

# Calculate metrics
mse = mean_squared_error(Z_val, predictions)
mae = mean_absolute_error(Z_val, predictions)
r2 = r2_score(Z_val, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"r2_score: {r2}")
