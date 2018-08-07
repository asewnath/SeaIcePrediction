"""
Combined CNN LSTM Model for Spatial Map Forecasting
Author: Akira Sewnath
Date: 7/27/18
"""
import sys
sys.path.append('../')
from model_functions import create_dataset
from model_functions import shuffle_input

import numpy as np
import keras
import os

checkpoint_path = "cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Control Panel:
imDim = 7
startYear = 1985
stopYear  = 2014
batchSize = 300
startMonth = 4
stopMonth  = 8
numChannels = 4
resolution  = 100
numTimeSeries = 3

units = numTimeSeries

def create_model():
    
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', data_format="channels_first",
                     padding='valid',
                     input_shape=(numChannels, imDim, imDim))) 
    cnn.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                  data_format="channels_first"))
    cnn.add(keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same', 
                                  data_format="channels_first"))
    cnn.add(keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same', 
                                  data_format="channels_first"))       
    #cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid',
    #                                    data_format="channels_first"))
    cnn.add(keras.layers.Flatten())
    
    model = keras.models.Sequential()
    model.add(keras.layers.TimeDistributed(cnn, input_shape=(numTimeSeries, numChannels, imDim, imDim)))
    model.add(keras.layers.LSTM(units, activation='relu', recurrent_activation='hard_sigmoid', 
                                                  use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                                                  bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                                                  recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                                                  kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.2, 
                                                  recurrent_dropout=0.2, implementation=1, return_sequences=False, return_state=False, 
                                                  go_backwards=False, stateful=False, unroll=False))
    
    model.add(keras.layers.Dense(150, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation='linear'))
    
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=keras.optimizers.Adamax(),
                  loss='mse',      
                  metrics=['mae'])  # mean absolute error
  
    return model


#WARNING: Dataset creation is time consuming. Run once and save to file
feat, gt, size = create_dataset(startYear, stopYear, startMonth, stopMonth, resolution, 
                   numTimeSeries, imDim)

#feat = np.load("cnn_lstm_feat.npy")
#gt = np.load("cnn_lstm_gt.npy")

feat, gt = shuffle_input(feat, gt)


feat   = np.reshape(feat, (size, numTimeSeries, numChannels, imDim, imDim))
feat   = np.float32(feat)
labels = np.reshape(gt, (size, 2))

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False,
                                                         save_weights_only=False,verbose=1)

model = create_model()

model.fit(feat, labels, batch_size=batchSize, verbose=1, steps_per_epoch=None)
model.save('my_model.h5')


