"""
Combined CNN LSTM Model for Spatial Map Forecasting
Author: Akira Sewnath
Date: 7/27/18
"""

from cnn_lstm_functions import create_dataset
from cnn_lstm_functions import shuffle_input

import numpy as np
import keras
import os


checkpoint_path = "cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Control Panel:
imDim = 7
startYear = 1985
stopYear  = 2014
batchSize = 400
startMonth = 5
stopMonth  = 9
numChannels = 4
resolution  = 100
numTimeSeries = 3

units = numTimeSeries

def create_model():
    
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', data_format="channels_first",
                     padding='valid',
                     input_shape=(numChannels, imDim, imDim))) #check this shit
    cnn.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                  data_format="channels_first"))
    cnn.add(keras.layers.SeparableConv2D(128, (2, 2), activation='relu', padding='same', 
                                  data_format="channels_first"))
    cnn.add(keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same', 
                                  data_format="channels_first"))       
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid',
                                        data_format="channels_first"))
    cnn.add(keras.layers.Flatten())
    
    model = keras.models.Sequential()
    model.add(keras.layers.TimeDistributed(cnn)) #input shape here?
    model.add(keras.layers.LSTM(keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', 
                                                  use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                                                  bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                                                  recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                                                  kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
                                                  recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, 
                                                  go_backwards=False, stateful=False, unroll=False)))
    
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation='linear'))
    
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss='mse',      
                  metrics=['mae'])  # mean absolute error
  
    return model


#WARNING: Dataset creation is time consuming
feat, gt = create_dataset(startYear, stopYear, startMonth, stopMonth, resolution, 
                   numTimeSeries, imDim)

feat, gt = shuffle_input(feat, gt)

#feat   = np.reshape(feat, (size, numChannels, imDim, imDim))
#feat   = np.float32(feat)
#labels = np.reshape(labels, (size, 2))

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False,
                                                         save_weights_only=False,verbose=1)

model = create_model()

#Do gridcv shit.
model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)
model.fit(feat, gt, batch_size=batchSize, verbose=2,steps_per_epoch=None, epochs=1)
model.save('cnn_lstm_m1.h5')



'''
months = np.arange(startMonth,stopMonth)
for year in range(startYear, stopYear+1):

    months = np.random.permutation(months)
    
    for index in range(np.size(months)):

        data, groundTruth, size = create_cube(months[index], year, numForecast, imageSize, resolution)

                
        data   = np.reshape(data, (size, numChannels, imageSize, imageSize))
        labels = np.reshape(labels, (size, 2))
        data   = np.float32(data)

        # Create checkpoint callback
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                         save_best_only=False,
                                                         save_weights_only=False,
                                                         verbose=1)
        if((year == startYear) and (index == 0)):
            model = create_model()
            model.summary()
        else:
            model = keras.models.load_model('cnn_lstm_m1.h5')
        
        print("Year:" + str(year) + ",  Month:" + str(months[index]))
        model.fit(data, labels, batch_size=batchSize, verbose=2,
                  steps_per_epoch=None, epochs=1)

        model.save('cnn_lstm_m1.h5')
'''




