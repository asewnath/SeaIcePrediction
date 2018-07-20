"""CNN using Keras with Tensorflow backend"""

#from cnn_functions import create_input
from cnn_functions import shuffle_input
from exp_functions import exp_create_input

import numpy as np
import tensorflow as tf #Using Tensorflow backend
import keras
import os

#Control Panel:
startYear = 1985
stopYear  = 2014
#May want to increase the imageSize to allow for more layers
imageSize = 5
numForecast = 2
numChannels = 8
resolution = 100
batchSize = 400
regBool = 1



def create_model():
    
    #Maybe add dropout
    #I think another conv layer may help
    #probably want to increase kernel size to use that spatial information before pooling
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', data_format="channels_first",
                     padding='valid',
                     input_shape=(numChannels, imageSize, imageSize)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), d'))
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same', 
                                  data_format="channels_first"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid',
                                        data_format="channels_first"))
    #model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
    #                              data_format="channels_first"))
    #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same',
    #                                    data_format="channels_first"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation='linear'))
    
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=keras.optimizers.Adagrad(),
                  loss='logcosh',      
                  metrics=['mae'])  # mean absolute error
  
    return model


checkpoint_path = "cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

startMonth = 5
stopMonth  = 9

months = np.arange(startMonth,stopMonth)
for year in range(startYear, stopYear+1):
    #make a randomized list of months and train with that every year to avoid cyclical training behavior
    months = np.random.permutation(months)
    
    for index in range(np.size(months)):

        data, groundTruth, size = exp_create_input(months[index], year, numForecast, imageSize, resolution)
        data, labels = shuffle_input(data, groundTruth)
                
        data   = np.reshape(data, (size, numChannels, imageSize, imageSize))
        labels = np.reshape(labels, (size, 2)) #concentration and thickness
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
            model = keras.models.load_model('model71918_1month.h5')
        
        print("Year:" + str(year) + ",  Month:" + str(months[index]))
        #increase batch size to 200
        model.fit(data, labels, batch_size=batchSize, verbose=2,
                  steps_per_epoch=None, epochs=1)

        model.save('model71918_1month.h5')





