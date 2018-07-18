"""CNN using Keras with Tensorflow backend"""

from cnn_functions import create_input
from cnn_functions import shuffle_input
import numpy as np
import tensorflow as tf #Using Tensorflow backend
import keras
import os

#Control Panel:
startYear = 1985
stopYear  = 2012
#May want to increase the imageSize to allow for more layers
imageSize = 21
numForecast = 6
numChannels = numForecast+1
resolution = 50
batchSize = 400


def create_model():
    
    #Maybe add dropout
    #I think another conv layer may help
    #probably want to increase kernel size to use that spatial information before pooling
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu', data_format="channels_first",
                     padding='valid',
                     input_shape=(numChannels, imageSize, imageSize)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                        data_format="channels_first", padding='valid'))
    model.add(keras.layers.Conv2D(64, (5, 5), activation='relu', padding='valid', #(3,3)
                                  data_format="channels_first"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid',
                                        data_format="channels_first"))
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
                                  data_format="channels_first"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                        data_format="channels_first"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1, activation='linear'))
    
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=keras.optimizers.Adadelta(0.001),
                  loss='logcosh',      
                  metrics=['mae'])  # mean absolute error
  
    return model


checkpoint_path = "cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

months = np.arange(0,12)
for year in range(startYear, stopYear+1):
    #make a randomized list of months and train with that every year to avoid cyclical training behavior
    months = np.random.permutation(months)
    
    for index in range(0, 12):

        data, groundTruth, size = create_input(months[index], year, numForecast, imageSize, resolution)
        data, labels = shuffle_input(data, groundTruth)
                
        data   = np.reshape(data, (size, numChannels, imageSize, imageSize))
        labels = np.reshape(labels, (size, 1))
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
            model = keras.models.load_model('my_modeltest.h5')
        
        print("Year:" + str(year) + ",  Month:" + str(months[index]))
        #increase batch size to 200
        model.fit(data, labels, batch_size=batchSize, verbose=2,
                  steps_per_epoch=None, epochs=1)

        model.save('my_modeltest.h5')





