"""Focusing on evaluating 3 months prediction"""

import numpy as np
import tensorflow as tf
import keras
from cnn_functions import create_input
from cnn_functions import create_input_with_predictions 
from cnn_functions import graph_pred_truth
from exp_functions import exp_create_input
from cnn_functions import retrieve_grid
from exp_functions import border_grid

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imageSize = 5
numForecast = 2
resolution = 100

regBool = 1

#Load model for evaluation
model = keras.models.load_model('model71918_2month.h5')

#Get test data to retrieve predictions for month+1
year  = 2015
month = 8 #June

data, _, size = exp_create_input(month, year, numForecast, imageSize, resolution)
data   = np.float32(data)

month_1_pred = model.predict(data)
#Get sea ice concentration
month_1_pred = month_1_pred[:,0]
dim = int(np.sqrt(size))
month_1_pred = np.reshape(month_1_pred, (dim, dim))
#border with zeroes to see it
month_1_pred = border_grid(month_1_pred, 2)
graph_pred_truth(month_1_pred, month+2, year, resolution) #July
#grid = retrieve_grid(month+1, year, resolution)


