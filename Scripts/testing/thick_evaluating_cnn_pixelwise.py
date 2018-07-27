""" Evaluating model trained on thickness data """

import numpy as np
import tensorflow as tf
import keras
from cnn_functions import create_input_thickness
from cnn_functions import create_input_with_predictions
from cnn_functions import graph_pred_truth

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imageSize = 21
numForecast = 6
resolution = 100

regBool = 1

#Load model for evaluation
model = keras.models.load_model('my_modelthick.h5')

#Get test data to retrieve predictions for month+1
year  = 2013
month = 2 #June

data, _, size = create_input_thickness(month, year, numForecast, imageSize, resolution, regBool)
data   = np.float32(data)

month_1_pred = model.predict(data)
dim = int(np.sqrt(size))
month_1_pred = month_1_pred[:,0]
month_1_pred = np.reshape(month_1_pred, (dim, dim))
graph_pred_truth(month_1_pred, month+1, year, resolution) #July

'''
#Get test data to retrieve predictions for month+2
month_2 = month+1
monthList = [month_1_pred]
data, _, _ = create_input_with_predictions(monthList, month_2, year, numForecast, imageSize, resolution, regBool)
data = np.float32(data)

month_2_pred = model.predict(data)
month_2_pred = np.reshape(month_2_pred, (dim, dim))
graph_pred_truth(month_2_pred, month_2+1, year, resolution) #August

#Get test data to retrieve predictions for month+3
month_3 = month_2+1
monthList = [month_1_pred, month_2_pred]
data, _, _ = create_input_with_predictions(monthList, month_3, year, numForecast, imageSize, resolution, regBool)
data = np.float32(data)

month_3_pred = model.predict(data)
month_3_pred = np.reshape(month_3_pred, (dim, dim))
graph_pred_truth(month_3_pred, month_3+1, year, resolution) #September
'''