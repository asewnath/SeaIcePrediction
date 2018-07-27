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
from exp_functions import pred_exp_create_input
from exp_functions import get_detrend_grid

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imageSize = 7
numForecast = 2
resolution = 100

regBool = 1

#Load model for evaluation
model = keras.models.load_model('model72318_pixpos.h5')

#Get test data to retrieve predictions for month+1
year  = 2015
month = 7 #June
fMonth = month+1

data, _, size = exp_create_input(month, year, numForecast, imageSize, resolution)
data   = np.float32(data)

outputMonth1 = model.predict(data)

#Get sea ice concentration
predMonth1 = outputMonth1[:,0]
dim = int(np.sqrt(size))
predMonth1 = np.reshape(predMonth1, (dim, dim))
graph_pred_truth(predMonth1, fMonth, year, resolution) #July

startYear = 1985
stopYear = 2014
detrendGrid = get_detrend_grid(month, startYear, stopYear, resolution)
graph_pred_truth(detrendGrid, fMonth, year, resolution) #July


groundTruth = retrieve_grid(fMonth, year, resolution)

#Root mean square error of the ice areas (assuming each grid cell is 100 km)
#cnnError = np.sum(np.absolute(groundTruth-predMonth1)) / (57**2)
#linError = np.sum(np.absolute(groundTruth-detrendGrid)) / (57**2)


cnnArea = predMonth1 * 100
linArea = detrendGrid * 100
gtArea  = groundTruth * 100

rmseCnn = np.sqrt((np.sum(cnnArea-gtArea)**2)/(57**2))
rmseLin = np.sqrt((np.sum(linArea-gtArea)**2)/(57**2))


#Calculate ice extents
#cnnNoIce = np.where(predMonth1 >=0.15)
#cnnIceExtent = predMonth


'''
iceConc  = outputMonth1[:,0]
iceConc = np.reshape(iceConc, (dim, dim))
iceThick = outputMonth1[:,1]
iceThick = np.reshape(iceThick, (dim, dim))

procMonth1 = np.stack((iceConc, iceThick))
'''
'''
#Get test data to retrieve predictions for month+2
month2 = month+1
monthList = [procMonth1]
data, _, _ = pred_exp_create_input(monthList, month2, year, numForecast, imageSize, resolution)
data = np.float32(data)

outputMonth2 = model.predict(data)
predMonth2 = outputMonth2[:,0]
predMonth2 = np.reshape(predMonth2, (dim, dim))
graph_pred_truth(predMonth2, month2+1, year, resolution) #August
'''







