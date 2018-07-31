"""
Evaluate CNN-LSTM Model
Author: Akira Sewnath
Date: 7/30/19
    
"""

import numpy as np
import keras
from cnn_functions import graph_pred_truth
from cnn_functions import retrieve_grid
from exp_functions import get_detrend_grid
from exp_functions import pred_detrend_grid

from cnn_lstm_functions import create_dataset
from cnn_lstm_functions import pred_create_dataset

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imDim = 7
numTimeSeries = 3
resolution = 100

#Load model for evaluation
model = keras.models.load_model('cnn_lstm_m4.h5')

#Get test data to retrieve predictions for month+1
year  = 2015
month = 5 #June
fMonth = month+3


data, _ = create_dataset(year, year, month, month, resolution, 
                   numTimeSeries, imDim)
data   = np.float32(data)
data = np.reshape(data, (3249, 3, 4, 7, 7))
outputMonth1 = model.predict(data)

#Get sea ice concentration
predMonth1 = outputMonth1[:,0]
dim = 57#int(np.sqrt(size))
predMonth1 = np.reshape(predMonth1, (dim, dim))
graph_pred_truth(predMonth1, month+1, year, resolution) #July


iceConc  = outputMonth1[:,0]
iceConc = np.reshape(iceConc, (dim, dim))
iceThick = outputMonth1[:,1]
iceThick = np.reshape(iceThick, (dim, dim))

procMonth1 = np.stack((iceConc, iceThick))

'''
#Get test data to retrieve predictions for month+2
month2 = month+1
monthList = [procMonth1]
data, _ = pred_create_dataset(monthList, month2, year, resolution, numTimeSeries, imDim)
data = np.float32(data)

outputMonth2 = model.predict(data)
predMonth2 = outputMonth2[:,0]
predMonth2 = np.reshape(predMonth2, (dim, dim))
graph_pred_truth(predMonth2, month2+1, year, resolution) #August

startYear = 1985
stopYear = 2014
#detrendGrid = get_detrend_grid(fMonth, startYear, stopYear, resolution)
#detrendGrid = pred_detrend_grid(month, fMonth, startYear, stopYear, resolution)
#graph_pred_truth(detrendGrid, fMonth, year, resolution) #July

#groundTruth = retrieve_grid(fMonth, year, resolution)



iceConc  = outputMonth1[:,0]
iceConc = np.reshape(iceConc, (dim, dim))
iceThick = outputMonth1[:,1]
iceThick = np.reshape(iceThick, (dim, dim))

procMonth2 = np.stack((iceConc, iceThick))


#Get test data to retrieve predictions for month+2
month3 = month+2
monthList.append(procMonth2)
data, _ = pred_create_dataset(monthList, month3, year, resolution, numTimeSeries, imDim)
data = np.float32(data)

outputMonth3 = model.predict(data)
predMonth3 = outputMonth2[:,0]
predMonth3 = np.reshape(predMonth2, (dim, dim))
graph_pred_truth(predMonth3, month3+1, year, resolution) #September


startYear = 1985
stopYear = 2014
#detrendGrid = get_detrend_grid(fMonth, startYear, stopYear, resolution)
detrendGrid = pred_detrend_grid(month, fMonth, startYear, stopYear, resolution)
graph_pred_truth(detrendGrid, fMonth, year, resolution) #September

groundTruth = retrieve_grid(fMonth, year, resolution)

#Root mean square error of the ice areas (assuming each grid cell is 100 km)
cnnArea = predMonth3 * 100
linArea = detrendGrid * 100
gtArea  = groundTruth * 100

rmseCnn = np.sqrt((np.sum(cnnArea-gtArea)**2)/(57**2))
rmseLin = np.sqrt((np.sum(linArea-gtArea)**2)/(57**2))
'''


