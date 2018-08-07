"""
Evaluate CNN-LSTM Model
Author: Akira Sewnath
Date: 7/30/18
    
"""

import sys
sys.path.append('../')
import numpy as np
import keras
import model_functions as mfunc

def get_prediction_grid(outputMonth, dim):
    
    """
    Purpose: Reformats the prediction sea ice concentration grid for use, 
             includes masking for incorrect predictions
    """
    
    predMonth = outputMonth[:,0]
    zeroMask = np.where(predMonth < 0.006)
    predMonth[zeroMask] = 0
    predMonth = np.reshape(predMonth, (dim, dim))
    
    return predMonth
    
def get_feature_grids(outputMonth):
    
    """
    Purpose: Reformats the feature grids for use, 
             includes masking for incorrect predictions
    """
    
    iceConc  = outputMonth[:,0]
    zeroMaskConc = np.where(iceConc < 0.006)
    iceConc[zeroMaskConc] = 0
    iceConc = np.reshape(iceConc, (dim, dim)) 
    
    iceThick = outputMonth[:,1]
    zeroMaskThick = np.where(iceThick < 0.006)
    iceThick[zeroMaskThick] = 0
    iceThick = np.reshape(iceThick, (dim, dim))
    
    return iceConc, iceThick
 

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imDim = 7
numTimeSeries = 3
resolution = 100
numChannels = 4

#Load model for evaluation
model = keras.models.load_model('cnn_lstm_m5.h5')

#Get test data to retrieve predictions for month+1
year  = 2015
month = 5 #June
fMonth = month+3

data, _, size = mfunc.create_dataset(year, year, month, month, resolution, 
                   numTimeSeries, imDim)
data = np.float32(data)
data = np.reshape(data, (size, numTimeSeries, numChannels, imDim, imDim))
outputMonth1 = model.predict(data)
dim = int(np.sqrt(size))    

predMonth1 = get_prediction_grid(outputMonth1, dim)    
mfunc.graph_pred_truth(predMonth1, month+1, year, resolution) #July

iceConc, iceThick = get_feature_grids(outputMonth1)
procMonth1 = np.stack((iceConc, iceThick))


#Get test data to retrieve predictions for month+2
month2 = month+1
monthList = [procMonth1]
data, _ = mfunc.pred_create_dataset(monthList, month2, year, resolution, numTimeSeries, imDim)
data = np.float32(data)
outputMonth2 = model.predict(data)

predMonth2 = get_prediction_grid(outputMonth2, dim)
mfunc.graph_pred_truth(predMonth2, month2+1, year, resolution) #August

iceConc, iceThick = get_feature_grids(outputMonth2)
procMonth2 = np.stack((iceConc, iceThick))


#Get test data to retrieve predictions for month+3
month3 = month+2
monthList.append(procMonth2)
data, _ = mfunc.pred_create_dataset(monthList, month3, year, resolution, numTimeSeries, imDim)
data = np.float32(data)

outputMonth3 = model.predict(data)
predMonth3 = get_prediction_grid(outputMonth3, dim)
mfunc.graph_pred_truth(predMonth3, month3+1, year, resolution) #September





#Code for looking at linear spatial maps and RMSE
startYear = 1985
stopYear = 2014
linearGrid = mfunc.pred_lls_grid(month, fMonth, startYear, stopYear, resolution)
#graph_pred_truth(linearGrid, fMonth, year, resolution) #September

groundTruth = mfunc.get_conc_grid(fMonth, year, resolution)

#Root mean square error of the ice areas (assuming each grid cell is 100 km)
cnnArea = predMonth3 * (resolution**2)
linArea = linearGrid * (resolution**2)
gtArea  = groundTruth * (resolution**2)

rmseCnn = np.sqrt((np.sum(cnnArea-gtArea)**2)/(dim**2))
rmseLin = np.sqrt((np.sum(linArea-gtArea)**2)/(dim**2))



