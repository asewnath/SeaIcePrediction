"""
Month to Month Model Evaluation of CNN Pixelwise Regression
"""
import sys
sys.path.append('../')
import numpy as np
import keras
import model_functions as mfunc

#MAKE SURE THAT THESE MATCH THE PARAMETERS IN TRAINING
imageSize = 7
numForecast = 2
resolution = 100
regBool = 1

#Load model for evaluation
model = keras.models.load_model('model72318_pixpos.h5')

#Get test data to retrieve predictions for month+1
year  = 2015
month = 7
fMonth = month+1

data, _, size = mfunc.cnn_create_input(month, year, numForecast, imageSize, resolution)
data   = np.float32(data)

outputMonth1 = model.predict(data)

#Get sea ice concentration
predMonth1 = outputMonth1[:,0]
dim = int(np.sqrt(size))
predMonth1 = np.reshape(predMonth1, (dim, dim))
mfunc.graph_pred_truth(predMonth1, fMonth, year, resolution)

startYear = 1985
stopYear = 2014
linearGrid = mfunc.get_linear_grid(month, startYear, stopYear, resolution)
mfunc.graph_pred_truth(linearGrid, fMonth, year, resolution)


groundTruth = mfunc.get_conc_grid(fMonth, year, resolution)

#Root mean square error of the ice areas (assuming each grid cell is 100 km)
cnnArea = predMonth1 * 100
linArea = linearGrid * 100
gtArea  = groundTruth * 100

#57 hardcoded for a 7*7 image size
rmseCnn = np.sqrt((np.sum(cnnArea-gtArea)**2)/(57**2))
rmseLin = np.sqrt((np.sum(linArea-gtArea)**2)/(57**2))






