#Script to gather data for sequential model experimentation

"""
Data Required:
    Both models require an input vector of month number, regional sea ice
    concentrations, and average ice thickness.
    
    Month number may have to be normalized to match magnitude of other features

"""
import forecast_funcs as ff
import numpy as np
from pylab import array

def data_collection():

    rawDataPath = '../../Data/'
    derivedDataPath = '../../DataOutput/'
    forecastVar = 'conc'
    hemStr = 'N'
    
    startYr    = 1980
    forecastYr = 2012
    startMonth = 1 
    stopMonth  = 12 
    regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')
    
    numFeat = 18
    numSamples = (forecastYr - startYr) * (stopMonth+1 - startMonth)
    
    #data collection
    #start off with just the summer months and without ice thickness
    
    feat = []
    for year in range(startYr, forecastYr):
        for index, month in enumerate(range(startMonth, stopMonth+1)):
            
            #set month to array
            sample = np.zeros(numFeat)
            sample[0] = (month-1) #gridded months are actually indexed at 0
            #divide by 100 to put in the same magnitude as the other values
            sample[1] = np.ma.mean(ff.get_pmas_month(rawDataPath, year, month-1))
            
            #set region features to array
            iceConc = ff.get_gridvar(derivedDataPath, forecastVar, month, 
                              array(year), hemStr)
            
            
            for regInd in range(16):
                    
                regionData = iceConc.data
                regionData[iceConc.mask == True] = 0 #get rid of nan
                desiredRegion = regionMask == regInd
                sample[regInd+2] = np.ma.mean(np.multiply(regionData, desiredRegion))
                #sample[regInd+1] = np.ma.mean(np.multiply(regionData, desiredRegion))
                
            """        
            for regInd in range(numFeat-1):
                
                regionData = iceThick.data
                regionData[iceThick.mask == True] = 0 #get rid of nan
                desiredRegion = regionMask == regInd
                sample[numFeat+regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))                   
            """    
            feat.append(sample)
            
    feat = np.reshape(feat, (numSamples, numFeat))
    groundTruth = feat[1:np.size(feat,0), 1: np.size(feat,1)]  
    feat = feat[0:np.size(feat,0)-1, :]
        
    return feat, groundTruth



def mat_preprocess():
    #Treat like a pseudo time series
    feat, _ = data_collection() #get data
    
    data = []
    groundTruth = []
    for index in range(np.size(feat,0)-3):
        data.append(np.row_stack((feat[index], feat[index+1], feat[index+2])))
        groundTruth.append(feat[index+3])
    
    data = np.reshape(data, (380, 54))
    groundTruth = np.reshape(groundTruth, (380, 18))
    groundTruth = groundTruth[:, 1: np.size(groundTruth,1)]
    
    return data, groundTruth
    
    
    
    

#This will come later... (hopefully not)

#def lstm_preprocessing():
feat, _ = data_collection() #get data

#Create time series input vectors and ground truth
groundTruth = []
for index in range(np.size(feat,0)):
    
























  
