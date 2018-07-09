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

#def data_collection():

rawDataPath = '../../Data/'
derivedDataPath = '../../DataOutput/'
forecastVar = 'conc'
hemStr = 'N'

startYr    = 1980
forecastYr = 2015
startMonth = 6 #June
stopMonth  = 9 #September
regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')

numFeat = 21
numSamples = (forecastYr - startYr) * (stopMonth+1 - startMonth)

#data collection
#start off with just the summer months and without ice thickness

feat = []
for year in range(startYr, forecastYr):
    for index, month in enumerate(range(startMonth, stopMonth+1)):
        
        #set month to array
        sample = np.zeros((numFeat*2) + 2)
        sample[0] = (month-1)/100 #gridded months are actually indexed at 0
        #divide by 100 to put in the same magnitude as the other values
        
        #set region features to array
        iceConc = ff.get_gridvar(derivedDataPath, forecastVar, month, 
                          array(year), hemStr)
        iceThick = ff.get_pmas_month(rawDataPath, year, month)
        
        for regInd in range(numFeat-1):
                
            regionData = iceConc.data
            regionData[iceConc.mask == True] = 0 #get rid of nan
            desiredRegion = regionMask == regInd
            sample[regInd+1] = np.ma.mean(np.multiply(regionData, desiredRegion))   
                
        for regInd in range(numFeat-1):
            
            regionData = iceThick.data
            regionData[iceThick.mask == True] = 0 #get rid of nan
            desiredRegion = regionMask == regInd
            sample[numFeat+regInd] = np.ma.mean(np.multiply(regionData, desiredRegion))                   
            
        feat.append(sample)
            
    #feat = np.reshape(feat, (numSamples, numFeat))
    #groundTruth = feat[1:np.size(feat,0), 1: np.size(feat,1)]  
    #feat = feat[0:np.size(feat,0)-1, :]
    
    #return feat, groundTruth