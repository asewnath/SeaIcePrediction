#Script to gather data for sequential model experimentation

"""
Data Required:
    Model requires input vector of sea ice concentrations
    
    Month number may have to be normalized to match magnitude of other features

"""
import forecast_funcs as ff
import numpy as np
from pylab import array


def data_collection():

    rawDataPath = '../../Data/'
    derivedDataPath = '../../DataOutput/'
    forecastVar = 'conc'
    iceType = 'extent'
    hemStr = 'N'
    siiVersion ='v3.0'
    
    startYr    = 1990
    forecastYr = 2010
    startMonth = 2 
    stopMonth  = 11 
    regionMask = np.load(derivedDataPath + 'Regions/regionMaskA100km')
    
    numFeat = 18
    numSamples = (forecastYr - startYr) * (stopMonth+1 - startMonth)
    
    #data collection
    #start off with just the summer months and without ice thickness
    
    feat = []
    groundTruth = []
    for year in range(startYr, forecastYr):
        for index, month in enumerate(range(startMonth, stopMonth+1)):
            
            _, extent = ff.get_ice_extentN(rawDataPath, month, year, year, iceType, siiVersion, hemStr)
            groundTruth.append(extent[0])
            
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
       
            feat.append(sample)
            
    feat = np.reshape(feat, (numSamples, numFeat))
    groundTruth = np.reshape(groundTruth, (numSamples, 1))
        
    return feat, groundTruth