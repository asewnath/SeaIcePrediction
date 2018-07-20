'''Experiment 1 Functions'''

import sys
sys.path.append('../')
import numpy as np
from pylab import *
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from netCDF4 import Dataset
import forecast_funcs as ff

from cnn_functions import get_lat_lon_arr
from cnn_functions import retrieve_grid
from cnn_functions import get_ice_thickness
from cnn_functions import get_month_str
#import numpy as np

def border_grid(grid, padding):
    
    dim = np.size(grid,0)
    vertZeros = np.zeros((padding, dim))
    hortZeros = np.zeros((dim+(2*padding), padding))
    
    grid = np.vstack((vertZeros, grid))
    grid = np.vstack((grid, vertZeros))
    grid = np.hstack((hortZeros, grid))
    grid = np.hstack((grid, hortZeros)) 
    
    return grid


def exp_create_input(month, year, numForecast, imDim, resolution):

    mat = []
    matYear = year
    monthMod = month
    subVal = 0
    padding = int(np.floor(imDim/2))
    
    for index in range(numForecast+1):
        
        if(monthMod-subVal < 0):
            monthMod = 11
            matYear = matYear - 1
            grid = retrieve_grid(monthMod, matYear, resolution)
            #grid = border_grid(grid, padding)
            subVal = 0 #reset value to subtract 
            mat.append(grid)
            iceThickness = get_ice_thickness(monthMod, matYear, resolution)
            #iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling  
        else:    
            grid = retrieve_grid(monthMod-subVal, matYear, resolution)
            #grid = border_grid(grid, imDim)
            mat.append(grid)
            iceThickness = get_ice_thickness(monthMod-subVal, matYear, resolution)
            #iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling    
        
        subVal = subVal+1
     
    
    lats, lons = get_lat_lon_arr(resolution)
    lats = lats/100
    lons = lons/100
    mat.append(lats)
    mat.append(lons)
    
    mat = np.reshape(mat, (np.size(mat,0),np.size(mat[0],0),np.size(mat[0],0)))
    
    
    gtMat = []
    #Get ground truth data.. (account for January transition)
    if(month == 11):
        gtMat.append(retrieve_grid(0, year+1, resolution))
        iceThickness = get_ice_thickness(0, year+1, resolution)
        gtMat.append(iceThickness/100)
    else:   
        gtMat.append(retrieve_grid(month+1, year, resolution))
        iceThickness = get_ice_thickness(month+1, year, resolution)
        gtMat.append(iceThickness/100)    
    gtMat = np.reshape(gtMat, (np.size(gtMat,0),np.size(gtMat[0],0),np.size(gtMat[0],0)))
    
    #Retrieve grid dimensions
    matChannels = np.size(mat, 0)
    matRows = np.size(mat[0],0)
    matCols = np.size(mat[0],1)
    gtChannels = np.size(gtMat, 0)
    
    inputs = []
    gt = []
    #Create sliding window to extract volumes and add them to list
    for row in range(matRows-(2*padding)):
        for col in range(matCols-(2*padding)):
            inputs.append(mat[0:matChannels, row:row+imDim, col:col+imDim])
            gt.append(gtMat[0:gtChannels, row, col])
        
    size = np.size(inputs,0) 

    return inputs, gt, size




