"""
CNN-LSTM Functions
Author: Akira Sewnath
Date: 7/27/18
"""

import sys
sys.path.append('../')
import numpy as np
from pylab import *
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from netCDF4 import Dataset
import forecast_funcs as ff
import statsmodels.api as sm
from scipy import stats


def get_conc_grid(month, year, resolution):
    
    poleStr='A'# 'A: Arctic, AA: Antarctic
    alg=0 #0=Nasa team
    
    # File paths
    datapath = '../../Data/'
    
    # Get map projection and create regularly spaced grid from this projection
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    
    dx_res = resolution * 1000
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)  
    
    # Get lon/lats pf the ice concentration data on polar sterographic grid
    lats, lons = ff.get_psnlatslons(datapath)
    xpts, ypts =m(lons, lats)
    
    f = Dataset(datapath+'/OTHER/NIC_valid_ice_mask.N25km.01.1972-2007.nc', 'r')
    ice_flag = f.variables['valid_ice_flag'][:]
    
    if (year>2015):
        ice_conc = ff.get_month_concSN_NRT(datapath, year, month, alg=alg, pole=poleStr, monthMean=1)
        # Mask values below 0.15
        ice_conc=ma.masked_where(ice_conc<=0.15, ice_conc) #I don't get this...
    else:
        ice_conc = ff.get_month_concSN(datapath, year, month, alg=alg, pole=poleStr)
    
    # fill ice concentration data with zeroes
    ice_conc = ice_conc.filled(0)
    ice_conc = where((ice_flag >=1.5), 0, ice_conc)
    
    # Note the pole hole due to the incomplete satellite orbit
    if (year<1987):
        pmask=84.
    elif((year==1987)&(month<=5)):
        pmask=84.
    elif ((year==1987)&(month>5)):
        pmask=86.5
    elif ((year>1987)&(year<2008)):
        pmask=86.5
    else:
        pmask=88.5
    
    # Grid data
    ice_concG = griddata((xpts.flatten(), ypts.flatten()),ice_conc.flatten(), (xptsG, yptsG), method='linear')
    ice_conc_ma=ma.masked_where(np.isnan(ice_concG), ice_concG)
    #ice_conc_ma=ma.masked_where((latsG>pmask), ice_conc_ma)
    
    test = np.where(latsG>pmask)
    gridData = ice_conc_ma.data
    gridData[ice_conc_ma.mask == True] = 0
    
    #WARNING: Super ghetto solution to not just set the hole to 1
    arr1 = np.arange(np.min(test[0]), np.max(test[0]))
    arr2 = np.ones((1, np.size(arr1)), dtype=np.int ) * (np.min(test[0])-1)
    tup  = (arr1, arr2)
    gridMean = sum(gridData[tup])/np.size(arr1)
    gridData[test] = gridMean #fix
    
    return gridData


def get_thick_grid(month, year, resolution):

    # File paths
    datapath = '../../Data/'
    
    # Get map projection and create regularly spaced grid from this projection
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    
    dx_res = resolution*1000. # 100 km
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)
    
    # Get lon/lats on polar sterographic grid
    lats, lons = ff.get_psnlatslons(datapath)
    xpts, ypts =m(lons, lats)
    
    xptsP, yptsP, thickness=ff.get_pmas_month(m, datapath, year, month)
    iceThicknessG = griddata((xptsP, yptsP),thickness, (xptsG, yptsG), method='linear')
    
    return iceThicknessG


def border_grid(grid, padding):
    
    dim = np.size(grid,0)
    vertZeros = np.zeros((padding, dim))
    hortZeros = np.zeros((dim+(2*padding), padding))
    
    grid = np.vstack((vertZeros, grid))
    grid = np.vstack((grid, vertZeros))
    grid = np.hstack((hortZeros, grid))
    grid = np.hstack((grid, hortZeros)) 
    
    return grid



def build_ice_cube(mat, month, year, resolution, padding):
    
    """
    Purpose: Build 3D volume for time series, focusing only on adding ice
             features
    """
    #conc = get_conc_grid(month, year, resolution)
    #mat.append(border_grid(conc, padding))
    #thick = get_thick_grid(month, year, resolution)
    #mat.append(border_grid(thick, padding))
    
    mat.append(get_conc_grid(month, year, resolution))
    mat.append(get_thick_grid(month, year, resolution))
    
    return mat


def create_dataset(startYear, stopYear, startMonth, stopMonth, resolution, 
                   numTimeSeries, imDim):
    
    """
    Purpose: Create the grid cube required for time distributed training of
             the keras CNN-LSTM model. Input will only consider summer months.
    """
    
    padding = int(np.floor(imDim/2))
    
    #Create pixel coordinate matricies (includes the borders)
    xVect = np.arange(0, 57) / 57
    xMat = np.tile(xVect, (57, 1))
    yMat = xMat.transpose()
    
    feat = []
    groundTruth = []
    
    for year in range(startYear, stopYear+1):
        print(year)
        for month in range(startMonth, stopMonth+1):
    
            #Creating 3D volume for 3 element time series
            mat = []
            gtMat = []
            
            for index in range(numTimeSeries):
            
                if(month-index < 0):
                    month = 11
                    year = year-1
                    build_ice_cube(mat, month, year, resolution, padding)
                else:
                    build_ice_cube(mat, month-index, year, resolution, padding) 
                #Add position arrays
                mat.append(xMat)
                mat.append(yMat)
     
            #Reshaping into numpy 4D array
            mat = np.reshape(mat, (numTimeSeries, int(np.size(mat, 0)/numTimeSeries), np.size(mat[0],0), np.size(mat[0], 1)))
            
            #Get ground truth data (account for January transition)
            if(month == 11):
                gtMat.append(get_conc_grid(1, year+1, resolution))
                iceThickness = get_thick_grid(0, year+1, resolution)
                gtMat.append(iceThickness/100)
            else:   
                gtMat.append(get_conc_grid(month+1, year, resolution))
                iceThickness = get_thick_grid(month+1, year, resolution)
                gtMat.append(iceThickness/100)  
            #Reshaping into numpy 3D array    
            gtMat = np.reshape(gtMat, (np.size(gtMat,0),np.size(gtMat[0],0),np.size(gtMat[0],1)))
            
            #Retrieve grid dimensions
            matChannels = np.size(mat, 1)
            matRows = np.size(mat[0],1)
            matCols = np.size(mat[0],2)
            gtChannels = 2 #predicting both ice concentration and thickness    
            
            #Create sliding window to extract volumes and add them to list
            for row in range(matRows-(2*padding)):
                for col in range(matCols-(2*padding)):
                    feat.append(mat[0:numTimeSeries, 0:matChannels, row:row+imDim, col:col+imDim])
                    groundTruth.append(gtMat[0:gtChannels, row, col])   
                    
            print(month)
              
    return feat, groundTruth


def shuffle_input(feat, groundTruth): 
    
    #Get random integer
    seed = np.random.randint(0, 999999) 
    Random(seed).shuffle(feat)
    Random(seed).shuffle(groundTruth)
    return feat, groundTruth


feat, groundTruth = create_dataset(1985, 2014, 5, 9, 100, 3, 7)



