"""
Model Experiment Functions
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
from random import shuffle
from random import Random

def get_lat_lon_arr(resolution):
    
    """
    Purpose: Get longitude and latitude grids to use as features
    """
    
    datapath = '../../../Data/'
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    dx_res = resolution * 1000
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    lonsG, latsG, _, _ = m.makegrid(nx, ny, returnxy=True)  
    return latsG, lonsG


#Take this out and just load in the grids like how Alek did. Run separately for different resolution databases
#This is the same thing as retrieve grid. Remove one
def get_conc_grid(month, year, resolution):
    
    """
    Purpose: Retrieve the concentration grids using given month, year, and resolution
    """
    
    poleStr='A'# 'A: Arctic, AA: Antarctic
    alg=0 #0=Nasa team
    
    # File paths
    datapath = '../../../Data'
    
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

    """
    Purpose: Retrieve the concentration grids using given month, year, and resolution
    """
    
    # File paths
    datapath = '../../../Data/'
    
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
    
    """
    Purpose: Pad grid with zeros according to the image size so that each available
             grid is used for training. Especially useful if there's data at the edge.
    """
    
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
    conc = get_conc_grid(month, year, resolution)
    mat.append(border_grid(conc, padding))
    thick = get_thick_grid(month, year, resolution)
    mat.append(border_grid(thick, padding))
    
    #Code to not include zero padding
    #mat.append(get_conc_grid(month, year, resolution))
    #mat.append(get_thick_grid(month, year, resolution))
    
    return mat


def create_dataset(startYear, stopYear, startMonth, stopMonth, resolution, 
                   numTimeSeries, imDim):
    
    """
    Purpose: Create the grid cube required for time distributed training of
             the keras CNN-LSTM model. Input will only consider summer months.
    """
    
    padding = int(np.floor(imDim/2))
    
    #Create pixel coordinate matricies (includes the borders)
    xVect = np.arange(0, 63) / 63
    xMat = np.tile(xVect, (63, 1))
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
                gtMat.append(iceThickness/10)
            else:   
                gtMat.append(get_conc_grid(month+1, year, resolution))
                iceThickness = get_thick_grid(month+1, year, resolution)
                gtMat.append(iceThickness/10)  
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
    size = np.size(feat, 0)
              
    return feat, groundTruth, size

def pred_create_dataset(monthList, month, year, resolution, 
                   numTimeSeries, imDim):
    
    """
    Purpose: Create the grid cube required for time distributed training of
             the keras CNN-LSTM model. Input will include predicted months
    """
    
    padding = int(np.floor(imDim/2))
    
    #Create pixel coordinate matricies (includes the borders)
    xVect = np.arange(0, 63) / 63
    xMat = np.tile(xVect, (63, 1))
    yMat = xMat.transpose()
    
    feat = []
    groundTruth = []

    #Creating 3D volume for 3 element time series
    mat = []
    gtMat = []
    
    monthMod = month
    numTimeSeriesMod = numTimeSeries
    #Pad the predictions:
    for index in range(np.size(monthList, 0)):
        grid = monthList[index]
        iceConcGrid = grid[0]
        iceConcGrid = border_grid(iceConcGrid, padding)
        mat.append(iceConcGrid)
        iceThickGrid = grid[1]
        iceThickGrid = border_grid(iceThickGrid, padding)
        mat.append(iceThickGrid)
        if(monthMod-1 < 0):
            monthMod = 11
        else:    
            monthMod = monthMod - 1
        #if(index > 0):    
        numTimeSeriesMod = numTimeSeriesMod - 1    
        mat.append(xMat)
        mat.append(yMat)   
    
    
    for index in range(numTimeSeriesMod):
    
        if(monthMod-index < 0):
            monthMod = 11
            year = year-1
            build_ice_cube(mat, monthMod, year, resolution, padding)
        else:
            build_ice_cube(mat, monthMod-index, year, resolution, padding) 
        #Add position arrays
        mat.append(xMat)
        mat.append(yMat)
 
    #Reshaping into numpy 4D array
    mat = np.reshape(mat, (numTimeSeries, int(np.size(mat, 0)/numTimeSeries), np.size(mat[0],0), np.size(mat[0], 1)))
    
    #Get ground truth data (account for January transition)
    if(month == 11):
        gtMat.append(get_conc_grid(1, year+1, resolution))
        iceThickness = get_thick_grid(0, year+1, resolution)
        gtMat.append(iceThickness/10)
    else:   
        gtMat.append(get_conc_grid(month+1, year, resolution))
        iceThickness = get_thick_grid(month+1, year, resolution)
        gtMat.append(iceThickness/10)  
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
              
    return feat, groundTruth


def shuffle_input(feat, groundTruth): 
    
    """
    Purpose: Shuffle input features and ground truth
    """
    
    
    #Get random integer
    seed = np.random.randint(0, 999999) 
    Random(seed).shuffle(feat)
    Random(seed).shuffle(groundTruth)
    return feat, groundTruth


def get_month_str(month):
    
    """
    Purpose: Get month string for plots
    """

    if(month==0):
        monthStr = 'January'
    elif(month==1):
        monthStr = 'February'
    elif(month==2):
        monthStr = 'March'        
    elif(month==3):
        monthStr = 'April'         
    elif(month==4):
        monthStr = 'May'
    elif(month==5):
        monthStr = 'June'
    elif(month==6):
        monthStr = 'July'
    elif(month==7):
        monthStr = 'August'
    elif(month==8):
        monthStr = 'Sept'   
    elif(month==9):
        monthStr = 'October'
    elif(month==10):
        monthStr = 'November'
    elif(month==11):
        monthStr = 'December' 
        
    return monthStr 

def graph_pred_truth(predictions, forMonth, year, resolution):

    """
    Purpose: Graph prediction and truth grids
    """
    
    #Graph predictions
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    
    dx_res = resolution * 1000
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    #grid_str=str(int(dx_res/1000))+'km'
    lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)
    
    # Get lon/lats pf the ice concentration data on polar sterographic grid
    datapath = '../../../Data/'
    lats, lons = ff.get_psnlatslons(datapath)
    xpts, ypts =m(lons, lats)
    
    monthStr = get_month_str(forMonth)
    yearStr = str(year)
    
    fig_title = 'Comparison for ' + monthStr + ' ' + yearStr
    fig = figure(figsize=(6,6))
    fig.suptitle(fig_title, fontsize=18)
    
    ax = fig.add_subplot(221)
    ax.set_title('Predictions for ' + monthStr + ' ' + yearStr)
    im1 = m.pcolormesh(xptsG , yptsG, predictions, cmap=cm.Blues_r, vmin=0, vmax=1,shading='flat', zorder=2)
    
    truthGrid = get_conc_grid(forMonth, year, resolution)
    ax = fig.add_subplot(223)
    ax.set_title('Truth for ' + monthStr + ' ' + yearStr)
    im2 = m.pcolormesh(xptsG , yptsG, truthGrid, cmap=cm.Blues_r, vmin=0, vmax=1,shading='flat', zorder=2)
    
    #Calculate ice extent and ice areas
    areaStr   = "Ice Area: \n\nIce Extent: "
    
    ax = fig.add_subplot(222)
    ax.axis('off')
    ax.text(0.05, 0.95, areaStr,  fontsize=12,
            verticalalignment='top')
    
    ax = fig.add_subplot(224)
    ax.axis('off')
    ax.text(0.05, 0.95, areaStr,  fontsize=12,
            verticalalignment='top')
    
    plt.show()

def pred_lls_grid(month, fMonth,startYear, stopYear, resolution):
    
    """
    Purpose: Get a spatial map using least linear squares (LLS)
    """
    
    monthIceConc = get_conc_grid(month, startYear, resolution)
    finalMonthIceConc = get_conc_grid(fMonth, startYear, resolution)
    for year in range(startYear+1, stopYear+1):
        monthIceConc = np.dstack((monthIceConc, get_conc_grid(month, year, resolution)))
        finalMonthIceConc = np.dstack((finalMonthIceConc, get_conc_grid(fMonth, year, resolution)))
        
    fMonthIceConc = get_conc_grid(month, stopYear+1, 100)
    finalCellForecast = []
    for row in range(np.size(monthIceConc, 0)):
        for col in range(np.size(monthIceConc, 1)):
            vect = monthIceConc[row, col, :]
            fVect = finalMonthIceConc[row, col, :]
            model=sm.OLS(fVect, vect)
            fit=model.fit()
            cellForecast = fit.predict(fMonthIceConc[row, col])[0]
            finalCellForecast.append(cellForecast)
    
    #reshape list into final grid
    finalCellForecast = np.reshape(finalCellForecast, (57, 57))   
     
    return finalCellForecast 


def get_linear_grid(month, startYear, stopYear, resolution):

    """
    Purpose: get the linear trend spatial map
    """
    
    #get all sea ice concentration grids for the specific month and stack them
    years = np.arange(startYear, stopYear+1)

    monthIceConc = get_conc_grid(month, startYear, resolution)
    for year in range(startYear+1, stopYear+1):
        monthIceConc = np.dstack((monthIceConc, get_conc_grid(month, year, resolution)))

    #forMonthIceConc = retrieve_grid(month, stopYear+1, 100)
    finalCellForecast = []
    for row in range(np.size(monthIceConc, 0)):
        for col in range(np.size(monthIceConc, 1)):
            vect = monthIceConc[row, col, :]
            
            trendT, interceptT, _, _, _ = stats.linregress(years,vect)
            forecast = trendT*(stopYear+1) + interceptT
            finalCellForecast.append(forecast)
    
    #reshape list into final grid
    finalCellForecast = np.reshape(finalCellForecast, (57, 57))        
    return finalCellForecast 




def pred_cnn_create_input(monthList, month, year, numForecast, imDim, resolution):

    """
    Create CNN dataset using month predictions for features
    """    
    
    mat = []
    matYear = year
    monthMod = month
    subVal = 0
    padding = int(np.floor(imDim/2))
    
    #Pad the predictions:
    for index in range(np.size(monthList, 0)):
        grid = monthList[index]
        iceConcGrid = grid[0]
        iceConcGrid = border_grid(iceConcGrid, padding)
        mat.append(iceConcGrid)
        iceThickGrid = grid[1]
        iceThickGrid = border_grid(iceThickGrid, padding)
        mat.append(iceThickGrid)
        if(monthMod-1 < 0):
            monthMod = 11
        else:    
            monthMod = monthMod - 1
        #if(index > 0):    
        numForecast = numForecast - 1
   
       
    for index in range(numForecast+1):
        
        if(monthMod-subVal < 0):
            monthMod = 11
            matYear = matYear - 1
            grid = get_conc_grid(monthMod, matYear, resolution)
            grid = border_grid(grid, padding)
            subVal = 0 #reset value to subtract 
            mat.append(grid)
            iceThickness = get_thick_grid(monthMod, matYear, resolution)
            iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling  
        else:    
            grid = get_conc_grid(monthMod-subVal, matYear, resolution)
            grid = border_grid(grid, padding)
            mat.append(grid)
            iceThickness = get_thick_grid(monthMod-subVal, matYear, resolution)
            iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling    
        
        subVal = subVal+1
    
    xDim = np.size(mat[0], 0)
    
    #Create pixel coordinate matricies (includes the borders)
    xVect = np.arange(0, xDim) / xDim
    xMat = np.tile(xVect, (xDim, 1))
    yMat = xMat.transpose()
    mat.append(xMat)
    mat.append(yMat)
    
    mat = np.reshape(mat, (np.size(mat,0),np.size(mat[0],0),np.size(mat[0],0)))
     
    gtMat = []
    #Get ground truth data.. (account for January transition)
    if(month == 11):
        gtMat.append(get_conc_grid(1, year+1, resolution))
        iceThickness = get_thick_grid(0, year+1, resolution)
        gtMat.append(iceThickness/100)
    else:   
        gtMat.append(get_conc_grid(month+1, year, resolution))
        iceThickness = get_thick_grid(month+1, year, resolution)
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
    
    
def cnn_create_input(month, year, numForecast, imDim, resolution):

    """
    Purpose: Create input for CNN
    """

    mat = []
    matYear = year
    monthMod = month
    subVal = 0
    padding = int(np.floor(imDim/2))
    
    for index in range(numForecast+1):
        
        if(monthMod-subVal < 0):
            monthMod = 11
            matYear = matYear - 1
            grid = get_conc_grid(monthMod, matYear, resolution)
            grid = border_grid(grid, padding)
            subVal = 0 #reset value to subtract 
            mat.append(grid)
            iceThickness = get_thick_grid(monthMod, matYear, resolution)
            iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling  
        else:    
            grid = get_conc_grid(monthMod-subVal, matYear, resolution)
            grid = border_grid(grid, padding)
            mat.append(grid)
            iceThickness = get_thick_grid(monthMod-subVal, matYear, resolution)
            iceThickness = border_grid(iceThickness, padding)
            mat.append(iceThickness/100) #scaling    
        
        subVal = subVal+1
    
    xDim = np.size(mat[0], 0)
    #yDim = np.size(mat[0], 1)
    
    #Create pixel coordinate matricies (includes the borders)
    xVect = np.arange(0, xDim) / xDim
    xMat = np.tile(xVect, (xDim, 1))
    yMat = xMat.transpose()
    mat.append(xMat)
    mat.append(yMat)
    
    mat = np.reshape(mat, (np.size(mat,0),np.size(mat[0],0),np.size(mat[0],0)))
     
    gtMat = []
    #Get ground truth data(account for January transition)
    if(month == 11):
        gtMat.append(get_conc_grid(1, year+1, resolution))
        iceThickness = get_thick_grid(0, year+1, resolution)
        gtMat.append(iceThickness/100)
    else:   
        gtMat.append(get_conc_grid(month+1, year, resolution))
        iceThickness = get_thick_grid(month+1, year, resolution)
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











