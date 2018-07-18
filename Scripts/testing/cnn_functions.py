import sys
sys.path.append('../')
import forecast_funcs as ff
import numpy as np
from pylab import *
from random import shuffle
from random import Random
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from netCDF4 import Dataset

import keras

#month=2 # 5=June, 0=January

def retrieve_grid(month, year, resolution):
    
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
    gridData[test] = 1
    
    return gridData
    
def create_input(month, year, numForecast, imDim, resolution, regBool):
   #Clear up the number of channels thing 
    
    padding = int(np.floor(imDim/2))
    data = retrieve_grid(month, year, resolution)
    dim = np.size(data,0)
    vertZeros = np.zeros((padding, dim))
    hortZeros = np.zeros((dim+(2*padding), padding))
    
    mat = []
    matYear = year
    for index in range(numForecast+1):
        if(month-index < 0):
            matYear = matYear - 1
            grid = retrieve_grid(11, matYear, resolution)
        else:    
            grid = retrieve_grid(month-index, matYear, resolution)
        #Create zero padding manually
        grid = np.vstack((vertZeros, grid))
        grid = np.vstack((grid, vertZeros))
        grid = np.hstack((hortZeros, grid))
        grid = np.hstack((grid, hortZeros)) 
        mat.append(grid)
    
    if(regBool == 1):
        regionMask = grid_region_mask(resolution)
        regionMask = np.vstack((vertZeros, regionMask))
        regionMask = np.vstack((regionMask, vertZeros))
        regionMask = np.hstack((hortZeros, regionMask))
        regionMask = np.hstack((regionMask, hortZeros)) 
        mat.append(regionMask)
        numChannels = numForecast+1
        mat = np.reshape(mat, (numChannels+1,np.size(mat[0],0),np.size(mat[0],0)))
    else:
        numChannels = numForecast   
        mat = np.reshape(mat, (numChannels+1,np.size(mat[0],0),np.size(mat[0],0)))

    #Get ground truth data.. (account for January transition)
    if(month == 11):
        gtGrid = retrieve_grid(0, year+1, resolution)
    else:    
        gtGrid = retrieve_grid(month+1, year, resolution)
    
    #Retrieve grid dimensions
    matRows = np.size(mat[0],0)
    matCols = np.size(mat[0],1)

    inputs = []
    gt = []
    #Create sliding window to extract volumes and add them to list
    for row in range(matRows-(2*padding)):
        for col in range(matCols-(2*padding)):
            inputs.append(mat[0:numChannels+1, row:row+imDim, col:col+imDim])
            gt.append(gtGrid[row, col])
        
    size = np.size(inputs,0)    
    
    return inputs,gt,size    



def shuffle_input(data, groundTruth): 
    
    #Get random integer
    seed = np.random.randint(0, 999999) 
    Random(seed).shuffle(data)
    Random(seed).shuffle(groundTruth)
    return data, groundTruth


def get_month_str(month):
    
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
        monthStr = 'September'   
    elif(month==9):
        monthStr = 'October'
    elif(month==10):
        monthStr = 'November'
    elif(month==11):
        monthStr = 'December' 
        
    return monthStr      
    

def create_input_with_predictions(predList, month, year, numForecast, imDim, resolution):
    
    padding = int(np.floor(imDim/2))
    data = retrieve_grid(month, year, resolution)
    dim = np.size(data,0)
    vertZeros = np.zeros((padding, dim))
    hortZeros = np.zeros((dim+(2*padding), padding))
    
    mat = []
    matYear = year
    monthMod = month
    numForMod = numForecast
    #Pad the predictions:
    for index in range(np.size(predList, 0)):
        grid = predList[index]
        #Create zero padding manually
        grid = np.vstack((vertZeros, grid))
        grid = np.vstack((grid, vertZeros))
        grid = np.hstack((hortZeros, grid))
        grid = np.hstack((grid, hortZeros))
        monthMod = monthMod - 1
        numForMod = numForMod - 1
        mat.append(grid)        
    
    for index in range(numForMod+1):
        if(monthMod-index < 0):
            matYear = matYear - 1
            grid = retrieve_grid(11, matYear, resolution)
        else:    
            grid = retrieve_grid(month-index, matYear, resolution)
        #Create zero padding manually
        grid = np.vstack((vertZeros, grid))
        grid = np.vstack((grid, vertZeros))
        grid = np.hstack((hortZeros, grid))
        grid = np.hstack((grid, hortZeros)) 
        mat.append(grid)
        
    mat = np.reshape(mat, (numForecast+1,np.size(mat[0],0),np.size(mat[0],0)))

    #Get ground truth data.. (account for January transition)
    if(month == 11):
        gtGrid = retrieve_grid(0, year+1, resolution)
    else:    
        gtGrid = retrieve_grid(month+1, year, resolution)
    
    #Retrieve grid dimensions
    matRows = np.size(mat[0],0)
    matCols = np.size(mat[0],1)

    inputs = []
    gt = []
    #Create sliding window to extract volumes and add them to list
    for row in range(matRows-(2*padding)):
        for col in range(matCols-(2*padding)):
            inputs.append(mat[0:numForecast+1, row:row+imDim, col:col+imDim])
            gt.append(gtGrid[row, col])
        
    size = np.size(inputs,0)    
        
    return inputs, gt, size   



def graph_pred_truth(predictions, forMonth, year, resolution):

    #Graph predictions
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    
    dx_res = resolution * 1000
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    #grid_str=str(int(dx_res/1000))+'km'
    lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)
    
    # Get lon/lats pf the ice concentration data on polar sterographic grid
    datapath = '../../Data/'
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
    
    truthGrid = retrieve_grid(forMonth, year, resolution)
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


def grid_region_mask(resolution):
    
    dataPath = '../../Data/'
    
    # BASEMAP INSTANCE
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l'  )
    dx_res = resolution*1000 #100000.
    nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
    lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)
    
    
    region_mask, xpts, ypts = ff.get_region_mask_sect(dataPath, m, xypts_return=1)
    region_mask=ma.masked_where(region_mask==1.5, region_mask)
    region_mask=ma.masked_where(region_mask>19.5, region_mask)
    
    region_maskG = griddata((xpts.flatten(), ypts.flatten()),region_mask.flatten(), (xptsG, yptsG), method='nearest')
    return region_maskG.data




'''
data, groundTruth, size = create_input(6, 1990, 6,  15, 100) #6=July
data   = np.float32(data)
model = keras.models.load_model('my_model.h5')
predictions = model.predict(data) #Predicts for August

#reconstruct predictions into grid to be displayed
predictions = np.reshape(predictions, (57, 57))

#graph_pred_truth(predictions, 7, 1990, 100)
'''

#regionMask = grid_regionmask(100)
grid = create_input(6, 1995, 3, 11, 100, 1)






      