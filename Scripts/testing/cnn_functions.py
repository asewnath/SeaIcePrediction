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
        ice_conc=ma.masked_where(ice_conc<=0.15, ice_conc)
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
    ice_conc_ma=ma.masked_where((latsG>pmask), ice_conc_ma)
    
    gridData = ice_conc_ma.data
    gridData[ice_conc_ma.mask == True] = 0
    
    return gridData
    

def create_input(month, year, numForecast, imDim, resolution):
   
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
        
    return inputs,gt,size    


def shuffle_input(data, groundTruth): 
    
    #Get random integer
    seed = np.random.randint(0, 999999) 
    Random(seed).shuffle(data)
    Random(seed).shuffle(groundTruth)
    return data, groundTruth

data, groundTruth, size = create_input(0, 1985, 5,  11, 100)
#data, labels = shuffle_input(data, groundTruth)

#new_data = np.reshape(data, (3249, 5, 5, 4))
#data = np.reshape(data, (3249, 7, 11, 11))

















      