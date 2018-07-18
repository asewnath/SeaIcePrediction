############################################################## 
# Date: 01/01/16
# Name: grid_PIOMAS.py
# Author: Alek Petty
# Description: Grid the PIOMAS data onto the forecast grid

import matplotlib
matplotlib.use("AGG")
import pred_funcs as pfuncs
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from matplotlib import rc
from glob import glob
import matplotlib.patches as patches
from scipy.interpolate import griddata
import pandas as pd
from scipy import stats
import pred_funcs as pfuncs

import forecast_functions as ff



m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l'  )

#poleStr='A'# 'A: Arctic, AA: Antarctic
month=7 # 5=June, 0=January
year=1990
alg=0 #0=Nasa team

# File paths
datapath = '../../Data/'
#rawdatapath='../../Data/'
#dataoutpath='../../DataOutput/IceConcA/'
#figpath='../../Figures/Arctic/IceConc/'

# Get map projection and create regularly spaced grid from this projection
m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')

dx_res = 100000. # 100 km
nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
grid_str=str(int(dx_res/1000))+'km'
lonsG, latsG, xptsG, yptsG = m.makegrid(nx, ny, returnxy=True)

# Get lon/lats pf the ice concentration data on polar sterographic grid
lats, lons = ff.get_psnlatslons(datapath)
xpts, ypts =m(lons, lats)

#rawdatapath='../../../DATA/'
rawdatapath='../../Data'
pmasdatapath=rawdatapath+'/PIOMAS/heff/'
#dataoutpath='./Data_output/PMAS_OUT/'
#meltoutpath='./Data_output/MELT_OUT/'
#figpath='./Figures/'

#grid_str='100km'

#xpts100 =load(meltoutpath+'xpts'+grid_str)
#ypts100 =load(meltoutpath+'ypts'+grid_str)

#xpts100.dump(dataoutpath+'xpts'+grid_str)
#ypts100.dump(dataoutpath+'ypts'+grid_str)

start_year=1979
end_year=2015
month=4 #June



#for year in xrange(start_year, end_year+1, 1):
	#print year
xptsP, yptsP, thickness=ff.get_pmas_month(m, rawdatapath, year,month=month)#Figure out with this means by month please

#thickness_year = griddata((xptsP, yptsP),thickness, (xpts100, ypts100), method='linear')
#thickness_year_ma=ma.masked_where(thickness_year<0.01, thickness_year)

ice_thicknessG = griddata((xpts.flatten(), ypts.flatten()),thickness.flatten(), (xptsG, yptsG), method='linear')

#thickness_year.dump(dataoutpath+'pmas'+grid_str+str(year)+str(month))

fig = figure(figsize=(6,6))
ax1=gca()

im1 = m.pcolormesh(xptsG , yptsG, ice_thicknessG, cmap=cm.Blues_r, vmin=0, vmax=1,shading='flat', zorder=2)

#m.drawcoastlines(linewidth=0.25, zorder=5)
#m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=3)
#m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=3)
#m.fillcontinents(color='0.8',lake_color='0.7', zorder=2) #This throws an error before change

subplots_adjust( right = 0.99, left = 0.01, top=0.99, bottom=0.01)
#savefig(figpath+'/Arctic_BaKa_ice_extent_map_day1.png', dpi=1000)
plt.show()
#print(np.size(im1))


'''

poleStr='A'# 'A: Arctic, AA: Antarctic
month=7 # 5=June, 0=January
year=1990
alg=0 #0=Nasa team

# File paths
datapath = '../../Data/'
rawdatapath='../../Data/'
dataoutpath='../../DataOutput/IceConcA/'
figpath='../../Figures/Arctic/IceConc/'

# Get map projection and create regularly spaced grid from this projection
m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')

dx_res = 100000. # 100 km
nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
grid_str=str(int(dx_res/1000))+'km'
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

fig = figure(figsize=(6,6))
ax1=gca()

im1 = m.pcolormesh(xptsG , yptsG, ice_conc_ma, cmap=cm.Blues_r, vmin=0, vmax=1,shading='flat', zorder=2)

#m.drawcoastlines(linewidth=0.25, zorder=5)
#m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=3)
#m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=3)
#m.fillcontinents(color='0.8',lake_color='0.7', zorder=2) #This throws an error before change

subplots_adjust( right = 0.99, left = 0.01, top=0.99, bottom=0.01)
#savefig(figpath+'/Arctic_BaKa_ice_extent_map_day1.png', dpi=1000)
plt.show()
print(np.size(im1))

'''
