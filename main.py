import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
#
from ll_xy import lonlat_to_xy
#
from cartoplot import cartoplot
#
import cartopy
import cartopy.crs as ccrs
#
from scipy.spatial import KDTree


# Files to process
file_paths = ["ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc",
              "BTRin20110313000000424SSF1801GL.nc"
            ]

hemisphere = "n"


"""Convert all the lon/lat coordinates to a sensible x,y coordinate system"""
# SIT
dataset = nc.Dataset(file_paths[0], "r", format="NETCDF4")
lon_SIT = np.array(dataset["lon"])
lat_SIT = np.array(dataset["lat"])
SIT = np.array(dataset["sea_ice_thickness"])
dataset.close()

x_SIT,y_SIT = lonlat_to_xy(lon_SIT, lat_SIT, hemisphere)
#cartoplot(x_SIT, y_SIT, SIT, cbar_label="Sea ice thickness [m]")

# TB
dataset = nc.Dataset(file_paths[1], "r", format="NETCDF4")
lon_TB = np.array(dataset.groups["scene_env1"].variables["lon"]).flatten()
lat_TB = np.array(dataset.groups["scene_env1"].variables["lat"]).flatten()
TB = np.array(dataset.groups["scene_env1"].variables["tb"])
dataset.close()

# The first thing I would do is just drop all of the TBs below 60°N latitude as soon as you load the files. 
# Just subset the dataframe/array straight away
lat_level = 60
lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     # Set all values smaller the 60 (Condition) to True (NaN), False (Same)
mask = np.where(~np.isnan(lat_TB), True, False)         # Make a mask to remove all NaN values
lat_TB = lat_TB[mask]
lon_TB = lon_TB[mask]
TB = TB[:,0,:].flatten()[mask]  # Channel for data
TB = np.where(TB==-9e+33, np.nan, TB) # -9e+33 are filler values from the dataset


x_TB,y_TB = lonlat_to_xy(lon_TB, lat_TB, hemisphere)

cartoplot(x_TB, y_TB, TB, cbar_label="Temperature brightness [K]")

"""
# For every xy coordinate corresponding to an SIT measurement, you need to find the nearest xy coordinate of a TB measurement
# Build tree on Tbs, querry tree on envisat -> dist, index of nearest tb

TB_coord = np.column_stack((x_TB.ravel(), y_TB.ravel()))
SIT_coord = np.column_stack((x_SIT.ravel(), y_SIT.ravel()))

plt.figure()
plt.scatter(x_TB, y_TB, s=1, c="blue", alpha=0.3, label="TB")
plt.scatter(x_SIT, y_SIT, s=10, c="red", alpha=0.8, label="SIT")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

tree = KDTree(TB_coord)  
distances, indices = tree.query(SIT_coord)    

#print("Index i TB:", indices)
print("Avstånd:", distances)

print("Medelavstånd:", np.mean(distances))
print("Maxavstånd:", np.max(distances))
print("Minavstånd:", np.min(distances))
"""