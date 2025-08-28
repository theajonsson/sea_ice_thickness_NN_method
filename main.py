import netCDF4 as nc
import numpy as np
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
#
import formatTBdata as fTBd 


file_paths = ["ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc", # SIT
              "BTRin20060313000000410SIF1401GL.nc", # SSM/I
              "BTRin20060313000000424SSF1601GL.nc", # SSMIS
              "ice_conc_nh_polstere-100_multi_201103131200.nc"
            ]
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
hemisphere = "n"
lat_level = 60



# SIT
x_SIT, y_SIT, SIT = fTBd.format_SIT(file_paths[0])
#cartoplot(x_SIT, y_SIT, SIT, cbar_label="Sea ice thickness [m]")



columns = ["TB_V19", "TB_H19", "TB_V37", "TB_H37", "SIT"]
df_TB_SSMIS = pd.DataFrame(columns=columns) 
index = 0

df_TB_SSMIS["SIT"] = SIT


df_TB_SSM_I = df_TB_SSMIS.copy(deep=True)

# TB
# SSM/I
for i in range(4):
  vh = [0, 1, 3, 4]     # Channel number
  TB_freq, nearest_TB_coords = fTBd.format_SSM_I(x_SIT, y_SIT, file_paths[1], group_SSM_I, vh[i], debug=True)

  #cartoplot(nearest_TB_coords[:,0], nearest_TB_coords[:,1], TB_freq, cbar_label=" [K]")

  df_TB_SSM_I[columns[index]] = TB_freq
  index += 1  

#df_TB_SSM_I.to_csv("/Users/theajonsson/Desktop/TrainingData_SSM_I.csv", index=False)


index = 0
# SSMIS
for i in range(len(group_SSMIS)):
  for j in range(2):
    vh = [1, 0]
    TB_freq, nearest_TB_coords = fTBd.format_SSMIS(x_SIT, y_SIT, file_paths[2], group_SSMIS[i], vh[j], debug=True)
    
    #cartoplot(nearest_TB_coords[:,0], nearest_TB_coords[:,1], TB_freq, cbar_label=" [K]")

    df_TB_SSMIS[columns[index]] = TB_freq
    index += 1

#df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TrainingData_SSMIS.csv", index=False)













"""
# You can further reduce the size of the dataset by following Soriot et al. and removing all rows where you think the sea ice concentration is <80%. 
# You can do this by downloading the day's sea ice concentration from OSISAF. 
# Given we're focused on sea ice volume we may want to relax this condition later, but for now let's follow Soriot et al. closely.
dataset = nc.Dataset(file_paths[-1], "r", format="NETCDF4")
lon_SIC = np.array(dataset["lon"]).flatten()
lat_SIC = np.array(dataset["lat"]).flatten()
#SIC = dataset["ice_conc"][:].filled(np.nan)     # _FillValue: -999
SIC = np.array(dataset["ice_conc"])
dataset.close()

x_SIC,y_SIC = lonlat_to_xy(lon_SIC, lat_SIC, hemisphere)
#cartoplot(x_SIC, y_SIC, SIC.flatten(), cbar_label="Concentration of sea ice [%]")

SIC_level = 80
SIC = np.where(SIC<SIC_level, np.nan, SIC)     # Set all values smaller the 60 (Condition) to True (NaN), False (Same)
mask = np.where(~np.isnan(SIC), True, False)         # Make a mask to remove all NaN values
"""