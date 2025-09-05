"""
File:       main.py
Purpose:    

Function:   

Other:      Created by Thea Jonsson 2025-08-20
"""

import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
from cartoplot import multi_cartoplot
#
import format_data as fTBd 


file_paths = ["ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc", # SIT
              "BTRin20060313000000410SIF1401GL.nc", # SSM/I
              "BTRin20060313000000424SSF1601GL.nc" # SSMIS
            ]
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
hemisphere = "n"
lat_level = 60



# SIT
# Läsa in flera filer från en map och lägger in the i en data frame
if False:

  columns = ["TB_V19", "TB_H19", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 
  index = 0

  all_x_SIT = np.array([])
  all_y_SIT = np.array([])
  all_SIT = np.array([])

  all_TB_V19 = np.array([])
  all_TB_H19 = np.array([])
  all_TB_V37 = np.array([])
  all_TB_H37 = np.array([])

  folder_path_SIT = "/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2006/03/"
  folder_path_SSMIS = "/Volumes/Thea_SSD_1T/TB_2006_03_SSMIS/"

  files_SIT = sorted(os.listdir(folder_path_SIT))
  files_SSMIS = sorted(os.listdir(folder_path_SSMIS))

  for file_SIT, file_SSMIS in zip(files_SIT, files_SSMIS):

    path_SIT = os.path.join(folder_path_SIT, file_SIT)
    path_SSMIS = os.path.join(folder_path_SSMIS, file_SSMIS)

    
    if (not file_SIT[0].isalnum()) or (not file_SSMIS[0].isalnum()):
      continue
    
    x_SIT, y_SIT, SIT = fTBd.format_SIT(path_SIT)

    all_x_SIT = np.append(all_x_SIT, x_SIT)
    all_y_SIT = np.append(all_y_SIT, y_SIT)
    all_SIT = np.append(all_SIT, SIT)

    index = 0
    for i in range(len(group_SSMIS)):
      for j in range(2):
        vh = [1, 0]     # Channel number: scene_env1 -> [V19, H19], scene_env2 -> [V37, H37]
        x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, near_TB, nearest_TB_coords = fTBd.format_SSMIS(x_SIT, y_SIT, path_SSMIS, group_SSMIS[i], vh[j], debug=False)

        if index == 0:
          all_TB_V19 = np.append(all_TB_V19, near_TB)
        elif index == 1:
          all_TB_H19 = np.append(all_TB_H19, near_TB)
        elif index == 2:
          all_TB_V37 = np.append(all_TB_V37, near_TB)
        else:
          all_TB_H37 = np.append(all_TB_H37, near_TB)  

        index += 1

  df_TB_SSMIS["SIT"] = all_SIT
  df_TB_SSMIS["X_SIT"] = all_x_SIT
  df_TB_SSMIS["Y_SIT"] = all_y_SIT

  df_TB_SSMIS["TB_V19"] = all_TB_V19 
  df_TB_SSMIS["TB_H19"] = all_TB_H19 
  df_TB_SSMIS["TB_V37"] = all_TB_V37 
  df_TB_SSMIS["TB_H37"] = all_TB_H37 

  df_TB_SSMIS = df_TB_SSMIS.dropna()

  breakpoint()

  #all_SIT = np.where(all_SIT > 3.2, np.nan, all_SIT)
  #multi_cartoplot([all_x_SIT], [all_y_SIT], [all_SIT], cbar_label="Sea ice thickness [m]")

# Läs in en fil
if True:
  x_SIT, y_SIT, SIT = fTBd.format_SIT(file_paths[0])
  #multi_cartoplot([x_SIT, x_SIT], [y_SIT, y_SIT], [SIT, SIT], cbar_label="Sea ice thickness [m]")
  #multi_cartoplot([x_SIT], [y_SIT], [SIT, SIT], cbar_label="Sea ice thickness [m]")
  if False:
    plt.hist(SIT)
    #plt.ylim(0,30)
    plt.xlabel("SIT [m]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of SIT")
    #filename = f"/Users/theajonsson/Desktop/Histogram_{vh[i]}.png"
    #plt.savefig("/Users/theajonsson/Desktop/Histogram_SIT", dpi=300, bbox_inches="tight") 
    plt.show()
    plt.close()



columns = ["TB_V19", "TB_H19", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
df_TB_SSM_I = pd.DataFrame(columns=columns) 
index = 0
df_TB_SSM_I["SIT"] = SIT
df_TB_SSM_I["X_SIT"] = x_SIT
df_TB_SSM_I["Y_SIT"] = y_SIT
df_TB_SSMIS = df_TB_SSM_I.copy(deep=True)

# TB
# SSM/I
for i in range(4):
  vh = [0, 1, 3, 4]     # Channel number: scene_env -> [V19, H19, V37, H37]
  x_TB_SSM_I, y_TB_SSM_I, TB_SSM_I, TB_freq, nearest_TB_coords = fTBd.format_SSM_I(x_SIT, y_SIT, file_paths[1], group_SSM_I, vh[i], debug=False)

  # Plot histogram to indicate how much of a problem the extreme brightness temp values are
  if False:
    plt.hist(TB)
    plt.ylim(0,30)
    plt.xlabel("Brightness temperature [K]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of channel {vh[i]}")
    #filename = f"/Users/theajonsson/Desktop/Histogram_{vh[i]}.png"
    #plt.savefig(filename, dpi=300, bbox_inches="tight") 
    plt.show()
    plt.close()

  # Plot to check conversion of lon/lat to x/y coordinates 
  #multi_cartoplot([x_TB_SSM_I], [y_TB_SSM_I], [TB_SSM_I], cbar_label="Brightness temperature [K]")

  # Plot TB positions of nearest neighbor -> line up with SITs position
  #multi_cartoplot([nearest_TB_coords[:,0]], [nearest_TB_coords[:,1]], TB_freq, cbar_label=" [K]")

  df_TB_SSM_I[columns[index]] = TB_freq
  index += 1  

df_TB_SSM_I = df_TB_SSM_I.dropna()
#df_TB_SSM_I.to_csv("/Users/theajonsson/Desktop/TrainingData_SSM_I.csv", index=False)

# Plot TB positions of nearest neighbor -> line up with SITs position
if False:
  multi_cartoplot([np.array(df_TB_SSM_I["X_SIT"])],[np.array(df_TB_SSM_I["Y_SIT"])],
                  [np.array(df_TB_SSM_I["TB_V19"]), np.array(df_TB_SSM_I["TB_H19"]), np.array(df_TB_SSM_I["TB_V37"]), np.array(df_TB_SSM_I["TB_H37"])], 
                  cbar_label="Brightness temperature [K]",
                  title=["TB_V19","TB_H19","TB_V37","TB_H37"])





# SSMIS
# i, j, group, channel: 0 0 scene_env1 1, 0 1 scene_env1 0, 1 0 scene_env2 1, 1 1 scene_env2 0
index = 0
for i in range(len(group_SSMIS)):
  for j in range(2):
    vh = [1, 0]     # Channel number: scene_env1 -> [V19, H19], scene_env2 -> [V37, H37]
    x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, TB_freq, nearest_TB_coords = fTBd.format_SSMIS(x_SIT, y_SIT, file_paths[2], group_SSMIS[i], vh[j], debug=False)

    # Plot histogram to indicate how much of a problem the extreme brightness temp values are
    if False:
      plt.hist(TB)
      #plt.ylim(0,30)
      plt.xlabel("Brightness temperature [K]")
      plt.ylabel("Frequency")
      plt.title(f"Histogram of channel {group_SSMIS[i]}{vh[j]}")
      #filename = f"/Users/theajonsson/Desktop/Histogram_{group_SSMIS[i]}_{vh[j]}.png"
      plt.savefig(filename, dpi=300, bbox_inches="tight") 
      plt.show()
      plt.close()

    # Plot to check conversion of lon/lat to x/y coordinates
    #multi_cartoplot([x_TB_SSMIS], [y_TB_SSMIS], [TB_SSMIS], cbar_label="Brightness temperature [K]")

    # Plot TB positions of nearest neighbor -> line up with SITs position
    #multi_cartoplot([nearest_TB_coords[:,0]], [nearest_TB_coords[:,1]], TB_freq, cbar_label=" [K]")


    df_TB_SSMIS[columns[index]] = TB_freq     
    index += 1

df_TB_SSMIS = df_TB_SSMIS.dropna()
#df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TrainingData_SSMIS_xy.csv", index=False)



# Plot differences in TBs on same day and same channels between satellite missions
if False:
  # Problem: SSM/I << SSMIS
  from scipy.spatial import KDTree
  SSM_I_coord = np.column_stack((x_TB_SSM_I, y_TB_SSM_I))   # SIT
  SSMIS_coord = np.column_stack((x_TB_SSMIS, y_TB_SSMIS))   # TB
  
  tree = KDTree(SSMIS_coord)  # TB
  distances, indices = tree.query(SSM_I_coord)  # SIT
  nearest_coords = SSMIS_coord[indices] # TB

  nearest_TB_SSMIS = TB_SSMIS[indices]  # TB
  difference =  nearest_TB_SSMIS - TB_SSM_I 

  print("Max: ", np.nanmax(difference))
  print("Min: ", np.nanmin(difference))
  print("Mean: ", np.nanmean(difference))

  #multi_cartoplot([nearest_coords[:,0]], [nearest_coords[:,1]], difference)
