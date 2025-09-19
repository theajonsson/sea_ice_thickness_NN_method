"""
File:       main.py
Purpose:    Create training data for the NN.py
            Divided into three larger sections: TD for one month (SSMIS), TD for one day (SSMIS), TD for one day (SSM/I)
            The training data (TD) is filtered for different things with format_data.py,
            5 plots can be plotted for each TD section 

Needs:      cartoplot.py, format_data.py

Other:      Created by Thea Jonsson 2025-08-20
"""

import os
import time
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
from cartoplot import multi_cartoplot
#
import format_data as fTBd 
#
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
#
start_time = time.time() 

""" ========================================================================================== """
group_SSM_I = "scene_env"
group_SSMIS = ["scene_env1", "scene_env2"]
hemisphere = "n"
lat_level = 60
""" ========================================================================================== """





# Training data (.csv) used for the NN.py for ONE MONTH
""" ========================================================================================== """
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

  df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TD_SSMIS_1month_conv.csv", index=False)
  print(df_TB_SSMIS.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")


  
""" ==========================================================================================
          5 different type of plots to check for different things to consider
========================================================================================== """
# Plot: Histogram of SIT values to see if there are still some extreme SIT values left after convolve
if False:
    counts, edges, bars = plt.hist(np.array(df_TB_SSMIS["SIT"]))
    plt.bar_label(bars)
    plt.xlabel("SIT [m]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of SIT")
    plt.show()
    plt.close()

# Plot: SIT values visualized geographically
if False:
  multi_cartoplot([np.array(df_TB_SSMIS["X_SIT"])], [np.array(df_TB_SSMIS["Y_SIT"])], [np.array(df_TB_SSMIS["SIT"])], cbar_label="Sea ice thickness [m]")

# Plot with 4 subfigures: Check if nearest neighbor TBs value for each channel lines up with the SITs position visualized geographically 
if False:
  multi_cartoplot([np.array(df_TB_SSMIS["X_SIT"])],[np.array(df_TB_SSMIS["Y_SIT"])],
                  [np.array(df_TB_SSMIS["TB_V19"]), np.array(df_TB_SSMIS["TB_H19"]), np.array(df_TB_SSMIS["TB_V37"]), np.array(df_TB_SSMIS["TB_H37"])], 
                  cbar_label="Brightness temperature [K]",
                  title=["TB_V19","TB_H19","TB_V37","TB_H37"])

# Plot with 4 subfigures: Histogram to indicate how much of a problem the extreme brightness temperature values are for each channel
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    counts, edges, bars = ax.hist(df_TB_SSMIS[channel])
    ax.bar_label(bars)
    ax.set_title(f"Histogram of {channel}")
    ax.set_xlabel("Brightness temperature [K]")
    ax.set_ylabel("Frequency")

  plt.tight_layout() 
  plt.show()

# Plot with 4 subfigures: TB for each channel (y-axis) against SIT (x-axis) with a fitted line and R^2 score
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel])
    r_squared = r_value**2
  
    #ax.scatter(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], alpha=0.6)
    ax.hexbin(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], gridsize=50, mincnt=6)
    ax.plot(df_TB_SSMIS["SIT"], intercept + slope * df_TB_SSMIS["SIT"], color="red",
            label=f"Fitted line\nR^2: {r_squared:.3f}")
    
    ax.set_title(f"{channel} vs SIT")
    ax.set_xlabel("SIT [m]")
    ax.set_ylabel(f"{channel} [K]")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

  plt.tight_layout()
  #plt.savefig("/Users/theajonsson/Desktop/TBvsSIT_kernel300.png", dpi=300, bbox_inches="tight") 
  plt.show()









# Training data (.csv) used for the NN.py for ONE DAY for SSMIS
""" ========================================================================================== """
if False:
  file_paths = ["ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc", # SIT
                "BTRin20060313000000424SSF1601GL.nc" # TB
                ]

  x_SIT, y_SIT, SIT = fTBd.format_SIT(file_paths[0])

  columns = ["TB_V19", "TB_H19", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 
  index = 0
  df_TB_SSMIS["SIT"] = SIT
  df_TB_SSMIS["X_SIT"] = x_SIT
  df_TB_SSMIS["Y_SIT"] = y_SIT

  index = 0
  for i in range(len(group_SSMIS)):
    for j in range(2):
      vh = [1, 0]     # Channel number: scene_env1 -> [V19, H19], scene_env2 -> [V37, H37]
      x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, TB_freq, nearest_TB_coords = fTBd.format_SSMIS(x_SIT, y_SIT, file_paths[1], group_SSMIS[i], vh[j], debug=False)

      df_TB_SSMIS[columns[index]] = TB_freq     
      index += 1

  df_TB_SSMIS = df_TB_SSMIS.dropna()
  df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TD_SSMIS_1day.csv", index=False)
  print(df_TB_SSMIS.shape)
  end_time = time.time()
    
  print(f"Elapsed time: {end_time - start_time}")
  #exit()



""" ==========================================================================================
          5 different type of plots to check for different things to consider
========================================================================================== """
# Plot: Histogram of SIT values to see if there are still some extreme SIT values left after convolve
if False:
    counts, edges, bars = plt.hist(np.array(df_TB_SSMIS["SIT"]))
    plt.bar_label(bars)
    plt.xlabel("SIT [m]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of SIT")
    plt.show()
    plt.close()

# Plot: SIT values visualized geographically
if False:
  multi_cartoplot([np.array(df_TB_SSMIS["X_SIT"])], [np.array(df_TB_SSMIS["Y_SIT"])], [np.array(df_TB_SSMIS["SIT"])], cbar_label="Sea ice thickness [m]")

# Plot with 4 subfigures: Check if nearest neighbor TBs value for each channel lines up with the SITs position visualized geographically
if False:
  multi_cartoplot([np.array(df_TB_SSMIS["X_SIT"])],[np.array(df_TB_SSMIS["Y_SIT"])],
                  [np.array(df_TB_SSMIS["TB_V19"]), np.array(df_TB_SSMIS["TB_H19"]), np.array(df_TB_SSMIS["TB_V37"]), np.array(df_TB_SSMIS["TB_H37"])], 
                  cbar_label="Brightness temperature [K]",
                  title=["TB_V19","TB_H19","TB_V37","TB_H37"])

# Plot with 4 subfigures: Histogram to indicate how much of a problem the extreme brightness temperature values are for each channel
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    counts, edges, bars = ax.hist(df_TB_SSMIS[channel])
    ax.bar_label(bars)
    ax.set_title(f"Histogram of {channel}")
    ax.set_xlabel("Brightness temperature [K]")
    ax.set_ylabel("Frequency")

  plt.tight_layout()
  plt.show()

# Plot with 4 subfigures: TB for each channel (y-axis) against SIT (x-axis) with a fitted line and R^2 score
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel])
    r_squared = r_value**2
  
    #ax.scatter(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], alpha=0.6)
    ax.hexbin(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], gridsize=50, mincnt=6)
    ax.plot(df_TB_SSMIS["SIT"], intercept + slope * df_TB_SSMIS["SIT"], color="red",
            label=f"Fitted line\nR^2: {r_squared:.3f}")
    
    ax.set_title(f"{channel} vs SIT")
    ax.set_xlabel("SIT [m]")
    ax.set_ylabel(f"{channel} [K]")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

  plt.tight_layout()
  #plt.savefig("/Users/theajonsson/Desktop/TBvsSIT.png", dpi=300, bbox_inches="tight") 
  plt.show()










# Training data (.csv) used for the NN.py for ONE DAY for SSM/I
""" ========================================================================================== """
if False:
  file_paths = ["/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2006/03/ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20060331-fv2.0.nc",#"ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc", # SIT
                "/Volumes/Thea_SSD_1T/TB_2006_03_SSM_I/BTRin20060331000000410SIF1401GL.nc" #"BTRin20060313000000410SIF1401GL.nc" # SSM/I
            ]

  x_SIT, y_SIT, SIT = fTBd.format_SIT(file_paths[0])
  
  columns = ["TB_V19", "TB_H19", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
  df_TB_SSM_I = pd.DataFrame(columns=columns) 
  index = 0
  df_TB_SSM_I["SIT"] = SIT
  df_TB_SSM_I["X_SIT"] = x_SIT
  df_TB_SSM_I["Y_SIT"] = y_SIT

  index = 0
  for i in range(4):
    vh = [0, 1, 3, 4]     # Channel number: scene_env -> [V19, H19, V37, H37]
    x_TB_SSM_I, y_TB_SSM_I, TB_SSM_I, TB_freq, nearest_TB_coords = fTBd.format_SSM_I(x_SIT, y_SIT, file_paths[1], group_SSM_I, vh[i], debug=False)

    df_TB_SSM_I[columns[index]] = TB_freq
    index += 1  

  df_TB_SSM_I = df_TB_SSM_I.dropna()
  #df_TB_SSM_I.to_csv("/Users/theajonsson/Desktop/TD_SSM_I_1day.csv", index=False)
  print(df_TB_SSM_I.shape)
  end_time = time.time()
    
  print(f"Elapsed time: {end_time - start_time}")
  #exit()



""" ==========================================================================================
          5 different type of plots to check for different things to consider
========================================================================================== """
# Plot: Histogram of SIT values to see if there are still some extreme SIT values left after convolve
if False:
    counts, edges, bars = plt.hist(np.array(df_TB_SSM_I["SIT"]))
    plt.bar_label(bars)
    plt.xlabel("SIT [m]")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of SIT")
    plt.show()
    plt.close()

# Plot: SIT values visualized geographically
if False:
  multi_cartoplot([np.array(df_TB_SSM_I["X_SIT"])], [np.array(df_TB_SSM_I["Y_SIT"])], [np.array(df_TB_SSM_I["SIT"])], cbar_label="Sea ice thickness [m]")

# Plot with 4 subfigures: Check if nearest neighbor TBs value for each channel lines up with the SITs position visualized geographically
if False:
  multi_cartoplot([np.array(df_TB_SSM_I["X_SIT"])],[np.array(df_TB_SSM_I["Y_SIT"])],
                  [np.array(df_TB_SSM_I["TB_V19"]), np.array(df_TB_SSM_I["TB_H19"]), np.array(df_TB_SSM_I["TB_V37"]), np.array(df_TB_SSM_I["TB_H37"])], 
                  cbar_label="Brightness temperature [K]",
                  title=["TB_V19","TB_H19","TB_V37","TB_H37"])

# Plot with 4 subfigures: Histogram to indicate how much of a problem the extreme brightness temperature values are for each channel
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    counts, edges, bars = ax.hist(df_TB_SSM_I[channel])
    ax.bar_label(bars)
    ax.set_title(f"Histogram of {channel}")
    ax.set_xlabel("Brightness temperature [K]")
    ax.set_ylabel("Frequency")

  plt.tight_layout()
  plt.show()

# Plot with 4 subfigures: TB for each channel (y-axis) against SIT (x-axis) with a fitted line and R^2 score
if False:
  channel = columns[:4]
  fig, axes = plt.subplots(2,2, figsize=[10,5])
  axes = axes.flatten()

  for i, channel in enumerate(channel):
    ax = axes[i]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df_TB_SSM_I["SIT"], df_TB_SSM_I[channel])
    r_squared = r_value**2
  
    #ax.scatter(df_TB_SSM_I["SIT"], df_TB_SSM_I[channel], alpha=0.6)
    ax.hexbin(df_TB_SSM_I["SIT"], df_TB_SSM_I[channel], gridsize=50, mincnt=6)
    ax.plot(df_TB_SSM_I["SIT"], intercept + slope * df_TB_SSM_I["SIT"], color="red",
            label=f"Fitted line\nR^2: {r_squared:.3f}")
    
    ax.set_title(f"{channel} vs SIT")
    ax.set_xlabel("SIT [m]")
    ax.set_ylabel(f"{channel} [K]")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

  plt.tight_layout()
  #plt.savefig("/Users/theajonsson/Desktop/TBvsSIT_kernel300.png", dpi=300, bbox_inches="tight") 
  plt.show()









"""
#math 
distance = np.array([np.nan])
dist_sum = 0
for i in range(len(x_SIT)-1):
  if dist_sum >= 76000:
    breakpoint()
    dist_sum = 0
  #Pythagorassats  
  distance = np.append(distance, np.sqrt((x_SIT[i] - x_SIT[i+1])**2 + (y_SIT[i] - y_SIT[i+1])**2))
  dist_sum += np.sqrt((x_SIT[i] - x_SIT[i+1])**2 + (y_SIT[i] - y_SIT[i+1])**2)

breakpoint()

df_TB_SSM_I["lon_SIT"] = lon_SIT
df_TB_SSM_I["lat_SIT"] = lat_SIT
df_TB_SSM_I["Dist"] = distance

df_TB_SSM_I.to_csv("/Users/theajonsson/Desktop/SITdata_with_NaN.csv", index=False)
breakpoint()
"""



















# BEHÖVS SKRIVAS OM SÅ DEN KAN TA PER CHANNEL
""" ========================================================================================== """
# Plot differences in TBs on same day and same channels between satellite missions
if False:
  # Problem different amount of data points: SSM/I << SSMIS
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

  multi_cartoplot([nearest_coords[:,0]], [nearest_coords[:,1]], [difference])
