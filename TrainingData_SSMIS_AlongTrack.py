"""
File:       TrainingData_SSMIS_AlongTrack.py
Purpose:    Create training data on along track satellite data for the NN.py

Needs:      cartoplot.py, format_data.py

Other:      Created by Thea Jonsson 2025-09-19
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
import format_data as fd 
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
if True:

  columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT", "GR_V", "GR_H", "PR_19", "PR_37"]
  df_TB_SSMIS = pd.DataFrame(columns=columns) 
  index = 0

  all_x_SIT = np.array([])
  all_y_SIT = np.array([])
  all_SIT = np.array([])

  all_TB_V19 = np.array([])
  all_TB_H19 = np.array([])
  all_TB_V22 = np.array([])
  all_TB_V37 = np.array([])
  all_TB_H37 = np.array([])

  folder_path_SIT = "/Volumes/Thea_SSD_1T/Master Thesis/Envisat_SatSwath/2011/03"
  folder_path_SSMIS = "/Volumes/Thea_SSD_1T/TB_SSMIS/2011/03/"

  files_SIT = sorted([f for f in os.listdir(folder_path_SIT) if f[0].isalnum()])
  files_SSMIS = sorted([f for f in os.listdir(folder_path_SSMIS) if f[0].isalnum()])

  lons_valid, lats_valid, land_mask_data = fd.land_mask()

  for file_SIT, file_SSMIS in zip(files_SIT, files_SSMIS):

    path_SIT = os.path.join(folder_path_SIT, file_SIT)
    path_SSMIS = os.path.join(folder_path_SSMIS, file_SSMIS)
    
    x_SIT, y_SIT, SIT = fd.format_SIT(path_SIT)

    all_x_SIT = np.append(all_x_SIT, x_SIT)
    all_y_SIT = np.append(all_y_SIT, y_SIT)
    all_SIT = np.append(all_SIT, SIT)

    index = 0
    for i in range(len(group_SSMIS)):
      if i == 0:
         vh = [1, 0, 2]
      else:
         vh = [1, 0]
      for j in range(len(vh)):
        x_TB_SSMIS, y_TB_SSMIS, TB_SSMIS, near_TB, nearest_TB_coords = fd.format_SSMIS(x_SIT, y_SIT, path_SSMIS, group_SSMIS[i], vh[j], lons_valid, lats_valid, land_mask_data, debug=False)

        if index == 0:
          all_TB_V19 = np.append(all_TB_V19, near_TB)
        elif index == 1:
          all_TB_H19 = np.append(all_TB_H19, near_TB)
        elif index == 2:
          all_TB_V22 = np.append(all_TB_V22, near_TB)
        elif index == 3:
          all_TB_V37 = np.append(all_TB_V37, near_TB)
        else:
          all_TB_H37 = np.append(all_TB_H37, near_TB)  

        index += 1
    print(file_SIT)
    print(file_SSMIS)
    
  df_TB_SSMIS["SIT"] = all_SIT
  df_TB_SSMIS["X_SIT"] = all_x_SIT
  df_TB_SSMIS["Y_SIT"] = all_y_SIT

  df_TB_SSMIS["TB_V19"] = all_TB_V19 
  df_TB_SSMIS["TB_H19"] = all_TB_H19 
  df_TB_SSMIS["TB_V22"] = all_TB_V22 
  df_TB_SSMIS["TB_V37"] = all_TB_V37 
  df_TB_SSMIS["TB_H37"] = all_TB_H37 

  df_TB_SSMIS = df_TB_SSMIS.dropna(subset=["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"])

  # Split the tracks up into segments by distance
  # Try different distances to optimise your R2 score
  #df_TB_SSMIS = fd.split_tracks(df_TB_SSMIS, distance_segment = 40000)
  
  # Gradient ratio (GR) and Polarisation ratio (PR)
  df_TB_SSMIS["GR_V"] = (df_TB_SSMIS["TB_V37"] - df_TB_SSMIS["TB_V19"])/(df_TB_SSMIS["TB_V37"] + df_TB_SSMIS["TB_V19"])
  df_TB_SSMIS["GR_H"] = (df_TB_SSMIS["TB_H37"] - df_TB_SSMIS["TB_H19"])/(df_TB_SSMIS["TB_H37"] + df_TB_SSMIS["TB_H19"])
  df_TB_SSMIS["PR_19"] = (df_TB_SSMIS["TB_V19"] - df_TB_SSMIS["TB_H19"])/(df_TB_SSMIS["TB_V19"] + df_TB_SSMIS["TB_H19"])
  df_TB_SSMIS["PR_37"] = (df_TB_SSMIS["TB_V37"] - df_TB_SSMIS["TB_H37"])/(df_TB_SSMIS["TB_V37"] + df_TB_SSMIS["TB_H37"])

  df_TB_SSMIS.to_csv("/Users/theajonsson/Desktop/TD_SSMIS_1month_201103_maskland.csv", index=False)
  print(df_TB_SSMIS.shape)

  end_time = time.time()
  print(f"Elapsed time: {end_time - start_time}")
  
""" ==========================================================================================
          1 different type of plots to check for different things to consider
========================================================================================== """
# TB mot SIT
if False:
   channel_groups = [
      ["TB_V19", "TB_H19"],   
      ["TB_V22"],             
      ["TB_V37", "TB_H37"]    
    ]
   for group in channel_groups:
    fig, axs = plt.subplots(1, len(group), figsize=(7 * len(group), 6), constrained_layout=True)

    if len(group) == 1:
       axs = [axs]

    for ax, channel in zip(axs, group):
        slope, intercept, r_value, p_value, std_err = linregress(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel])
        r_squared = r_value ** 2

        hb = ax.hexbin(df_TB_SSMIS["SIT"], df_TB_SSMIS[channel], gridsize=50, mincnt=6)
        ax.plot(df_TB_SSMIS["SIT"], intercept + slope * df_TB_SSMIS["SIT"], color="red",
                label=f"Fitted line\nR^2 = {r_squared:.3f}")

        ax.set_title(f"{channel} vs SIT")
        ax.set_xlabel("SIT [m]")
        ax.set_ylabel(f"{channel} [K]")
        ax.legend()
        ax.grid(True)

    plt.savefig("/Users/theajonsson/Desktop/TBvsSIT_.png", dpi=300, bbox_inches="tight")
    plt.show()

# Histogram
plt.hist(df_TB_SSMIS["SIT"])
plt.savefig("/Users/theajonsson/Desktop/Hist_201011.png", dpi=300, bbox_inches="tight")





# Training data (.csv) used for the NN.py for ONE DAY for SSMIS
""" ========================================================================================== """
if False:
    file_paths = ["ESACCI-SEAICE-L2P-SITHICK-RA2_ENVISAT-NH-20110313-fv2.0.nc", # SIT
                  "BTRin20060313000000424SSF1601GL.nc" # TB
                  ]

    x_SIT, y_SIT, SIT = fd.format_SIT(file_paths[0])

    #columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT", "GR_V", "GR_H", "PR_19", "PR_37"]
    columns = ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT", "X_SIT", "Y_SIT"]
    df_TB_SSMIS = pd.DataFrame(columns=columns) 
    index = 0
    df_TB_SSMIS["SIT"] = SIT
    df_TB_SSMIS["X_SIT"] = x_SIT
    df_TB_SSMIS["Y_SIT"] = y_SIT

    index = 0
    for i in range(len(group_SSMIS)):
        if i == 0:
            vh = [1, 0, 2]  # Channel number: scene_env1 -> [V19, H19, V22]
        else:
            vh = [1, 0]     # Channel number: scene_env2 -> [V37, H37]
        
        for j in range(len(vh)):
            x_TB, y_TB, TB, TB_freq, nearest_TB_coords = fd.format_SSMIS(x_SIT, y_SIT, file_paths[1], group_SSMIS[i], vh[j], debug=False)

            df_TB_SSMIS[columns[index]] = TB_freq 
    
            index += 1
    
    df_TB_SSMIS = df_TB_SSMIS.dropna(subset=["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37"])

    # Split the tracks up into segments by distance
    #df_TB_SSMIS = fd.split_tracks(df_TB_SSMIS, distance_segment = 1000)

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

# Plot with 5 subfigures: Histogram to indicate how much of a problem the extreme brightness temperature values are for each channel
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
