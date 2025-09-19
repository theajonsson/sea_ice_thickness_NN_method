"""
File:       format_data.py
Purpose:    Provides functions for formating data and performance of necessary steps for further analysis
            Uses file ll_xy.py

Function:   nearest_neighbor, format_SIT, format_SSM_I, format_SSMIS

Other:      Created by Thea Jonsson 2025-08-28
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree





"""
Function:   convolve_SIT
Purpose:    Average the SIT values with a kernel size

Input:      SIT (float)
Return:     SIT_avg (float)
"""
def convolve_SIT(SIT, kernel=5, debug=False):
    
    SIT_avg = np.convolve(SIT, np.ones(kernel)/kernel, mode="valid")

    if debug:
        fig = plt.figure(figsize=[10,6])
        ax = fig.add_subplot(2,1,1)
        ax.plot(SIT, color="Blue")
        ax.plot([0, 40000], [np.mean(SIT), np.mean(SIT)], color="red", label=f"Mean SIT {np.mean(SIT)}")
        ax.set_title("SIT Unfilterd")
        ax.legend()
        ax = fig.add_subplot(2,1,2)
        ax.plot(SIT_avg, color="Blue")
        ax.plot([0, 40000], [np.mean(SIT_avg), np.mean(SIT_avg)], color="red", label=f"Mean SIT {np.mean(SIT_avg)}")
        ax.set_title(f"SIT Filtered with: {kernel}")
        ax.legend()
        plt.show()

    return SIT_avg



"""
Function:   nearest_neighbor
Purpose:    Find closest TB data point to each SIT data point

Input:      x (float): x coordinates for SIT and TB
            y (float): y coordinates for SIT and TB
            TB (float)
Return:     distances (float): distance from each SIT point to its nearest TB point
            nearest_TB_coords (float): coordinates of that nearest TB point
            TB_freq (float): corresponding TB value (for choosen frequency) at that nearest point
"""
def nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB):

    SIT_coord = np.column_stack((x_SIT, y_SIT))
    TB_coord = np.column_stack((x_TB, y_TB))

    tree = KDTree(TB_coord)                         # K-Dimensional Tree on TB coordinates 
    distances, indices = tree.query(SIT_coord)      # Queries the tree to find closest TB coordinate point for each SIT coordinate point

    nearest_TB_coords = TB_coord[indices]           # Coordinates of nearest TB point
    TB_freq = TB[indices]                           # TB value at that point

    return distances, nearest_TB_coords, TB_freq





"""
Function:   format_SIT
Purpose:    Format file from the RA-2 instrument on the Envisat satellite
            Read NetCDF file, loads data(lon, lat, SIT) and replaces fillvalue with NaN, filter all data below 60°N lat and removes NaN, 
            uses function convolve_SIT() to average SIT values, converts lon/lat into x/y coordinates 

Input:      file_paths (string)
Return:     x_SIT (float)
            y_SIT (float)
            SIT (float)
"""
def format_SIT(file_paths, lat_level=60, hemisphere="n"):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"])
    lat_SIT = np.array(dataset["lat"])
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan)     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()
 
    lat_SIT = np.where(lat_SIT<lat_level, np.nan, lat_SIT)
    mask_lat = np.where(~np.isnan(lat_SIT))         
    lat_SIT = lat_SIT[mask_lat]
    lon_SIT = lon_SIT[mask_lat]
    SIT = SIT[mask_lat]
   
    SIT = np.where(SIT<0, np.nan, SIT)
    mask_neg = np.where(~np.isnan(SIT)) # leave as nan
    lat_SIT = lat_SIT[mask_neg]
    lon_SIT = lon_SIT[mask_neg]
    SIT = SIT[mask_neg]

    mask_nan = np.isnan(SIT)
    SIT = SIT[~mask_nan]
    lat_SIT = lat_SIT[~mask_nan] 
    lon_SIT = lon_SIT[~mask_nan] 

    SIT = convolve_SIT(SIT, kernel=500)
  
    x_SIT,y_SIT = lonlat_to_xy(lon_SIT, lat_SIT, hemisphere)

    x_SIT = x_SIT[:len(SIT)]
    y_SIT = y_SIT[:len(SIT)]

    return x_SIT, y_SIT, SIT





"""
Function:   format_SSM_I
Purpose:    Format file from the SSM/I instrument on the DMSP-F14 satellite
            Read NetCDF file, loads data(lon, lat, TB) and replaces fillvalue with NaN, 
            filter all data below 60°N lat and removes NaN, converts lon/lat into x/y coordinates,
            uses function nearest_neighbor() to find nearest TB points to each SIT points

Input:      x_SIT (float)
            y_SIT (float)
            file_paths (string)
            group (string)
            channel (int)
Return:     x_TB (float)
            y_TB (float)
            TB (float)
            TB_freq (float)
            nearest_TB_coords (float, (N,2)-array)
"""
def format_SSM_I(x_SIT, y_SIT, file_paths, group, channel, 
                 lat_level=60, hemisphere="n", debug=False):

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())      # NaN instead of _FillValue=-9e+33          
    #ical = np.array(dataset.groups[group].variables["ical"][:,channel,:].filled(np.nan).flatten())
    dataset.close()
    #TB = TB + ical

    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask] 
    #ical = ical.flatten()[mask]

    x_TB,y_TB = lonlat_to_xy(lon_TB, lat_TB, hemisphere)

    distances, nearest_TB_coords, TB_freq = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)

    # Plot positions of nearest_TB_coords to check if they line up closely to the positions of SITs
    if debug:
        print("Group: ", group)
        print("Channel: ", channel)
        print("Mean distance:", np.mean(distances))
        print("Max distance:", np.max(distances))
        print("Min distance:", np.min(distances))

        plt.scatter(x_TB, y_TB, s=0.5, c="blue", alpha=0.2, label="TB")
        plt.scatter(x_SIT, y_SIT, s=15, c="red", label="SIT")
        plt.scatter(nearest_TB_coords[:, 0], nearest_TB_coords[:, 1], s=5, c="green", label="Nearst TB")
        plt.legend(loc="upper right")
        plt.title("SIT and their nearest TB neighbor")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        #plt.savefig("/Users/theajonsson/Desktop/nearestTB.png", dpi=300, bbox_inches="tight")
        plt.show()

    return x_TB, y_TB, TB, TB_freq, nearest_TB_coords





"""
Function:   format_SSMIS
Purpose:    Format file from the SSMIS instrument on the DMSP-F16 satellite
            Read NetCDF file, loads data(lon, lat, TB) and replaces fillvalue with NaN, 
            filter all data below 60°N lat and removes NaN, converts lon/lat into x/y coordinates,
            uses function nearest_neighbor() to find nearest TB points to each SIT points

Input:      x_SIT (float)
            y_SIT (float)
            file_paths (string)
            group (string)
            channel (int)
Return:     x_TB (float)
            y_TB (float)
            TB (float)
            TB_freq (float)
            nearest_TB_coords (float, (N,2)-array)
"""
def format_SSMIS(x_SIT, y_SIT, file_paths, group, channel, 
                 lat_level=60, hemisphere="n", debug=False):
    
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()   
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())      # NaN instead of _FillValue=-9e+33
    dataset.close()

    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask] 
  
    x_TB,y_TB = lonlat_to_xy(lon_TB, lat_TB, hemisphere)

    distances, nearest_TB_coords, TB_freq = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)

    # Plot positions of nearest_TB_coords to check if they line up closely to the positions of SITs
    if debug:
        print("Group: ", group)
        print("Channel: ", channel)
        print("Mean distance:", np.mean(distances))
        print("Max distance:", np.max(distances))
        print("Min distance:", np.min(distances))

        plt.scatter(x_TB, y_TB, s=0.5, c="blue", alpha=0.2, label="TB")
        plt.scatter(x_SIT, y_SIT, s=15, c="red", label="SIT")
        plt.scatter(nearest_TB_coords[:, 0], nearest_TB_coords[:, 1], s=5, c="green", label="Nearst TB")
        plt.legend(loc="upper right")
        plt.title("SIT and their nearest TB neighbor")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        #plt.savefig("/Users/theajonsson/Desktop/nearestTB.png", dpi=300, bbox_inches="tight")
        plt.show()

    return x_TB, y_TB, TB, TB_freq, nearest_TB_coords

