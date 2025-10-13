"""
File:       format_data.py
Purpose:    Provides functions for formating data and performance of necessary steps for further analysis
            Uses file ll_xy.py

Function:   format_SIT, nearest_neighbor, split_tracks, format_SSM_I, format_SSMIS

Other:      Created by Thea Jonsson 2025-08-28
"""

import time
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from cartoplot import cartoplot


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
    TB_data = TB[indices]                           # TB value at that point

    return distances, nearest_TB_coords, TB_data





"""
Function:   land_mask
Purpose:    Create a mask of with a distance from land and corresponding coordiantes 

Input:      
Return:     
"""
def land_mask(min_distance_km=50, lat_level=60):

    d = nc.Dataset("NSIDC0772_LatLon_EASE2_N3.125km_v1.1.nc", "r", format="NETCDF4")
    lats = np.array(d['latitude'])
    lons = np.array(d['longitude'])
    d.close()

    grid_dir = "EASE2_N3.125km.LOCImask_land50_coast0km.5760x5760.bin"
    land_data = np.fromfile(grid_dir, dtype=np.uint8).reshape(5760, 5760)

    land_binary = np.isin(land_data, [0, 101, 252]).astype(np.uint8)

    # Distance from land
    dist_pixels = distance_transform_edt(1 - land_binary)
    dist_km = dist_pixels * 3.125  # each pixel = 3.125 km

    # Keep only water pixels (255) that are >= min_distance_km away from land
    water_mask = (land_data == 255) & (dist_km >= min_distance_km)

    lats_flat = lats.flatten()
    lons_flat = lons.flatten()
    water_flat = water_mask.flatten()

    valid = (~np.isnan(lats_flat)) & (lats_flat >= lat_level) & water_flat
    lats_valid = lats_flat[valid]
    lons_valid = lons_flat[valid]
    land_mask_data = np.full_like(lats_valid, 255, dtype=np.uint8)  # all are valid water

    return lons_valid, lats_valid, land_mask_data




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
    lon_SIT = np.array(dataset["lon"]).flatten()
    lat_SIT = np.array(dataset["lat"]).flatten()
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan).flatten()     # NaN instead of _FillValue=9.969209968386869e+36
    dataset.close()

    #mask = (lat_SIT >= lat_level) & ((SIT >= 0) | np.isnan(SIT))
    mask = (lat_SIT >= lat_level) & (SIT >= 0)
    lat_SIT = lat_SIT[mask]
    lon_SIT = lon_SIT[mask]
    SIT = SIT[mask]

    lons_valid, lats_valid, land_mask_data = land_mask()
    distances, nearest_coords, land_mask_data = nearest_neighbor(lon_SIT, lat_SIT, lons_valid, lats_valid, land_mask_data)

    x_SIT,y_SIT = lonlat_to_xy(nearest_coords[:,1], nearest_coords[:,0], hemisphere)

    mask = land_mask_data == 255
    x_SIT = x_SIT[mask]
    y_SIT = y_SIT[mask]
    SIT = SIT[mask]

    return x_SIT, y_SIT, SIT





"""
Function:   split_tracks
Purpose:    

Input:      
Return:     
"""
def split_tracks(df, distance_segment = 50000):

    # Calculate distance between data points
    x = df["X_SIT"].values
    y = df["Y_SIT"].values
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    distances = np.insert(distances, 0, 0)  # First data point: distance = 0

    result = {
        "TB_V19": [], "TB_H19": [], "TB_V22": [], "TB_V37": [], "TB_H37": [], "SIT": [], "X_SIT": [], "Y_SIT": []
    }
    temp = {
        "TB_V19": [], "TB_H19": [], "TB_V22": [], "TB_V37": [], "TB_H37": [], "SIT": [], "X_SIT": [], "Y_SIT": []
    }
    cumulative_distance = 0
    seg = 1
    for idx in range(len(df)):
        row = df.iloc[idx]

        for i in temp:
            temp[i].append(row[i])
        
        cumulative_distance += distances[idx]    

        if (idx + 1) > (df.shape[0]-1):
            next_dist = 0                   # Om sista tal, nästa distans är 0
        else:
            next_dist = distances[idx+1]    # Nästa distans
        
        if (cumulative_distance + next_dist) >= distance_segment:
            #print(f"Segment({seg}) is: {cumulative_distance * 0.001:.2f} [km] and contains {len(temp['SIT'])} values")

            # Remove segments with land contamination
            """ Some code to read the binary file with comments about where the data are from are in Landmask.ipynb
            use kdtree to find the distance from a given SIT measurement/TB measurement to the nearest land pixel.
            Just made a small change to show how to plot the mask at the end. Don't foget that "ice" in this case is "land" and can contaminate our pixels"""
            # Land masking of 35 km (half 70 km FOV of 19 GHz channel)
            #mask = np.fromfile(file_paths, dtype=np.uint8).reshape((5760, 5760))
            #land_mask = (mask == 0)
            # land koordinater
            #q=1900
            #lons[q:-q,q:-q],lats[q:-q,q:-q],data[q:-q,q:-q]
            # KDtree

            # Remove segments where more than x% of values are NaN
            if np.isnan(temp["SIT"]).sum() * (100 / len(temp["SIT"])) >= 50:
                for i in temp:
                    temp[i] = []
                cumulative_distance = 0
                continue

            # Average all values in each valid segment with np.nanmean
            for i in ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT"]:
                result[i].append(np.nanmean(temp[i]))
            
            # Middle coordinate of the whole segment
            x_mid = (temp["X_SIT"][-1] + temp["X_SIT"][0]) / 2
            y_mid = (temp["Y_SIT"][-1] + temp["Y_SIT"][0]) / 2
            result["X_SIT"].append(x_mid)
            result["Y_SIT"].append(y_mid)

            for i in temp:
                temp[i] = []
            cumulative_distance = 0
            seg += 1

    # If there are values left after segmenting
    if temp["SIT"]: 
        print(f"Segment({seg}) is: {cumulative_distance * 0.001:.2f} [km] and contains {len(temp['SIT'])} values")

        for i in ["TB_V19", "TB_H19", "TB_V22", "TB_V37", "TB_H37", "SIT"]:
            result[i].append(np.nanmean(temp[i]))

        x_mid = (temp["X_SIT"][-1] + temp["X_SIT"][0]) / 2
        y_mid = (temp["Y_SIT"][-1] + temp["Y_SIT"][0]) / 2
        result["X_SIT"].append(x_mid)
        result["Y_SIT"].append(y_mid)

    df_seg = pd.DataFrame(result)       # 41854 rows x 8 columns -> 777 rows x 8 columns

    return df_seg





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
            TB (float): masked TB values
            TB_freq (float): nearest TB values in realationship to the SIT values
            nearest_TB_coords (float, (N,2)-array)
"""
def format_SSMIS(x_SIT, y_SIT, file_paths, group, channel, lons_valid, lats_valid, land_mask_data,
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

    distances, nearest_coords, land_mask_data = nearest_neighbor(lon_TB, lat_TB, lons_valid, lats_valid, land_mask_data)
  
    x_TB,y_TB = lonlat_to_xy(nearest_coords[:,1], nearest_coords[:,0], hemisphere)

    mask = land_mask_data == 255
    x_TB = x_TB[mask]
    y_TB = y_TB[mask]
    TB = TB[mask]

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
