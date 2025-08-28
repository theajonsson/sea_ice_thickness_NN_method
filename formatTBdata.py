import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from ll_xy import lonlat_to_xy
from scipy.spatial import KDTree



def nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB):
    """K-Dimensional Tree with data of TBs, search in the tree after nearest neighbor to SIT data"""
    SIT_coord = np.column_stack((x_SIT, y_SIT))
    TB_coord = np.column_stack((x_TB, y_TB))

    tree = KDTree(TB_coord)  
    distances, indices = tree.query(SIT_coord)

    nearest_TB_coords = TB_coord[indices]

    TB_freq = TB[indices]

    return distances, nearest_TB_coords, TB_freq



def format_SIT(file_paths, lat_level=60, hemisphere="n"):
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_SIT = np.array(dataset["lon"])
    lat_SIT = np.array(dataset["lat"])
    SIT = dataset["sea_ice_thickness"][:].filled(np.nan)     # NaN instead of _FillValue of 9.969209968386869e+36
    dataset.close()

    lat_SIT = np.where(lat_SIT<lat_level, np.nan, lat_SIT)
    mask = np.where(~np.isnan(lat_SIT))         
    lat_SIT = lat_SIT[mask]
    lon_SIT = lon_SIT[mask]
    SIT = SIT[mask]               

    mask_SIT = np.isnan(SIT)
    SIT = SIT[~mask_SIT]
    lon_SIT = lon_SIT[~mask_SIT]
    lat_SIT = lat_SIT[~mask_SIT]  

    x_SIT,y_SIT = lonlat_to_xy(lon_SIT, lat_SIT, hemisphere)

    return x_SIT, y_SIT, SIT



def format_SSMIS(x_SIT, y_SIT, file_paths, group, channel, 
                 lat_level=60, hemisphere="n", debug=False):
    
    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()   
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())
    dataset.close()

    # Reduce all data below 60°N latitude
    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask] 
  
    x_TB,y_TB = lonlat_to_xy(lon_TB, lat_TB, hemisphere)

    distances, nearest_TB_coords, TB_freq = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)

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
        #plt.savefig("/Users/theajonsson/Desktop/nearestTB.png", dpi=300, bbox_inches='tight')
        plt.show()

    return TB_freq, nearest_TB_coords



def format_SSM_I(x_SIT, y_SIT, file_paths, group, channel, 
                 lat_level=60, hemisphere="n", debug=False):
    if False:
        dataset = nc.Dataset(file_paths[2], "r", format="NETCDF4")
        lon_TB = np.array(dataset.groups["scene_env1"].variables["lon"]).flatten()   
        lat_TB = np.array(dataset.groups["scene_env1"].variables["lat"]).flatten()
        TB = np.array(dataset.groups["scene_env1"].variables["tb"][:,0,:].filled(np.nan).flatten())
        dataset.close()

        lat_level = 60
        lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     # Set all values smaller than 60 (Condition) to True (NaN), False (Same)
        mask = np.where(~np.isnan(lat_TB))         # Make a mask to remove all NaN values
        lat_TB = lat_TB[mask]
        lon_TB = lon_TB[mask]
        TB = TB[mask] 

        maxvalue = np.nanmax(TB) 
        minvalue = np.nanmin(TB)  

    dataset = nc.Dataset(file_paths, "r", format="NETCDF4")
    lon_TB = np.array(dataset.groups[group].variables["lon"]).flatten()
    lat_TB = np.array(dataset.groups[group].variables["lat"]).flatten()
    TB = np.array(dataset.groups[group].variables["tb"][:,channel,:].filled(np.nan).flatten())            
    #ical = np.array(dataset.groups[group].variables["ical"][:,channel,:].filled(np.nan).flatten())
    dataset.close()

    #TB = TB + ical

    """
    mask = ~np.any(TB == -9e+33, axis=0)
    lon_TB = lon_TB[:,mask].flatten()
    lat_TB = lat_TB[:,mask].flatten()
    TB = TB[:,mask].flatten()
    """

    # Reduce all data below 60°N latitude
    lat_TB = np.where(lat_TB<lat_level, np.nan, lat_TB)     
    mask = np.where(~np.isnan(lat_TB))         
    lat_TB = lat_TB[mask]
    lon_TB = lon_TB[mask]
    TB = TB[mask]  

    #ical = ical.flatten()[mask]

    #TB = np.where((TB>maxvalue) | (TB<minvalue), np.nan, TB)

    x_TB,y_TB = lonlat_to_xy(lon_TB, lat_TB, hemisphere)
    #cartoplot(x_TB, y_TB, TB, cbar_label="Brightness temperature [K]")

    distances, nearest_TB_coords, TB_freq = nearest_neighbor(x_SIT, y_SIT, x_TB, y_TB, TB)

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
        #plt.savefig("/Users/theajonsson/Desktop/nearestTB.png", dpi=300, bbox_inches='tight')
        plt.show()

    return TB_freq, nearest_TB_coords
    