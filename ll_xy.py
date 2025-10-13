"""
File:       ll_xy.py
Purpose:    Provides function to be able to convert between two different coordinate systems (GCS/PCS) 

Function:   lonlat_to_xy

Other:      Based on Robbie Mallet's ll_xy.py (https://github.com/robbiemallett/custom_modules/blob/master/ll_xy.py) 
            Modified by Thea Jonsson since 2025-08-20
"""

from pyproj import Proj, Transformer
import numpy as np



"""
Function:   lonlat_to_xy
Purpose:    Converts longitude/latitude and EASE xy coordinates

Input:      lon (float): WGS84 longitude
            lat (float): WGS84 latitude
            hemisphere (string): 'n' or 's'
            inverse (bool): if true, converts xy to lon/lat
Return:     x/y or lon/lat (float)
"""
def lonlat_to_xy(coords_1, coords_2, hemisphere, inverse=False):

    EASE_Proj = {'n': 'EPSG:3408',
                 's': 'EPSG:3409'}
    
    WGS_Proj = 'EPSG:4326'
    
    for coords in [coords_1, coords_2]: assert isinstance(coords,(np.ndarray,list))

    if inverse == False: # lonlat to xy
        
        lon, lat = coords_1, coords_2
        
        transformer = Transformer.from_crs(WGS_Proj, EASE_Proj[hemisphere])
        
        x, y = transformer.transform(lon, lat)
        
        return (x, y)

    else: # xy to lonlat
        
        x, y = coords_1, coords_2
        
        transformer = Transformer.from_crs(EASE_Proj[hemisphere], WGS_Proj)
        
        lon, lat = transformer.transform(x, y)
        
        return (lon, lat)
