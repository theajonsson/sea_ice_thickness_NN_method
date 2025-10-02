"""
File:       cartoplot.py
Purpose:    Provide function for plotting polar maps

Function:   make_ax, multi_cartoplot

Other:      Based on Robbie Mallet's cartoplot.py (https://github.com/robbiemallett/custom_modules/blob/master/cartoplot.py)
            Modified by Thea Jonsson since 2025-08-20
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cartopy
import cartopy.crs as ccrs



"""
Function:   make_ax
Purpose:    Makes the subplots of the figure

Input:      fig 
            ax
            data (float): [coords_1, coords_2, data]
            cbar_label (string): title to colorbar
Return:     N/A
"""
def make_ax(fig, ax, data,
            title="", 
            land=True, ocean=False, bounding_lat=65,
            gridlines=True,
            maxlat=90, cbar_label=""):
    
    if ocean:
        ax.add_feature(cartopy.feature.OCEAN,zorder=2)
    if land:
        ax.add_feature(cartopy.feature.LAND, edgecolor='black',zorder=1)

    ax.set_extent([-180, 180, maxlat, bounding_lat], ccrs.PlateCarree())

    if gridlines:
        ax.gridlines()

    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right",size="5%", pad=0.05, axes_class=plt.Axes)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    m = ax.scatter(data[0], data[1], c=data[2],
                    s = 0.1,
                    transform=ccrs.epsg('3408'),
                    zorder=0, cmap="viridis_r")#,vmin=0, vmax=6)
    ax.set_title(title)

    cb = plt.colorbar(m, cax=ax_cb)
    cb.set_label(cbar_label)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)





"""
Function:   multi_cartoplot
Purpose:    Plots one or multiple (up to 4) polar plot using cartopy

Input:      coords_1 (float)
            coords_2 (float)
            data (float)
            cbar_label (string): title to colorbar
Return:     N/A
"""
def multi_cartoplot(coords_1, coords_2, data,
                title=[],
                figsize=[10,5],
                hem='n',
                land=True, ocean=False,
                bounding_lat=65,
                gridlines=True,
                cbar_label="", save_name = ""):
    

    fig = plt.figure(figsize=figsize)
    
    if hem == 'n':
        proj = ccrs.NorthPolarStereo()
        maxlat=90
    elif hem =='s':
        proj = ccrs.SouthPolarStereo()
        maxlat=-90
    else:
        raise

    # Set up for multiple plots
    plot_size = len(data)
    nr = len(coords_1)
    for i, arr in enumerate(data):

        if plot_size > 2:
            ax = fig.add_subplot(2, 2, i+1, projection=proj)
        else:
            ax = fig.add_subplot(1, 2, i+1, projection=proj)

        if nr > 1:
            plt_data = [coords_1[i], coords_2[i], arr]
        else:
            plt_data = [coords_1, coords_2, arr]

        if title:
            make_ax(fig, ax, plt_data,
                title=title[i], 
                land=land, ocean=ocean, bounding_lat=bounding_lat, 
                gridlines=gridlines, maxlat=maxlat, cbar_label=cbar_label)
        else:
            make_ax(fig, ax, plt_data, 
                land=land, ocean=ocean, bounding_lat=bounding_lat, 
                gridlines=gridlines, maxlat=maxlat, cbar_label=cbar_label)

        
    if  save_name:
        dir = "/Users/theajonsson/Desktop/"
        save_format = ".png"
        fig.savefig(dir+save_name+save_format, dpi=300, bbox_inches="tight")
    else:    
        plt.show()





def cartoplot(coords_1, coords_2, data,
                title=[],
                figsize=[10,5],
                hem='n',
                land=True, ocean=False,
                bounding_lat=65,
                gridlines=True,
                cbar_label="", save_name="", dot_size=0.1):
    

    fig = plt.figure(figsize=figsize)
    
    if hem == 'n':
        proj = ccrs.NorthPolarStereo()
        maxlat=90
    elif hem =='s':
        proj = ccrs.SouthPolarStereo()
        maxlat=-90
    else:
        raise

    data_array = [coords_1, coords_2, data]

    ax = fig.add_subplot(1, 1, 1, projection=proj)


    if ocean:
        ax.add_feature(cartopy.feature.OCEAN,zorder=2)
    if land:
        ax.add_feature(cartopy.feature.LAND, edgecolor='black',zorder=1)

    ax.set_extent([-180, 180, maxlat, bounding_lat], ccrs.PlateCarree())

    if gridlines:
        ax.gridlines()

    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right",size="5%", pad=0.05, axes_class=plt.Axes)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    for i in range(len(data)):
        m = ax.scatter(coords_1[i], coords_2[i], c=data[i],
                        s = dot_size,
                        transform=ccrs.epsg('3408'),
                        zorder=0, cmap="viridis_r", vmin=0, vmax=6)
    ax.set_title(title)

    cb = plt.colorbar(m, cax=ax_cb)
    cb.set_label(cbar_label)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)
        
    if  save_name:
        dir = "/Users/theajonsson/Desktop/"
        save_format = ".png"
        fig.savefig(dir+save_name+save_format, dpi=300, bbox_inches="tight")
    else:    
        plt.show()
