import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs



def cartoplot(lon, lat, data,
              bounding_lat=65,
              land=True,
              ocean=False,
              gridlines=True,
              figsize=[10,5],
              save_dir=None,
              show=True,
              hem='n',
              color_scale=(None,None),
              color_scheme='binary',
              cbar_label=""):
    
    """
    Plots a north polar plot using cartopy.
    Must be supplied with gridded arrays of lon, lat and data
    """

    # Make plot

    fig = plt.figure(figsize=figsize)
    
    if hem == 'n':
        proj = ccrs.NorthPolarStereo()
        maxlat=90
    elif hem =='s':
        proj = ccrs.SouthPolarStereo()
        maxlat=-90
    else:
        raise
        
    ax = plt.axes(projection=proj)
    
    
    if ocean == True:
        ax.add_feature(cartopy.feature.OCEAN,zorder=2)
    if land == True:
        ax.add_feature(cartopy.feature.LAND, edgecolor='black',zorder=1)

    ax.set_extent([-180, 180, maxlat, bounding_lat], ccrs.PlateCarree())
    
    if gridlines == True:
        ax.gridlines()

    m = ax.scatter(np.array(lon).ravel(), np.array(lat).ravel(), c=data,
                    s = 0.1,
                    transform=ccrs.epsg('3408'),
                    zorder=0)
    
    cbar = fig.colorbar(m)
    if cbar_label:
        cbar.set_label(cbar_label)

    #fig.savefig("/Users/theajonsson/Desktop/SSM_I.png", dpi=300, bbox_inches='tight')    
    plt.show()

