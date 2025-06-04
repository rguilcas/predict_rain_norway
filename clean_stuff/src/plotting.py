import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon

import cartopy.crs as ccrs
import matplotlib.ticker as mticker

def clean_map_ax(ax, boundaries=(-90, 50, 0, 90), gl=True):
    lon_min, lon_max , lat_min, lat_max = boundaries

    ax.coastlines( color='.5',linewidths=.3)
    # ax.set_extent((-60,60,0,80))
    # ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    
    # ax.set_extent((-60,60,20,80), crs=ccrs.PlateCarree())
    if gl:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    linewidth=.5, color='gray', alpha=0.5, linestyle='-')
        gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,17)[:-1])
        gl.ylocator = mticker.FixedLocator(np.arange(0,81,20))
        
    # longitudes = np.linspace(lon_min, lon_max, 100)
    # latitudes = np.linspace(lat_min, lat_max,100)
    # xx = list(longitudes)[::-1]+\
    #         [longitudes.min() for _ in range(latitudes.size)] + \
    #             list(longitudes)+\
    #                 [longitudes.max() for _ in range(latitudes.size)]
    # yy = [latitudes.max() for _ in range(longitudes.size)] \
    #         + list(latitudes)[::-1]+\
    #             [latitudes.min() for _ in range(longitudes.size)]+ \
    #                 list(latitudes)[::-1]
    # verts = np.array([xx, yy]).T
    # circle = Path(verts)
    # ax.set_boundary(circle, transform=ccrs.PlateCarree())

def plot_mean_attributions(ds_attributions):
    ds_attrs = ds_attributions.mean('time')
    ds_attrs = ds_attrs/ds_attrs.max(['longitude','latitude','var_name'])
    plot = ds_attrs.attributions.plot(col='timestep',row='var_name', transform=ccrs.PlateCarree(), subplot_kws=dict(projection=ccrs.PlateCarree()), aspect=ds_attrs.longitude.size/ds_attrs.latitude.size, size=2)
    for ax in plot.axs.flatten():
        clean_map_ax(ax)
    return plot

def plot_top1pct_pixels(ds_attributions, top_percent = .01):
    ds_top = (ds_attributions.stack(space=['latitude','longitude','var_name']).rank('space',pct=True)>(1-top_percent)).unstack('space').mean('time')
    plot = ds_top.attributions.where(ds_top.attributions>0).plot(col='timestep',row='var_name', transform=ccrs.PlateCarree(), subplot_kws=dict(projection=ccrs.PlateCarree()), aspect=ds_top.longitude.size/ds_top.latitude.size, size=2, levels=np.arange(0,1.1,.1), cmap='mako_r')
    for ax in plot.axs.flatten():
        clean_map_ax(ax)
    return plot