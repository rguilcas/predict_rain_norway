
import xarray as xr
import matplotlib.pyplot as plt 
import seaborn as sns
import cartopy.crs as ccrs
import seaborn.objects as so
import numpy as np
from dask.distributed import LocalCluster, Client
import flox.xarray
import xesmf as xe
from affine import Affine

from tqdm.notebook import tqdm
import scipy
import dask.bag as db
from pyproj import Proj, Transformer, CRS
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape

import pandas as pd
plt.style.use('robin')
from matplotlib.colors import LogNorm, Normalize
proj = ccrs.NorthPolarStereo(0,70)
import dask.bag as db

def sigmoid(x):
    return 1/(1+np.exp(-x))

run_id ='ima3ba04'

ds_attrs = xr.open_dataset(f'/Data/gfi/users/rogui7909/data/NN_outputs/attributions/newproj/{run_id}_ALL_attributions_v2.nc', chunks = dict(time_of_event=20)).load().transpose('y','x',...)


grid = ds_attrs.isel(time_of_event=0, timestep_past=0, var_name=0).attributions_lrp
lrp_normed = ds_attrs.attributions_lrp 
lrp_normed = lrp_normed.sum('var_name')/lrp_normed.sum('var_name').clip(0).sum(['x','y'])
ds_attrs['lrp_normed'] = lrp_normed

from esda.moran import Moran
from libpysal.weights import KNN, DistanceBand


times = ds_attrs.time_of_event.sel(time_of_event=slice('1979', '2024')).values

data = ds_attrs.sel(time_of_event='2016-08-09').sel(timestep_past=0).lrp_normed

X = data.to_dataframe().lrp_normed.reset_index().x.values
Y = data.to_dataframe().lrp_normed.reset_index().y.values
coords = np.column_stack([X, Y])
w = KNN.from_array(coords, k=4)  # or k=8

results = []

for time_of_event in tqdm(times):
    for timestep_past in range(-6,1):
        data_ = ds_attrs.sel(time_of_event=time_of_event).sel(timestep_past=timestep_past).lrp_normed
        numbers = data_.values.flatten()
        mi = Moran(numbers,w)
        results.append([time_of_event, timestep_past, mi.I, mi.p_sim])


ds = pd.DataFrame(results, columns = ['time_of_event','timestep_past','Moran_I','p_value']).set_index(['time_of_event','timestep_past']).to_xarray()
ds.to_netcdf(f'/Data/gfi/users/rogui7909/data/NN_outputs/attributions/newproj/{run_id}_moranI_lrp_normed_v2.nc')