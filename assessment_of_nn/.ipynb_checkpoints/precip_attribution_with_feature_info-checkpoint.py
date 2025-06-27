import xarray as xr
import dynlib.detect as detect
import dynlib.utils as dutils
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "8"
import glob
from pathlib import Path
from dynlib.metio import era5
from scipy.ndimage import label
import dynlib.gridlib as gridlib
from collections import Counter
import glob

def most_frequent_integer(matrix):
    # Flatten the 2D array into a 1D list    
    # Count occurrences using Counter
    m = matrix.flatten()
    
    count = Counter(m[m>0])
    
    # Find the most common integer
    most_common_num, _ = count.most_common(1)[0]  # Get the most frequent element
    
    return most_common_num

def feature_mask(mask, grid, pos_lon, pos_lat, ids): 

    lat_diff = np.abs(pos_lat[:, None] - grid.y[:,0][None, :])  # Distance for latitudes
    lon_diff = np.abs(pos_lon[:, None] - grid.x[0,:][None, :])  # Distance for longitudes


    # Get the lat/lon indices for all matching points
    lat_idx = np.argmin(lat_diff, axis=1)
    lon_idx = np.argmin(lon_diff, axis=1)

    # Update th}e mask for all points in one go
    for y_idx, x_idx,v in zip(lat_idx, lon_idx, ids):
        mask[y_idx, x_idx] = v  # Fill 3x3 grid
    return mask 
     

def netcdf_encoding(da, complevel=5): 
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in da.data_vars}
    return encoding

def generate_cyclone_mask(dates, cycpos, grid):
    '''
    Make cyclone mask with unique labels
    
    Input
    -----
        dates:  dates to find the cyclone mask 
        cycpos: dataframe of all cyclone position files 
        grid: dynlib gridlib
        
    Returns
    -------
        cyclmask: cyclone masked with track_id 
    '''
    
    cycmask = np.zeros((len(dates), grid.ny, grid.nx))
    label_mask = np.zeros_like(cycmask)
    mask_bool = np.zeros_like(cycmask)
    c_id_mask = np.zeros_like(cycmask)
    green_dist = dutils.dist_green_latlon(grid.x[0,:], grid.y[:,0])
    izero = np.argwhere(grid.x[0,:]==0).squeeze()
    
    # Here as well: The conservative approach (230 km) 
    A = 1e12
    r = 250e3# np.sqrt(A / np.pi) / 2

    for i, date in enumerate(dates):
        mask2d = cycmask[i]
        icyc = cycpos.query('date==@date and lat>0')
        
        lat_idx = np.abs(icyc.lat.values[:, None] - grid.y[:, 0][None, :]).argmin(axis=1)
        lon_idx = np.abs(icyc.lon.values[:, None] - grid.x[0, :][None, :]).argmin(axis=1)
        
        # Give the mask2d the track_id
        for y_idx, x_idx, v in zip(lat_idx, lon_idx, icyc.track_id.values):
            mask2d[y_idx, x_idx] = v
        
        cycmask[i] = mask2d
        dist = dutils.dist_from_mask_latlon(mask2d, green_dist, izero=izero)
        mask_bool[i] = dist < r

        structure = [[0,1,0],[1,1,1],[0,1,0]]
        cyclabel, ncyc = label(mask_bool[i], structure)
        # Make it work across the dateline 
        for y in range(cyclabel.shape[0]):
            if cyclabel[y, 0] > 0 and cyclabel[y, -1] > 0:
                cyclabel[cyclabel == cyclabel[y, -1]] = cyclabel[y, 0]

        label_mask[i] = cyclabel
        
        ## Relabel all the cycmasks 
        
        valid = (label_mask[i] > 0)
        for n in np.unique(label_mask[i][valid]):
            c_ids = np.unique(cycmask[i][(label_mask[i] == n) & (cycmask[i] > 0)])
            if c_ids.size > 0:
                c_id_mask[i][label_mask[i] == n] = c_ids[0]
    
    return c_id_mask




grid = era5.get_static()
thresh = 0.25
factor = -1
var = 'tot_precip'


for year in np.arange(1980,2022+1): 
    for month in np.arange(1,13): 
        month = "{:02d}".format(month)
        print(year); print(month)

        file_exists = glob.glob(f'/Data/gfi/scratch/kko033/attribution_varying_features/ea.for.{year}{month}.sfc.feature_info.{var}{thresh}.NH.nc')
        if not file_exists: 


            _f1 = xr.open_dataarray(f'/Data/gfi/share/era5/sfc/ea.for.{year}{month}.sfc.lsp.nc').sel(latitude=slice(90,0)) * 1000
            _f2 = xr.open_dataarray(f'/Data/gfi/share/era5/sfc/ea.for.{year}{month}.sfc.cp.nc').sel(latitude=slice(90,0))  * 1000
            field = _f1 + _f2
            field.name='tot_precip'

            blob_xr = xr.Dataset(None, coords = field.coords)
            dates = field.time.values

            lats,lons = np.meshgrid(field.latitude, field.longitude)
            grid = gridlib.grid_by_latlon(lats.T,lons.T)


            frovo = xr.open_dataset(f'/Data/gfi/spengler/jlu044/era5_4D_frovo/frovo_4D_v1_1_mountain_NH/frovo_4D_data_{year}-{month}.nc').rename(dict(lat = 'latitude',lon = 'longitude')).sel(pressure=850).interp_like(field, method = 'nearest')

            ## Caos are only detected in extended winter in NH 
            try:
                caos = xr.open_dataset(f'/Data/gfi/share/era5/detected_features/cold_air_outbreak_events/ea.ans.{year}{month}.sfc.nh_cao_id.nc')
            except FileNotFoundError:
                # Define shape and coords based on expected structure, e.g.:
                data = np.zeros(field.values.shape)
                caos = xr.Dataset({'nh_cao_id': (['time', 'latitude', 'longitude'], data)}, coords=field.coords)

            #caos = xr.open_dataset(f'/Data/gfi/share/era5/detected_features/cold_air_outbreak_events/ea.ans.{year}{month}.sfc.nh_cao_id.nc')

            #%% Need to open the features 
            cycs = xr.open_dataset('/Data/gfi/spengler/kko033/cyclone_clustering/data/all_combined_NH_trcks.nc').to_dataframe()
            #%% Grid the cyclones, select radius (250 km) --> might be good to test sensitivities 
            # and then label the mask according to the track_id 
            cycmask = generate_cyclone_mask(dates, cycs, grid)

            #%% Read in MTAs
            mta_df = xr.open_dataset(f'/Data/gfi/share/era5/detected_features/moisture_transport_axis/ea.ans.{year}{month}.sfc.mta.nc').to_dataframe()
            mta = np.zeros(frovo.frovo_id.values.shape)

            for i,date in enumerate(dates): 
                imta = mta_df.query('date == @date')
                mta[i,:,:] = feature_mask(mta[i], grid, imta.longitude.values, imta.latitude.values, imta.line_id)
            #Not the best solution
            mta[:,-1,:] = 0


            da = xr.Dataset(coords = field.coords)
            da['tp'] = field.where(field>thresh)
            da['C']          = (['time','latitude','longitude'], cycmask)
            da['F']          = frovo.frovo_id_mapped
            da['CAO']        = caos.nh_cao_id
            da['A']          = (['time','latitude','longitude'], mta)


            da = da.transpose('time','latitude','longitude')
            masks_conditions = [
                ('F', lambda x: (x.F > 0)),
                ('C', lambda x: (x.C > 0)),
                ('A', lambda x: (x.A > 0)),
                # Add more masks as needed
                # We don't include CAOs as they are filtered out 
            ]

            _da = (da.CAO>3) & (da.F<1)
            da['tot_precip'] = da.where(~_da).tp
            masks = {mask_name: np.zeros(da[var].shape).astype(np.float32) for mask_name, _ in masks_conditions}

            for i in range(da.sizes['time']):
                da_i = da.isel(time=i)
                blob_material = da_i[var].where(da_i[var]>thresh,0).values
                blobs,_ = detect.precip_blobs(blob_material*factor, grid, blob_mindist=750)
                blobs[blobs == 0] = np.nan

                dims = list(da_i.dims)
                coords = {dim: da_i[dim].values for dim in dims}

                da_i['blob'] = xr.DataArray(data=blobs.squeeze(), dims=dims, coords=coords, name = 'blob')

                # Loop through mask conditions
                for mask in masks:
                    blob_nr = np.unique(da_i.where(da_i[mask] > 0).blob.values)
                    blob_nr = blob_nr[~np.isnan(blob_nr)]

                    blob_f = np.zeros(da_i.blob.shape)
                    for n in blob_nr:
                        ## Need to find the frovo id in this timestep? 

                        blob_f = da_i.blob == n
                        f_ids = np.unique(da_i.where(da_i.blob == n)[mask])
                        f_ids = f_ids[(~np.isnan(f_ids)) & (f_ids > 0)]

                        if len(f_ids)>1: 
                            f_id = most_frequent_integer(f_ids)
                        else: 
                            f_id = f_ids[0]

                        # Put the value of the feature 
                        masks[mask][i, blob_f] = f_ids[0]


            # Create dataset for precipitation attribution
            precip_attribution = xr.Dataset(None, coords=field.coords).astype(np.float32)

            dims = list(field.dims)
            coords = {dim: field[dim].values for dim in dims}
            # Create data variables for masks
            for mask in masks:
                precip_attribution[mask] = xr.DataArray(data=masks[mask], dims=dims, coords=coords, name = mask)


            precip_attribution['CAO'] = da.where(da.CAO>3).CAO
            enc = netcdf_encoding(precip_attribution)
            precip_attribution.to_netcdf(f'/Data/gfi/scratch/kko033/attribution_varying_features/ea.for.{year}{month}.sfc.feature_info.{var}{thresh}.NH.nc', 
                                        encoding = enc)