import xbatcher as xb
import xarray as xr
import torch
import xbatcher.loaders.torch
import dask 
import numpy as np


def add_timesteps(ds_rain, num_timesteps_predicted):
    if num_timesteps_predicted>1:
        ds_rain = ds_rain.rolling(time=num_timesteps_predicted).construct('timestep')
        ds_rain = ds_rain.shift(time=-num_timesteps_predicted+1)[:-num_timesteps_predicted+1]
    else:
        ds_rain = ds_rain.expand_dims('timestep').assign_coords(timestep=[0]).transpose('time','timestep')
    return ds_rain 

def filter_by_season(ds, season):
    if season in ['DJF','MAM', 'JJA', 'SON']:
        ds = ds.where(ds.time.dt.season==season, drop=True)
        return ds
    elif season == 'all':
        return ds
    else:
        raise ValueError("Season should be 'all', 'DJF','MAM', 'JJA' or 'SON'.")

def preprocess_rain(ds_rain, type_predictions, quantile_extreme, quantile_extreme_based_on_rainy_days):
    match type_predictions:
        case 'regression':
            return ds_rain
        case 'quantiles':
            return (ds_rain.rank('time', pct=True)//.1).astype(int)
        case 'boolean':
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme, 'time')
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme, 'time')
            return ((ds_rain > quantile_extreme_rain)*1).astype(int)
        case 'three_classes':
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme, 'time')
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme, 'time')
            no_rain = xr.ones_like(ds_rain).where(ds_rain>1,0)
            return no_rain.where(ds_rain<quantile_extreme_rain,2).astype(int)
        case _:
            raise ValueError("type_predictions must be 'boolean', 'three_classes', 'quantiles' or 'regression'")

def get_loader_from_ds(ds, batch_size):
    X_bgen = xb.BatchGenerator(
        ds.features,
        input_dims={'time': batch_size, 'var_name': ds.var_name.size, 'latitude': ds.features.latitude.size, 'longitude': ds.features.longitude.size},
        preload_batch=True,
    )
    y_bgen = xb.BatchGenerator(
        ds.targets,
        input_dims={'time': batch_size},
        preload_batch=True,
    ) 
    
    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)
    
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
            prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
            num_workers=1,  # Use 4 parallel worker processes to load data concurrently
            persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
            multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
            )

def get_expanded_ds(ds, noisy_samples=10, noise_scale=.3, shuffle_after_noise=True):
    ds_noisy = ds.expand_dims(noisy_samples=noisy_samples).rename(time='true_time')
    ds_noisy = ds_noisy.stack(time=['true_time','noisy_samples'])
    time = ds_noisy['time'].indexes['true_time'].get_level_values(0).values
    ds_noisy = ds_noisy.drop_vars(['time', 'true_time', 'noisy_samples']).assign_coords(time = time)
    noise = xr.DataArray(dask.array.random.normal(size=ds_noisy.features.shape, loc=0, scale=noise_scale), coords=ds_noisy.features.coords, dims=ds_noisy.features.dims)
    ds_noisy['features'] += noise
    ds_out = xr.concat([ds,ds_noisy], dim='time')
    if shuffle_after_noise:
        shuffled_time = np.arange(ds_out.time.size)
        np.random.shuffle(shuffled_time)
        ds_out = ds_out.isel(time=shuffled_time)
    return ds_out