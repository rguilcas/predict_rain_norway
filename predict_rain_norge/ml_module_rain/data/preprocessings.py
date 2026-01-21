import xbatcher as xb
import xarray as xr
import torch
import xbatcher.loaders.torch
import dask 
import numpy as np

def sigmoid_soft_label(y, threshold, width=10.0):
    """
    Sigmoid-based soft label with control over width of transition.
    Wider = smoother.
    """
    return 1 / (1 + np.exp(-(y - threshold) / width))

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

### sigmoid for grey zone
def _compute_k(low, high, eps=0.05):
    if high <= low:
        raise ValueError("high must be > low")
    d = (high - low) / 2.0
    return np.log((1 - eps) / eps) / d

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def smooth_rain_soft(x, low, high, k=None, eps=0.05):
    """
    Soft asymptotic sigmoid: values approach 0/1 outside but not exact.
    x can be scalar or array.
    """
    m = 0.5*(low + high)
    if k is None:
        k = _compute_k(low, high, eps=eps)
    return sigmoid(k * (np.asarray(x) - m))

def smooth_rain_rescaled(x, low, high, k=None, eps=0.05):
    """
    Rescaled & clipped: exactly 0 at x=low, exactly 1 at x=high,
    smooth in between, clipped outside [low,high].
    """
    x = np.asarray(x)
    m = 0.5*(low + high)
    if k is None:
        k = _compute_k(low, high, eps=eps)
    raw = sigmoid(k * (x - m))
    raw_low = sigmoid(-k * (high - low) / 2.0)  # = eps by design
    raw_high = sigmoid( k * (high - low) / 2.0)  # = 1-eps by design
    # rescale to [0,1] between low and high, then clip outside:
    y = (raw - raw_low) / (raw_high - raw_low)
    return np.clip(y, 0.0, 1.0)


# true preprocess
def preprocess_rain(ds_rain, config):
    type_predictions = config['type_prediction']
    quantile_extreme = config['quantile_extreme']
    quantile_extreme_based_on_rainy_days = config['quantile_extreme_based_on_rainy_days']
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
            return ((ds_rain > quantile_extreme_rain)*1).astype(float)
        case 'boolean_smooth':
            if 'quantile_lower_grey_zone' in config:
                quantile_lower_grey_zone = config['quantile_lower_grey_zone']
            else:
                quantile_lower_grey_zone = 0.75
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme,[ 'time', 'timestep']).values
                quantile_extreme_rain_grey =  ds_rain.where(ds_rain>1).quantile(quantile_lower_grey_zone,[ 'time', 'timestep']).values
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme,[ 'time', 'timestep']).values
                quantile_extreme_rain_grey =  ds_rain.quantile(quantile_lower_grey_zone,[ 'time', 'timestep']).values
            rain_preproc = smooth_rain_rescaled(ds_rain, quantile_extreme_rain_grey, quantile_extreme_rain)
            rain_preproc = xr.DataArray(rain_preproc, dims=ds_rain.dims, coords=ds_rain.coords)
            return rain_preproc
        case 'boolean_smooth_regional':
            fraction_lower_grey_zone_region = config.get('fraction_lower_grey_zone_region', 0.)
            fraction_extreme_region = config.get('fraction_extreme_region', 0.2)
            quantile_extreme = config.get('quantile_extreme', 0.95)
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme,[ 'time', 'timestep']).values
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme,[ 'time', 'timestep']).values
            quantile_extreme_rain =  ds_rain.quantile(quantile_extreme,[ 'time']).values
            pixels_above_quantile = (ds_rain>quantile_extreme_rain).sum(['x','y'])
            count_pixels = ds_rain.isel(time=0).count(['x','y']).values.flatten()[0]
            rain_preproc = smooth_rain_rescaled(pixels_above_quantile, fraction_lower_grey_zone_region*count_pixels, fraction_extreme_region*count_pixels)
            rain_preproc = xr.DataArray(rain_preproc, dims=pixels_above_quantile.dims, coords=pixels_above_quantile.coords)
            return rain_preproc, pixels_above_quantile
        case 'three_classes':
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme, 'time')
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme, 'time')
            no_rain = xr.ones_like(ds_rain).where(ds_rain>1,0)
            return no_rain.where(ds_rain<quantile_extreme_rain,2).astype(int)
        case 'three_classes_grey_zone':
            if 'quantile_lower_grey_zone' in config:
                quantile_lower_grey_zone = config['quantile_lower_grey_zone']
            else:
                quantile_lower_grey_zone = 0.75
            if quantile_extreme_based_on_rainy_days:
                quantile_extreme_rain =  ds_rain.where(ds_rain>1).quantile(quantile_extreme, 'time')
                quantile_mid_rain = ds_rain.where(ds_rain>1).quantile(quantile_lower_grey_zone, 'time')
            else:
                quantile_extreme_rain =  ds_rain.quantile(quantile_extreme, 'time')
                quantile_mid_rain = ds_rain.quantile(quantile_lower_grey_zone, 'time')
            no_extreme = xr.ones_like(ds_rain).where(ds_rain>quantile_mid_rain,0)
            return no_extreme.where(ds_rain<quantile_extreme_rain*1,2).astype(int)
        case _:
            raise ValueError(f"type_predictions {type_predictions} must be 'boolean', 'three_classes', 'quantiles', 'boolean_smooth','boolean_smooth_regional' or 'regression'")

def get_loader_from_ds(ds, batch_size):
    X_bgen = xb.BatchGenerator(
        ds.features,
        input_dims={'time': batch_size, 'var_name': ds.var_name.size, 'x': ds.features.x.size, 'y': ds.features.y.size},
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