import xarray as xr
import torch
from torch.utils.data import Subset
import xbatcher as xb 
import xbatcher.loaders.torch
import dask
import numpy as np



def get_input_data_from_wandb_logger_three_types(wandb_logger, quantile = .9,load=True, lon_lim = (None,None),
                                                 add_noise = False, noisy_samples = 10,noise_scale=1,train_val_test_ratio=[.6,.2],
                                                 shuffle_before_xbatcher = True):
    config = wandb_logger.experiment.config
    batch_size = config['batch_size']
    num_timesteps_predicted = config['num_timesteps_predicted']
    output_path = config['file_name_data_out']
    input_path = config['file_name_data_in']
    input_variables = config['inputs'].split(' ')
    config['input_variables'] = input_variables

    season = config['season']

    ds_rain = get_ds_rain(output_path, num_timesteps_predicted, season)
    no_rain = xr.ones_like(ds_rain).where(ds_rain>1,0)
    q90 = ds_rain.where(no_rain==1).quantile(quantile).values
    ds_out = no_rain.where(ds_rain<q90,2).drop_vars(['longitude','latitude']).astype(int)

    ds_out.attrs['quantile'] = quantile

    # ds_out = xr.ones_like(ds_rain).where(ds_rain>ds_rain.quantile(quantile_thresh),0).drop_vars(['longitude','latitude'])
    lon_min, lon_max = lon_lim
    ds_in = xr.open_zarr(input_path).sel(time=ds_out.time).data_normed.sel(var_name = input_variables).sel(time=ds_out.time).sel(longitude=slice(lon_min, lon_max))
    if load:
        ds_in=ds_in.load()

    ds = xr.Dataset(dict(inputs=ds_in, targets=ds_out, rain=ds_rain))

    indices_train, indices_val, indices_test = get_train_val_test_split(ds_rain, ratio=train_val_test_ratio, shuffle=shuffle_before_xbatcher)

    ds_train = ds.isel(time=indices_train)
    ds_val = ds.isel(time=indices_val)
    ds_test = ds.isel(time=indices_test)

    if add_noise:
        ds_train = add_noise_ds(ds_train, noisy_samples=noisy_samples, noise_scale=noise_scale)
        # ds_val = add_noise_ds(ds_val, noisy_samples=noisy_samples, noise_scale=noise_scale)
        # ds_test = add_noise_ds(ds_test, noisy_samples=noisy_samples, noise_scale=noise_scale)
        
    train_loader = get_loader_from_ds(ds_train, batch_size=batch_size)
    val_loader = get_loader_from_ds(ds_val, batch_size=batch_size)
    test_loader = get_loader_from_ds(ds_test, batch_size=batch_size)


    wandb_logger.experiment.config['image_size'] = ds_val.longitude.size * ds_val.latitude.size

    print_info(wandb_logger, ds_val)
    return train_loader, val_loader, test_loader, ds_val


def get_ds_rain(output_path, num_timesteps_predicted, season):

    ds_rain = xr.open_dataset(output_path).tp
    if num_timesteps_predicted>1:
        ds_rain = ds_rain.rolling(time=num_timesteps_predicted).construct('timestep').shift(time=-num_timesteps_predicted+1)[:-num_timesteps_predicted+1]
    else:
        ds_rain = ds_rain.expand_dims('timestep').assign_coords(timestep=[0]).transpose('time','timestep')
    if season =='all':
        pass
    else:
        ds_rain = ds_rain.where(ds_rain.time.dt.season==season, drop=True)
        if ds_rain.time.size == 0:
            raise ValueError(f"No data in season {season}.")
    return ds_rain


def add_noise_dataarray(dataarray, noisy_samples=10, noise_scale=.3):
    expanded_dataarray = dataarray.expand_dims(noise=noisy_samples)
    noise_dataarray = xr.DataArray(dask.array.random.normal(size=expanded_dataarray.shape, loc=0, scale=noise_scale), coords=expanded_dataarray.coords, dims=expanded_dataarray.dims)
    noisy_dataarray = xr.concat([dataarray.expand_dims(noise=1), expanded_dataarray+noise_dataarray], dim='noise')
    dataarray = noisy_dataarray.rename(time='true_time').stack(time=['true_time','noise']).astype('float32').transpose('time','var_name','latitude','longitude')
    return dataarray

def extend_noise(dataarray, noisy_samples=10,):
    expanded_dataarray = dataarray.expand_dims(noise=noisy_samples)
    noisy_dataarray = xr.concat([dataarray.expand_dims(noise=1), expanded_dataarray], dim='noise')
    dataarray = noisy_dataarray.rename(time='true_time').stack(time=['true_time','noise']).astype('int').transpose('time','timestep')
    return dataarray

def add_noise_ds(ds, noisy_samples=10, noise_scale=.3, shuffle_after_noise=True):
    inputs = add_noise_dataarray(ds['inputs'], noisy_samples=noisy_samples, noise_scale=noise_scale)
    targets = extend_noise(ds['targets'], noisy_samples=noisy_samples)
    rain = extend_noise(ds['rain'], noisy_samples=noisy_samples)
    ds = xr.Dataset(dict(inputs=inputs, targets=targets, rain=rain))
    if shuffle_after_noise:
        indices = np.arange(ds.time.size)
        np.random.shuffle(indices)
        ds = ds.isel(time=indices)
    return ds

def get_loader_from_ds(ds, batch_size):
    X_bgen = xb.BatchGenerator(
        ds.inputs,
        input_dims={'time': batch_size, 'var_name': ds.var_name.size, 'latitude': ds.inputs.latitude.size, 'longitude': ds.inputs.longitude.size},
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

def get_train_val_test_split(ds_rain, ratio=[.6,.2], shuffle=True):
    all_indices = np.arange(ds_rain.time.size)
    total_size = all_indices.size
    indices_train = all_indices[:int(total_size*ratio[0])]
    indices_val = all_indices[int(total_size*ratio[0]):int(total_size*(ratio[0]+ratio[1]))]
    indices_test = all_indices[int(total_size*(ratio[0]+ratio[1])):] 
    if shuffle:
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)
    return indices_train, indices_val, indices_test

def print_info(wandb_logger, ds):
    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds.longitude.size}x{ds.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")





def get_input_data_from_wandb_logger(wandb_logger,load=True, lon_lim = (None,None),
                                     add_noise = False, noisy_sample = 10,noise_scale=1,):
    config = wandb_logger.experiment.config
    batch_size = config['batch_size']
    num_timesteps_predicted = config['num_timesteps_predicted']
    output_path = config['file_name_data_out']
    input_path = config['file_name_data_in']
    input_variables = config['inputs'].split(' ')
    config['input_variables'] = input_variables

    quantile_thresh = config['quantile_thresh']
    season = config['season']

    ds_rain = xr.open_dataset(output_path).tp
    if num_timesteps_predicted>1:
        ds_rain = ds_rain.rolling(time=num_timesteps_predicted).construct('timestep').shift(time=-num_timesteps_predicted+1)[:-num_timesteps_predicted+1]
    else:
        ds_rain = ds_rain.expand_dims('timestep').assign_coords(timestep=[0]).transpose('time','timestep')
    if season =='all':
        pass
    else:
        ds_rain = ds_rain.where(ds_rain.time.dt.season==season, drop=True)
        if ds_rain.time.size == 0:
            raise ValueError(f"No data in season {season}.")
    ds_out = xr.ones_like(ds_rain).where(ds_rain>ds_rain.quantile(quantile_thresh),0).drop_vars(['longitude','latitude'])

    ds_in = xr.open_zarr(input_path).sel(time=ds_out.time).data_normed.sel(var_name = input_variables).sel(time=ds_out.time)
    if load:
        ds_in=ds_in.load()
    lon_min, lon_max = lon_lim
    ds_in = ds_in.sel(longitude=slice(lon_min, lon_max))
    if add_noise:
        expanded_ds_in = ds_in.expand_dims(noise=noisy_sample)
        noise_ds_in = xr.DataArray(dask.array.random.normal(size=expanded_ds_in.shape, loc=0, scale=noise_scale), coords=expanded_ds_in.coords, dims=expanded_ds_in.dims)
        noisy_ds_in = xr.concat([ds_in.expand_dims(noise=1), expanded_ds_in+noise_ds_in], dim='noise')
        ds_in = noisy_ds_in.rename(time='true_time').stack(time=['true_time','noise']).astype('float32').transpose('time','var_name','latitude','longitude')

        expanded_ds_out = ds_out.expand_dims(noise=noisy_sample)
        noisy_ds_out = xr.concat([ds_out.expand_dims(noise=1), expanded_ds_out], dim='noise')
        ds_out = noisy_ds_out.rename(time='true_time').stack(time=['true_time','noise']).astype('int')

        expanded_ds_rain = ds_rain.expand_dims(noise=noisy_sample)
        noisy_ds_rain = xr.concat([ds_rain.expand_dims(noise=1), expanded_ds_rain], dim='noise')
        ds_rain = noisy_ds_rain.rename(time='true_time').stack(time=['true_time','noise'])

    X_bgen = xb.BatchGenerator(
        ds_in,
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': ds_in.latitude.size, 'longitude': ds_in.longitude.size},
        preload_batch=True,
    )
    y_bgen = xb.BatchGenerator(
        ds_out,
        input_dims={'time': batch_size, 'timestep': num_timesteps_predicted },
        preload_batch=True,
    )

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size 

    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))
    test_set = Subset(dataset, list(range(train_size + val_size, total_size)))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=4,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )

    test_indices = list(range((train_size + val_size)*batch_size, total_size*batch_size))
    val_indices = list(range(train_size*batch_size, (train_size + val_size)*batch_size))
    ds_val = xr.Dataset(dict(inputs=ds_in.isel(time=val_indices), 
                              targets = ds_out.isel(time=val_indices), 
                              rain = ds_rain.isel(time=val_indices)))

    wandb_logger.experiment.config['image_size'] = ds_val.longitude.size * ds_val.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_val.longitude.size}x{ds_val.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_val



def get_input_data_from_wandb_logger_deciles(wandb_logger,load=True):
    config = wandb_logger.experiment.config
    batch_size = config['batch_size']
    num_timesteps_predicted = config['num_timesteps_predicted']
    output_path = config['file_name_data_out']
    input_path = config['file_name_data_in']
    input_variables = config['inputs'].split(' ')
    config['input_variables'] = input_variables

    quantile_thresh = config['quantile_thresh']
    season = config['season']

    ds_rain = xr.open_dataset(output_path).tp
    # if num_timesteps_predicted>1:
    #     ds_rain = ds_rain.rolling(time=num_timesteps_predicted).construct('timestep').shift(time=-num_timesteps_predicted+1)[:-num_timesteps_predicted+1]
    # else:
    #     ds_rain = ds_rain.expand_dims('timestep').assign_coords(timestep=[0]).transpose('time','timestep')
    if season =='all':
        pass
    else:
        ds_rain = ds_rain.where(ds_rain.time.dt.season==season, drop=True)
        if ds_rain.time.size == 0:
            raise ValueError(f"No data in season {season}.")
    ds_out = ds_rain.rank('time', pct=True)
    ds_out = (ds_out*100)//10
    ds_out = ds_out.where(ds_out<10,9).drop_vars(['longitude','latitude']).astype(int)
    # ds_out = xr.ones_like(ds_rain).where(ds_rain>ds_rain.quantile(quantile_thresh),0).drop_vars(['longitude','latitude'])
    ds_in = xr.open_zarr(input_path).sel(time=ds_out.time).data_normed.sel(var_name = input_variables).sel(time=ds_out.time)
    if load:
        ds_in=ds_in.load()
    X_bgen = xb.BatchGenerator(
        ds_in,
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': ds_in.latitude.size, 'longitude': ds_in.longitude.size},
        preload_batch=True,
    )
    y_bgen = xb.BatchGenerator(
        ds_out,
        input_dims={'time': batch_size},
        preload_batch=True,
    )

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size 

    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))
    test_set = Subset(dataset, list(range(train_size + val_size, total_size)))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=4,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )

    test_indices = list(range((train_size + val_size)*batch_size, total_size*batch_size))
    val_indices = list(range(train_size*batch_size, (train_size + val_size)*batch_size))
    ds_val = xr.Dataset(dict(inputs=ds_in.isel(time=val_indices), 
                              targets = ds_out.isel(time=val_indices), 
                              rain = ds_rain.isel(time=val_indices)))

    wandb_logger.experiment.config['image_size'] = ds_val.longitude.size * ds_val.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_val.longitude.size}x{ds_val.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_val



def get_input_data_from_wandb_logger_quantiles(wandb_logger,quantiles = [0,.5,.75,.9,.95],load=True):
    config = wandb_logger.experiment.config
    batch_size = config['batch_size']
    num_timesteps_predicted = config['num_timesteps_predicted']
    output_path = config['file_name_data_out']
    input_path = config['file_name_data_in']
    input_variables = config['inputs'].split(' ')
    config['input_variables'] = input_variables

    season = config['season']

    ds_rain = xr.open_dataset(output_path).tp
    if num_timesteps_predicted>1:
        ds_rain = ds_rain.rolling(time=num_timesteps_predicted).construct('timestep').shift(time=-num_timesteps_predicted+1)[:-num_timesteps_predicted+1]
    else:
        ds_rain = ds_rain.expand_dims('timestep').assign_coords(timestep=[0]).transpose('time','timestep')
    if season =='all':
        pass
    else:
        ds_rain = ds_rain.where(ds_rain.time.dt.season==season, drop=True)
        if ds_rain.time.size == 0:
            raise ValueError(f"No data in season {season}.")
    ds_out = ds_rain.rank('time', pct=True)
    # quantiles = [.5,.75,.9,.95, 1]
    quantiles_da = xr.DataArray(quantiles, dims='quantiles', coords=dict(quantiles = quantiles))
    ds_out = (ds_out>quantiles_da).sum("quantiles").drop_vars(['longitude','latitude']).astype(int)
    ds_out.attrs['quantile'] = quantiles 

    # ds_out = xr.ones_like(ds_rain).where(ds_rain>ds_rain.quantile(quantile_thresh),0).drop_vars(['longitude','latitude'])
    ds_in = xr.open_zarr(input_path).sel(time=ds_out.time).data_normed.sel(var_name = input_variables).sel(time=ds_out.time)
    if load:
        ds_in=ds_in.load()
    X_bgen = xb.BatchGenerator(
        ds_in,
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': ds_in.longitude.size, 'longitude': ds_in.longitude.size},
        preload_batch=True,
    )
    y_bgen = xb.BatchGenerator(
        ds_out,
        input_dims={'time': batch_size},
        preload_batch=True,
    )

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size 

    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))
    test_set = Subset(dataset, list(range(train_size + val_size, total_size)))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=1,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        prefetch_factor=3,  # Prefetch up to 3 batches in advance to reduce data loading latency
        num_workers=4,  # Use 4 parallel worker processes to load data concurrently
        persistent_workers=True,  # Keep workers alive between epochs for faster subsequent epochs
        multiprocessing_context='forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
    )
    val_indices = list(range(train_size*batch_size, (train_size + val_size)*batch_size))
    test_indices = list(range((train_size + val_size)*batch_size, total_size*batch_size))
    ds_val = xr.Dataset(dict(inputs=ds_in.isel(time=val_indices), 
                             targets = ds_out.isel(time=val_indices), 
                             rain = ds_rain.isel(time=val_indices)))

    wandb_logger.experiment.config['image_size'] = ds_val.longitude.size * ds_val.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_val.longitude.size}x{ds_val.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_val


