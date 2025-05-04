import xarray as xr
import torch
from torch.utils.data import Subset
import xbatcher as xb 
import xbatcher.loaders.torch


def get_input_data_from_wandb_logger(wandb_logger,load=True):
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
    X_bgen = xb.BatchGenerator(
        ds_in,
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': 32, 'longitude': 128},
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
    ds_test = xr.Dataset(dict(inputs=ds_in.isel(time=test_indices), 
                              targets = ds_out.isel(time=test_indices), 
                              rain = ds_rain.isel(time=test_indices)))

    wandb_logger.experiment.config['image_size'] = ds_test.longitude.size * ds_test.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_test.longitude.size}x{ds_test.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_test



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
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': 32, 'longitude': 128},
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
    ds_test = xr.Dataset(dict(inputs=ds_in.isel(time=test_indices), 
                              targets = ds_out.isel(time=test_indices), 
                              rain = ds_rain.isel(time=test_indices)))

    wandb_logger.experiment.config['image_size'] = ds_test.longitude.size * ds_test.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_test.longitude.size}x{ds_test.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_test



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
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': 32, 'longitude': 128},
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
    ds_test = xr.Dataset(dict(inputs=ds_in.isel(time=test_indices), 
                              targets = ds_out.isel(time=test_indices), 
                              rain = ds_rain.isel(time=test_indices)))

    wandb_logger.experiment.config['image_size'] = ds_test.longitude.size * ds_test.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_test.longitude.size}x{ds_test.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_test


def get_input_data_from_wandb_logger_three_types(wandb_logger,quantile = .9,load=True):
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
    no_rain = xr.ones_like(ds_rain).where(ds_rain>1,0)
    q90 = ds_rain.where(no_rain==1).quantile(quantile).values
    ds_out = no_rain.where(ds_rain<q90,2).drop_vars(['longitude','latitude']).astype(int)
    
    ds_out.attrs['quantile'] = quantile

    # ds_out = xr.ones_like(ds_rain).where(ds_rain>ds_rain.quantile(quantile_thresh),0).drop_vars(['longitude','latitude'])
    ds_in = xr.open_zarr(input_path).sel(time=ds_out.time).data_normed.sel(var_name = input_variables).sel(time=ds_out.time)
    if load:
        ds_in=ds_in.load()
    X_bgen = xb.BatchGenerator(
        ds_in,
        input_dims={'time': batch_size, 'var_name': len(input_variables), 'latitude': 32, 'longitude': 128},
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
    ds_test = xr.Dataset(dict(inputs=ds_in.isel(time=test_indices), 
                              targets = ds_out.isel(time=test_indices), 
                              rain = ds_rain.isel(time=test_indices)))

    wandb_logger.experiment.config['image_size'] = ds_test.longitude.size * ds_test.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_test.longitude.size}x{ds_test.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variables']))
    print(f"    {wandb_logger.experiment.config['num_timesteps_predicted']} Predicted timesteps for future rainfall")
    print(f"    {wandb_logger.experiment.config['quantile_thresh']*100:.0f}th percentile predicted")
    


    return train_loader, val_loader, test_loader, ds_test