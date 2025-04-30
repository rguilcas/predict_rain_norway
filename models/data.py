import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_input_data_from_wandb_logger(wandb_logger):
    train_loader, valid_loader, test_loader, ds_test = get_input_data(
        input_variable=wandb_logger.experiment.config['input_variable'], 
        batch_size=wandb_logger.experiment.config['batch_size'],
        lags_between_predictor_and_predictants=wandb_logger.experiment.config['lags_between_predictor_and_predictants'],
        n_days_predictant=wandb_logger.experiment.config['num_days_predictant'], 
        n_days_predicted=wandb_logger.experiment.config['num_days_predicted'],
        file_name_data_in = wandb_logger.experiment.config['file_name_data_in'],
        file_name_data_out = wandb_logger.experiment.config['file_name_data_out'],
        )
    wandb_logger.experiment.config['image_size'] = ds_test.longitude.size * ds_test.latitude.size

    print(f'Data ready:')
    print(f"    Image size: {wandb_logger.experiment.config['image_size']} ({ds_test.longitude.size}x{ds_test.latitude.size})")
    print(f"    Input data: " + ', '.join(wandb_logger.experiment.config['input_variable']))
    print(f"    {wandb_logger.experiment.config['num_days_predictant']} Past days used for prediction")
    print(f"    {wandb_logger.experiment.config['num_days_predicted']} Predicted days for future rainfall")

    return train_loader, valid_loader, test_loader, ds_test

def get_input_data(input_variable, batch_size=64, 
                   lags_between_predictor_and_predictants = 0,
                   file_name_data_in = '/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_NH_32x128.nc',
                   file_name_data_out = '/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc',
                   n_days_predictant = 3,
                   region_predicted=14, n_days_predicted = 10,
                   trainval_test_split_ratio = 0.8,
                   train_val_split_ratio = 0.8,
                   ):
    """
    Daily data required
    """
    with xr.open_dataset(file_name_data_in) as data_in:
        ds_in = data_in.data_normed.load().sel(var_name=input_variable)

    
    all_shifts = []
    for k in range(n_days_predictant):
        ds_shift = ds_in.shift(time=k)
        ds_shift = ds_shift.assign_coords(lag_in_past=-k)
        all_shifts.append(ds_shift)
    ds_in = xr.concat(all_shifts[::-1], dim='lag_in_past').transpose('time','lag_in_past','var_name','latitude','longitude')
    ds_in = ds_in.isel(time=slice(n_days_predictant-1, None))

    with xr.open_dataarray(file_name_data_out) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        all_out_shift = []
        for k in range(n_days_predicted):
            data_out_shift = data_out.shift(time=-k).assign_coords(lag_in_future=k)
            all_out_shift.append(data_out_shift)
        ds_out = xr.concat(all_out_shift, dim='lag_in_future')
        if n_days_predicted >1 :
            ds_out = ds_out.isel(time=slice(None, -n_days_predicted+1))

    common_time = ds_in.time.loc[dict(time=ds_in.time.isin(ds_out.time))]
    if len(common_time)==0:
        raise ValueError("No common time between input and ouput datasets")
    ds_in, ds_out = ds_in.sel(time=common_time), ds_out.sel(time=common_time)

    if lags_between_predictor_and_predictants>0:
        ds_in = ds_in.shift(time=lags_between_predictor_and_predictants)
        ds_in = ds_in.isel(time=slice(lags_between_predictor_and_predictants,None))
        ds_out = ds_in.isel(time=slice(lags_between_predictor_and_predictants,None))
        
    sample_size = ds_in.time.size
    sample_size
    trainval_size, test_size = int(sample_size*trainval_test_split_ratio), sample_size - int(sample_size*trainval_test_split_ratio)
    train_size, val_size = int(trainval_size*train_val_split_ratio), trainval_size - int(trainval_size*train_val_split_ratio)
    times_shuffled = np.random.permutation(ds_in.time.values)
    train_times, valid_times, test_times = times_shuffled[:train_size], times_shuffled[train_size:train_size+val_size], times_shuffled[train_size+val_size:]

    train_set = TensorDataset(torch.Tensor(ds_in.sel(time=train_times).values).type(torch.float32), 
                            torch.Tensor(ds_out.sel(time=train_times).values).type(torch.float32).T)
    valid_set = TensorDataset(torch.Tensor(ds_in.sel(time=valid_times).values).type(torch.float32), 
                            torch.Tensor(ds_out.sel(time=valid_times).values).type(torch.float32).T)
    test_set = TensorDataset(torch.Tensor(ds_in.sel(time=test_times).values).type(torch.float32), 
                            torch.Tensor(ds_out.sel(time=test_times).values).type(torch.float32).T)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    ds_test = xr.Dataset(dict(data_in=ds_in.sel(time=test_times),
                            data_out = ds_out.sel(time=test_times)))

    return train_loader, valid_loader, test_loader, ds_test







































# def get_input_data_old(input_variable, batch_size, 
#                    lags_between_predictor_and_predictants = 0,
#                    file_name_data_in = '/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_NH_32x128.nc',
#                    file_name_data_out = '/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc',
#                    LSTM_sequence=True, n_days_predictant = 3):
#     ds_in = xr.open_dataset(file_name_data_in).data_normed.load().sel(var_name=input_variable)
    
#     if LSTM_sequence:
#         all_shifts = []
#         for k in range(n_days_predictant):
#             ds_shift = ds_in.shift(time=k)
#             ds_shift = ds_shift.assign_coords(lag=f"lag{k}" )
#             all_shifts.append(ds_shift)
#         ds_in = xr.concat(all_shifts[::-1], dim='lag').transpose('time','lag','var_name','latitude','longitude')
#     return ds_in
#     # if include_gradients:
#     #     gradients = ds_shift.diff('lag').assign_coords(lag=[f"Grad{k+1}-{k}" for k in range(n_days_predictant-1)][::-1])
#     #     ds_shift_grad = xr.concat([ds_shift, gradients], dim='lag')
#     # else:
#     #     ds_shift_grad = ds_shift
#     ds_shift = ds_shift.transpose('time','lag','var_name','latitude','longitude')
#     values_features = ds_shift.values
#     tensor_in = torch.Tensor(values_features).type(torch.float32)

#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         all_out_shift = []
#         for k in range(n_days_predicted):
#             data_out_shift = data_out.shift(time=-k).assign_coords(lag=f"day+{k}")
#             all_out_shift.append(data_out_shift)
#         data_out_shifts = xr.concat(all_out_shift, dim='lag')
#         data_out_values = data_out_shifts.values
#     tensor_out = torch.Tensor(data_out_values).type(torch.float32).T

#     tensor_in = tensor_in[n_days_predictant:-n_days_predicted]
#     tensor_out = tensor_out[n_days_predictant:-n_days_predicted]
#     if lags!=0:
#         tensor_in = tensor_in[:-lags]
#         tensor_out = tensor_out[lags:]
#     dataset = TensorDataset(tensor_in, tensor_out)

#     # use 20% of data for test
#     trainvalid_set_size = int(len(dataset) * 0.8)
#     test_set_size = len(dataset) - trainvalid_set_size
#     # split the train set into two
#     seed = torch.Generator().manual_seed(42)
#     trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

#     # use 20% of training data for validation
#     train_set_size = int(len(trainvalid_set) * 0.8)
#     valid_set_size = len(trainvalid_set) - train_set_size

#     # split the train set into two
#     seed = torch.Generator().manual_seed(43)
#     train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
#     # if augment_train_set:
#     #     train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.01, augment_factor=1)

#     train_loader = DataLoader(train_set, batch_size=batch_size)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)
#     test_loader = DataLoader(test_set, batch_size=batch_size)

#     ds_test = ds_shift.shift(time=lags).isel(time=test_loader.dataset.indices)

#     return train_loader, valid_loader, test_loader, ds_test
#     pass

# def get_input_data_small(input_variable, region_predicted, type_prediction, batch_size,
#                          augment_train_set = True,
#                         lags=0):

#     ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x32.nc').data_normed.load()
#     variables = {str(ds_in.var_name.values[k]):k for k in range(ds_in.var_name.size)}
#     ds_in = ds_in.sel(var_name=input_variable)
#     input_indices = [variables[k] for k in input_variable]
#     data_in = np.load('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x32.npy')[:,input_indices]
#     tensor_in = torch.Tensor(data_in).type(torch.float32)

#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         data_out = data_out.values
#     tensor_out = torch.Tensor(data_out).type(torch.float32)
#     if type_prediction=='quantiles':
#         tensor_out = tensor_out.type(torch.long)
#     elif type_prediction=='regression':
#         tensor_out = tensor_out.reshape(-1,1)

#     if lags!=0:
#         tensor_in = tensor_in[:-lags]
#         tensor_out = tensor_out[lags:]
#     dataset = TensorDataset(tensor_in, tensor_out)

#     # use 20% of data for test
#     trainvalid_set_size = int(len(dataset) * 0.8)
#     test_set_size = len(dataset) - trainvalid_set_size
#     # split the train set into two
#     seed = torch.Generator().manual_seed(42)
#     trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

#     # use 20% of training data for validation
#     train_set_size = int(len(trainvalid_set) * 0.8)
#     valid_set_size = len(trainvalid_set) - train_set_size

#     # split the train set into two
#     seed = torch.Generator().manual_seed(43)
#     train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
#     if augment_train_set:
#         train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.03, augment_factor=3)

#     train_loader = DataLoader(train_set, batch_size=batch_size)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)
#     test_loader = DataLoader(test_set, batch_size=batch_size)

#     ds_test = ds_in.shift(time=lags).isel(time=test_loader.dataset.indices)

#     return train_loader, valid_loader, test_loader, ds_test


# def get_input_data_small_extended(input_variable, region_predicted, type_prediction, batch_size,
#                          augment_train_set = True,
#                         lags=0):

#     ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64.nc').data_normed.load()
#     variables = {str(ds_in.var_name.values[k]):k for k in range(ds_in.var_name.size)}
#     ds_in = ds_in.sel(var_name=input_variable)
#     input_indices = [variables[k] for k in input_variable]
#     data_in = np.load('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64.npy')[:,input_indices]
#     tensor_in = torch.Tensor(data_in).type(torch.float32)

#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         data_out = data_out.values
#     tensor_out = torch.Tensor(data_out).type(torch.float32)
#     if type_prediction=='quantiles':
#         tensor_out = tensor_out.type(torch.long)
#     elif type_prediction=='regression':
#         tensor_out = tensor_out.reshape(-1,1)

#     if lags!=0:
#         tensor_in = tensor_in[:-lags]
#         tensor_out = tensor_out[lags:]
#     dataset = TensorDataset(tensor_in, tensor_out)

#     # use 20% of data for test
#     trainvalid_set_size = int(len(dataset) * 0.8)
#     test_set_size = len(dataset) - trainvalid_set_size
#     # split the train set into two
#     seed = torch.Generator().manual_seed(42)
#     trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

#     # use 20% of training data for validation
#     train_set_size = int(len(trainvalid_set) * 0.8)
#     valid_set_size = len(trainvalid_set) - train_set_size

#     # split the train set into two
#     seed = torch.Generator().manual_seed(43)
#     train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
#     if augment_train_set:
#         train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.01, augment_factor=1)

#     train_loader = DataLoader(train_set, batch_size=batch_size)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)
#     test_loader = DataLoader(test_set, batch_size=batch_size)

#     ds_test = ds_in.shift(time=lags).isel(time=test_loader.dataset.indices)

#     return train_loader, valid_loader, test_loader, ds_test



# def get_input_data_small_extended_multidays(
#         input_variable, region_predicted, type_prediction, batch_size,
#         augment_train_set = False, n_days_predictant=1, n_days_predicted=1,
#         include_gradients=True,
#         lags=0):
#     ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64.nc').data_normed.load().sel(var_name=input_variable)
#     all_shifts = []
#     for k in range(n_days_predictant):
#         ds_shift = ds_in.shift(time=k)
#         ds_shift = ds_shift.assign_coords(lag=f"lag{k}" )
#         all_shifts.append(ds_shift)
#     ds_shift = xr.concat(all_shifts[::-1], dim='lag')
#     if include_gradients:
#         gradients = ds_shift.diff('lag').assign_coords(lag=[f"Grad{k+1}-{k}" for k in range(n_days_predictant-1)][::-1])
#         ds_shift_grad = xr.concat([ds_shift, gradients], dim='lag')
#     else:
#         ds_shift_grad = ds_shift
#     ds_shift_grad = ds_shift_grad.stack(features=['var_name','lag']).transpose('time','features','latitude','longitude')
#     values_features = ds_shift_grad.values
#     tensor_in = torch.Tensor(values_features).type(torch.float32)

#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         all_out_shift = []
#         for k in range(n_days_predicted):
#             data_out_shift = data_out.shift(time=-k).assign_coords(lag=f"day+{k}")
#             all_out_shift.append(data_out_shift)
#         data_out_shifts = xr.concat(all_out_shift, dim='lag')
#         data_out_values = data_out_shifts.values
#     tensor_out = torch.Tensor(data_out_values).type(torch.float32).T

#     tensor_in = tensor_in[n_days_predictant:-n_days_predicted]
#     tensor_out = tensor_out[n_days_predictant:-n_days_predicted]
#     if lags!=0:
#         tensor_in = tensor_in[:-lags]
#         tensor_out = tensor_out[lags:]
#     dataset = TensorDataset(tensor_in, tensor_out)

#     # use 20% of data for test
#     trainvalid_set_size = int(len(dataset) * 0.8)
#     test_set_size = len(dataset) - trainvalid_set_size
#     # split the train set into two
#     seed = torch.Generator().manual_seed(42)
#     trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

#     # use 20% of training data for validation
#     train_set_size = int(len(trainvalid_set) * 0.8)
#     valid_set_size = len(trainvalid_set) - train_set_size

#     # split the train set into two
#     seed = torch.Generator().manual_seed(43)
#     train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
#     # if augment_train_set:
#     #     train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.01, augment_factor=1)

#     train_loader = DataLoader(train_set, batch_size=batch_size)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)
#     test_loader = DataLoader(test_set, batch_size=batch_size)

#     ds_test = ds_shift_grad.shift(time=lags).isel(time=test_loader.dataset.indices)

#     return train_loader, valid_loader, test_loader, ds_test

# class AugmentedRainfallDataset(Dataset):
#     def __init__(self, base_dataset, threshold=30.0, noise_std=0.02, augment_factor=1):
#         """
#         Custom dataset wrapper to augment extreme rainfall cases.
        
#         Args:
#             base_dataset (Dataset): Original dataset (expects __getitem__ to return (input, target))
#             threshold (float): Rainfall threshold for extreme cases
#             noise_std (float): Standard deviation of Gaussian noise
#             augment_factor (int): How many times to duplicate extreme samples
#         """
#         self.base_dataset = base_dataset
#         self.threshold = threshold
#         self.noise_std = noise_std
#         self.augment_factor = augment_factor
        
#         # Identify indices of extreme rainfall cases
#         self.extreme_indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] >= threshold]
        
#         # Create new dataset indices (original + augmented)
#         self.indices = list(range(len(base_dataset))) + self.extreme_indices * augment_factor
    
#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         original_idx = self.indices[idx]
#         x, y = self.base_dataset[original_idx]

#         # If augmented sample, add Gaussian noise
#         if original_idx in self.extreme_indices:
#             noise = torch.randn_like(x) * self.noise_std
#             x = x + noise  # Add noise only to input, not target

#         return x, y

# def get_input_data_small_for_attribution(input_variable, region_predicted, type_prediction):
   
#     ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64_ageostroph.nc').data_normed.sel(var_name=input_variable)
#     data_in = ds_in.values
#     tensor_in = torch.Tensor(data_in).type(torch.float32)

#     # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         data_out = data_out.values
#         coords = ds_out.coords

#     tensor_out = torch.Tensor(data_out).type(torch.float32)
#     if type_prediction=='quantiles':
#         tensor_out = tensor_out.type(torch.long)
#     elif type_prediction=='regression':
#         tensor_out = tensor_out.reshape(-1,1)

#     dataset = TensorDataset(tensor_in, tensor_out)


#     return dataset, ds_in


# def get_input_data(input_variable, region_predicted, type_prediction, batch_size,
#                    lags=0):
    
#     variable_order = dict(z500=0,u850=1,v850=2,tcwv=3,pr=4)
#     input_variable_index = [variable_order[variable] for variable in input_variable]

#     data = np.load( "/Data/gfi/users/rogui7909/data/ERA5/numpy_era5_in.npy")
#     data_in = data[:,input_variable_index]
#     tensor_in = torch.Tensor(data_in).type(torch.float32)
#     # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         data_out = data_out.values

#     tensor_out = torch.Tensor(data_out).type(torch.float32)
#     if type_prediction=='quantiles':
#         tensor_out = tensor_out.type(torch.long)
#     elif type_prediction=='regression':
#         tensor_out = tensor_out.reshape(-1,1)

#     if lags!=0:
#         tensor_in = tensor_in[lags:]
#         tensor_out = tensor_out[:-lags]
#     dataset = TensorDataset(tensor_in, tensor_out)

#     # use 20% of data for test
#     trainvalid_set_size = int(len(dataset) * 0.8)
#     test_set_size = len(dataset) - trainvalid_set_size
#     # split the train set into two
#     seed = torch.Generator().manual_seed(42)
#     trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

#     # use 20% of training data for validation
#     train_set_size = int(len(trainvalid_set) * 0.8)
#     valid_set_size = len(trainvalid_set) - train_set_size

#     # split the train set into two
#     seed = torch.Generator().manual_seed(43)
#     train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)


#     train_loader = DataLoader(train_set, batch_size=batch_size)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)
#     test_loader = DataLoader(test_set, batch_size=batch_size)

#     return train_loader, valid_loader, test_loader



# def get_input_data_for_attribution(input_variable, region_predicted, type_prediction, batch_size,):
#     variable_order = dict(z500=0,u850=1,v850=2,tcwv=3,pr=4)
#     input_variable_index = [variable_order[variable] for variable in input_variable]

#     data = np.load( "/Data/gfi/users/rogui7909/data/ERA5/numpy_era5_in.npy")
#     data_in = data[:,input_variable_index]
#     tensor_in = torch.Tensor(data_in).type(torch.float32)
#     # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


#     with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
#         data_out = ds_out.sel(mask_id=region_predicted)
#         if type_prediction=='quantiles':
#             data_out = (data_out.rank('time', pct=True)//.1).astype(int)
#         data_out = data_out.values
#         coords = ds_out.coords

#     tensor_out = torch.Tensor(data_out).type(torch.float32)
#     if type_prediction=='quantiles':
#         tensor_out = tensor_out.type(torch.long)
#     elif type_prediction=='regression':
#         tensor_out = tensor_out.reshape(-1,1)

#     dataset = TensorDataset(tensor_in, tensor_out)


#     return dataset, coords
