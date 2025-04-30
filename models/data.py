import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset

DATA_DIR = '/Data/gfi/users/rogui7909/code/CESM2_rain_prediction'



def get_input_data_small(input_variable, region_predicted, type_prediction, batch_size,
                         augment_train_set = True,
                        lags=0):

    ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x32.nc').data_normed.load()
    variables = {str(ds_in.var_name.values[k]):k for k in range(ds_in.var_name.size)}
    ds_in = ds_in.sel(var_name=input_variable)
    input_indices = [variables[k] for k in input_variable]
    data_in = np.load('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x32.npy')[:,input_indices]
    tensor_in = torch.Tensor(data_in).type(torch.float32)

    with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        if type_prediction=='quantiles':
            data_out = (data_out.rank('time', pct=True)//.1).astype(int)
        data_out = data_out.values
    tensor_out = torch.Tensor(data_out).type(torch.float32)
    if type_prediction=='quantiles':
        tensor_out = tensor_out.type(torch.long)
    elif type_prediction=='regression':
        tensor_out = tensor_out.reshape(-1,1)

    if lags!=0:
        tensor_in = tensor_in[:-lags]
        tensor_out = tensor_out[lags:]
    dataset = TensorDataset(tensor_in, tensor_out)

    # use 20% of data for test
    trainvalid_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - trainvalid_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

    # use 20% of training data for validation
    train_set_size = int(len(trainvalid_set) * 0.8)
    valid_set_size = len(trainvalid_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(43)
    train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
    if augment_train_set:
        train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.03, augment_factor=3)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    ds_test = ds_in.shift(time=lags).isel(time=test_loader.dataset.indices)

    return train_loader, valid_loader, test_loader, ds_test


def get_input_data_small_extended(input_variable, region_predicted, type_prediction, batch_size,
                         augment_train_set = True,
                        lags=0):

    ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64.nc').data_normed.load()
    variables = {str(ds_in.var_name.values[k]):k for k in range(ds_in.var_name.size)}
    ds_in = ds_in.sel(var_name=input_variable)
    input_indices = [variables[k] for k in input_variable]
    data_in = np.load('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64.npy')[:,input_indices]
    tensor_in = torch.Tensor(data_in).type(torch.float32)

    with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20, var_name=0)) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        if type_prediction=='quantiles':
            data_out = (data_out.rank('time', pct=True)//.1).astype(int)
        data_out = data_out.values
    tensor_out = torch.Tensor(data_out).type(torch.float32)
    if type_prediction=='quantiles':
        tensor_out = tensor_out.type(torch.long)
    elif type_prediction=='regression':
        tensor_out = tensor_out.reshape(-1,1)

    if lags!=0:
        tensor_in = tensor_in[:-lags]
        tensor_out = tensor_out[lags:]
    dataset = TensorDataset(tensor_in, tensor_out)

    # use 20% of data for test
    trainvalid_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - trainvalid_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

    # use 20% of training data for validation
    train_set_size = int(len(trainvalid_set) * 0.8)
    valid_set_size = len(trainvalid_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(43)
    train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)
    if augment_train_set:
        train_set = AugmentedRainfallDataset(train_set, threshold=30.0, noise_std=0.01, augment_factor=1)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    ds_test = ds_in.shift(time=lags).isel(time=test_loader.dataset.indices)

    return train_loader, valid_loader, test_loader, ds_test

class AugmentedRainfallDataset(Dataset):
    def __init__(self, base_dataset, threshold=30.0, noise_std=0.02, augment_factor=1):
        """
        Custom dataset wrapper to augment extreme rainfall cases.
        
        Args:
            base_dataset (Dataset): Original dataset (expects __getitem__ to return (input, target))
            threshold (float): Rainfall threshold for extreme cases
            noise_std (float): Standard deviation of Gaussian noise
            augment_factor (int): How many times to duplicate extreme samples
        """
        self.base_dataset = base_dataset
        self.threshold = threshold
        self.noise_std = noise_std
        self.augment_factor = augment_factor
        
        # Identify indices of extreme rainfall cases
        self.extreme_indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] >= threshold]
        
        # Create new dataset indices (original + augmented)
        self.indices = list(range(len(base_dataset))) + self.extreme_indices * augment_factor
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, y = self.base_dataset[original_idx]

        # If augmented sample, add Gaussian noise
        if original_idx in self.extreme_indices:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise  # Add noise only to input, not target

        return x, y

def get_input_data_small_for_attribution(input_variable, region_predicted, type_prediction):
   
    ds_in = xr.open_dataset('/Data/gfi/users/rogui7909/data/ERA5/dl_inputs_32x64_ageostroph.nc').data_normed.sel(var_name=input_variable)
    data_in = ds_in.values
    tensor_in = torch.Tensor(data_in).type(torch.float32)

    # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


    with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        if type_prediction=='quantiles':
            data_out = (data_out.rank('time', pct=True)//.1).astype(int)
        data_out = data_out.values
        coords = ds_out.coords

    tensor_out = torch.Tensor(data_out).type(torch.float32)
    if type_prediction=='quantiles':
        tensor_out = tensor_out.type(torch.long)
    elif type_prediction=='regression':
        tensor_out = tensor_out.reshape(-1,1)

    dataset = TensorDataset(tensor_in, tensor_out)


    return dataset, ds_in


def get_input_data(input_variable, region_predicted, type_prediction, batch_size,
                   lags=0):
    
    variable_order = dict(z500=0,u850=1,v850=2,tcwv=3,pr=4)
    input_variable_index = [variable_order[variable] for variable in input_variable]

    data = np.load( "/Data/gfi/users/rogui7909/data/ERA5/numpy_era5_in.npy")
    data_in = data[:,input_variable_index]
    tensor_in = torch.Tensor(data_in).type(torch.float32)
    # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


    with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        if type_prediction=='quantiles':
            data_out = (data_out.rank('time', pct=True)//.1).astype(int)
        data_out = data_out.values

    tensor_out = torch.Tensor(data_out).type(torch.float32)
    if type_prediction=='quantiles':
        tensor_out = tensor_out.type(torch.long)
    elif type_prediction=='regression':
        tensor_out = tensor_out.reshape(-1,1)

    if lags!=0:
        tensor_in = tensor_in[lags:]
        tensor_out = tensor_out[:-lags]
    dataset = TensorDataset(tensor_in, tensor_out)

    # use 20% of data for test
    trainvalid_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - trainvalid_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

    # use 20% of training data for validation
    train_set_size = int(len(trainvalid_set) * 0.8)
    valid_set_size = len(trainvalid_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(43)
    train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)


    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, valid_loader, test_loader



def get_input_data_for_attribution(input_variable, region_predicted, type_prediction, batch_size,):
    variable_order = dict(z500=0,u850=1,v850=2,tcwv=3,pr=4)
    input_variable_index = [variable_order[variable] for variable in input_variable]

    data = np.load( "/Data/gfi/users/rogui7909/data/ERA5/numpy_era5_in.npy")
    data_in = data[:,input_variable_index]
    tensor_in = torch.Tensor(data_in).type(torch.float32)
    # tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


    with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
        data_out = ds_out.sel(mask_id=region_predicted)
        if type_prediction=='quantiles':
            data_out = (data_out.rank('time', pct=True)//.1).astype(int)
        data_out = data_out.values
        coords = ds_out.coords

    tensor_out = torch.Tensor(data_out).type(torch.float32)
    if type_prediction=='quantiles':
        tensor_out = tensor_out.type(torch.long)
    elif type_prediction=='regression':
        tensor_out = tensor_out.reshape(-1,1)

    dataset = TensorDataset(tensor_in, tensor_out)


    return dataset, coords
