import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_DIR = '/Data/gfi/users/rogui7909/code/CESM2_rain_prediction'

def get_input_data(input_variable, region_predicted, type_prediction, batch_size):
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

print('Data ready')
