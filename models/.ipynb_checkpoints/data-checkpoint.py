import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = '/Data/gfi/users/rogui7909/code/CESM2_rain_prediction'
def load_output_data(region=14):
    ds_rain = xr.open_dataarray(f'{DATA_DIR}/cesm2_amip_pr.nc')*86400
    ds_mask = xr.open_dataset('/Data/gfi/users/jodor4442/rainfall_regions/K39_ERA5land_lat_weighted_2d_direct.nc')
    norge_mask = (ds_mask.tp==region).sel(longitude=slice(0,None)).rename(longitude='lon', latitude='lat')*1
    norge_mask = norge_mask.interp(lon=ds_rain.lon, lat=ds_rain.lat, method='nearest').fillna(0)    
    ds_out_rain = ds_rain.where(norge_mask==1).mean(['lon','lat'])
    return ds_out_rain, norge_mask

def get_rain_bool(ds_out_rain, thresh_mm = 1):
    ds_out_rain = ds_out_rain.where(ds_out_rain>thresh_mm, 0)
    ds_out_rain = ds_out_rain.where(ds_out_rain<thresh_mm, 1)
    return ds_out_rain
    
def load_input_data(normalized = True):
    ds_in1 = xr.open_dataarray(f'{DATA_DIR}/cesm2_amip_u850_NH.nc')#.sel(time=ds_out.time)
    ds_in2 = xr.open_dataarray(f'{DATA_DIR}/cesm2_amip_v850_NH.nc')#.sel(time=ds_out.time)
    ds_in3 = xr.open_dataarray(f'{DATA_DIR}/cesm2_amip_z500_NH.nc')#.sel(time=ds_out.time)
    if normalized:
        ds_in1 = ((ds_in1-ds_in1.mean())/ds_in1.std()).fillna(0)
        ds_in2 = ((ds_in2-ds_in2.mean())/ds_in2.std()).fillna(0)
        ds_in3 = ((ds_in3-ds_in3.mean())/ds_in3.std()).fillna(0)
    ds_in = xr.concat([ds_in1.assign_coords(var_name='u850'), 
                       ds_in2.assign_coords(var_name='v850'), 
                       ds_in3.assign_coords(var_name='z500')], dim='var_name')
    return ds_in


def xarray_to_dataloaders(ds_in, ds_out, batch_size, split=0.8):
    """
    split in fraction betwwen 0 and 1
    """
    indices = np.arange(ds_out.size)
    np.random.shuffle(indices)
    split = int(split*ds_out.size)
    indices_train, indices_test = indices[:split], indices[split:]
    data_in = ds_in.values.transpose(1,0,2,3)
    data_out = ds_out.values


    X_train = torch.tensor(data_in[indices_train]).type(torch.float32)
    X_test = torch.tensor(data_in[indices_test]).type(torch.float32)
    y_train = torch.tensor(data_out[indices_train]).type(torch.LongTensor)
    y_test = torch.tensor(data_out[indices_test]).type(torch.LongTensor)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, indices_train, indices_test

