print('Running imports...')
import os
from captum.attr import IntegratedGradients

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch # long
from lightning.pytorch import seed_everything
import xarray as xr
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from ml_module_rain.models.trained_models import load_trained_model

def change_time_of_prediction_to_time_of_event(ds_in):
    steps = ds_in.timestep_future.size
    ds_out = xr.concat([ds_in.isel(timestep_future=k).shift(time_of_prediction=-(steps-k-1)) for k in range(ds_in.timestep_future.size)], dim='timestep_future')
    ds_out = ds_out.rename(time_of_prediction='time_of_event', timestep_future='timestep_past')
    ds_out = ds_out.assign_coords(timestep_past = np.arange(0,-ds_out.timestep_past.size,-1))
    ds_out = ds_out.sel(timestep_past=np.arange(-ds_out.timestep_past.size+1,1))
    return ds_out

def get_ds_with_predictions(dataloader, lightning_model):
    if torch.cuda.is_available():
        device="cuda"
    else:
        device=='cpu'
    lightning_model.to(device)
    lightning_model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in tqdm(dataloader.val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = torch.sigmoid(lightning_model(x))
            all_preds.append(preds)
            all_targets.append(y)

    # Concatenate on GPU, then move to CPU once
    preds = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()

    ds_aligned = dataloader.ds_val.isel(time=range(targets.shape[0]))
    ds_aligned = ds_aligned.rename(time='time_of_prediction', timestep ='timestep_future')
    preds_da = xr.DataArray(preds, dims=["time_of_prediction", "timestep_future"], coords=ds_aligned.targets.coords)
    ds_aligned["predictions"] = preds_da
    
    # ds_out = change_time_of_prediction_to_time_of_event(ds_aligned[['predictions','targets','rain']])
    # ds_out_valid_times = ds_out.where(ds_out.targets.count('timestep_past') == ds_out.targets.count('timestep_past').max(), drop=True)
    # return ds_out_valid_times.transpose('time_of_event','timestep_past')
    
    ds_out = change_time_of_prediction_to_time_of_event(ds_aligned)
    ds_out_valid_times = ds_out.where(ds_out.targets.count('timestep_past') == ds_out.targets.count('timestep_past').max(), drop=True)
    return ds_out_valid_times.transpose('time_of_event','timestep_past','var_name','latitude','longitude')

def get_ds_with_attributions(ds_aligned, lightning_model):
    if torch.cuda.is_available():
        device="cuda"
    else:
        device=='cpu'
    steps = ds_aligned.timestep_past.size
    
    lightning_model.to(device)
    lightning_model.eval()

    features_cpu = torch.tensor(ds_aligned.features.values, dtype=torch.float32, device='cpu')    

    method = IntegratedGradients(lightning_model.model)
    all_attrs = []

    for idx in tqdm(range(features_cpu.shape[0])):  # iterate over time_of_event
        x_in = features_cpu[idx].to(device, non_blocking=True).requires_grad_()  # shape: (timestep_past, var_name, lat, lon)
        baseline = torch.zeros_like(x_in, device=device)

        attrs = method.attribute(
            x_in,  # add batch dim if model expects it
            baselines=baseline,
            target=[steps - 1 -k for k in range(0,steps)]
        )

        attrs_cpu = attrs.detach().cpu()  # remove batch dim

        # Rebuild DataArray
        da_attrs = xr.DataArray(
            attrs_cpu,
            dims=['timestep_past', 'var_name', 'latitude', 'longitude'],
            coords=dict(
                timestep_past=ds_aligned.timestep_past,
                var_name=ds_aligned.var_name,
                latitude=ds_aligned.latitude,
                longitude=ds_aligned.longitude
            )
        )
        all_attrs.append(da_attrs.assign_coords(time_of_event=ds_aligned.time_of_event[idx]))

        del x_in, attrs, baseline
        torch.cuda.empty_cache()

    
    ds_attributions = xr.concat(all_attrs, dim='time_of_event')
    
    baseline = torch.zeros_like(features_cpu[idx][:1], device=device)  # shape: (timestep_past, var_name, lat, lon)
    with torch.no_grad():
        baseline_prediction = lightning_model.model(baseline)[0]
    baseline_prediction = baseline_prediction.cpu().numpy()
    baseline_prediction_da = xr.DataArray(baseline_prediction,
                                          dims=['timestep_past'], 
                                          coords = dict(timestep_past=np.arange(0,-baseline_prediction.size,-1))).sortby('timestep_past')
    ds_aligned['attributions'] = ds_attributions
    ds_aligned['baseline_attributions'] = baseline_prediction_da
    return ds_aligned

def haversine_np(lon1, lat1, lon2, lat2, radius=6371):
    # Convert degrees to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    # Compute differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c

def get_distance_to_cyclone_date(date, final_ds, df_features, cycs, grid):
    date_min = (pd.to_datetime(date) - pd.Timedelta(final_ds.timestep_past.size-1,'D')).strftime("%Y-%m-%d")
    features_date = df_features.loc[date].dropna().iloc[:-2]
    if features_date.size==0:
        # print("No feature")
        return
    cyclones_date = features_date.loc[features_date.str.startswith('C_')]
    if cyclones_date.size==0:
        # print("No cyclone")
        return
    cyclones_date = cyclones_date.str.split('_', expand=True).iloc[:,1].astype(int).values
    cycs_extract = cycs.loc[cyclones_date].set_index('date').sort_index()
    cycs_extract = cycs_extract.loc[date_min:date]
    cycs_extract = cycs_extract[['lon','lat']].to_xarray().rename(date='time')

    distance_to_cyclones = haversine_np(grid.longitude,grid.latitude, cycs_extract.lon, cycs_extract.lat)

    min_distances_to_cyclone = distance_to_cyclones.resample(time='D').min()
    min_distances_to_cyclone = min_distances_to_cyclone.assign_coords(time_of_event = pd.to_datetime(date))
    timesteps = (pd.to_datetime(min_distances_to_cyclone.time) - pd.to_datetime(date)).days
    min_distances_to_cyclone = min_distances_to_cyclone.rename(time='timestep_past').assign_coords(timestep_past=timesteps)
    return min_distances_to_cyclone

def get_distance_to_cyclones(ds_aligned):
    cycs = xr.open_dataset('/Data/gfi/spengler/kko033/cyclone_clustering/data/all_combined_NH_trcks.nc').to_dataframe().set_index('track_id').sort_index()
    df_features = pd.read_csv("/Data/gfi/users/rogui7909/data/attribute_precip_features/western_norway/features_attributed_to_rain_per_day_WN.csv",index_col=0, parse_dates=True)
    grid = xr.zeros_like(ds_aligned.attributions.isel(time_of_event=0, var_name=0, timestep_past=0, drop=True))
    all_dist = []
    days_str = ds_aligned.time_of_event.dt.strftime('%Y-%m-%d').values

    for date in tqdm(days_str):
        dist = get_distance_to_cyclone_date(date, ds_aligned, df_features, cycs, grid)
        if dist is not None:
            all_dist.append(dist)
    dist_to_cyclone = xr.concat(all_dist, dim='time_of_event')
    return dist_to_cyclone


def main(run_id=None, samples_to_attribute='all'):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    dataloader, lightning_model = load_trained_model(run_id)
    ds_out = get_ds_with_predictions(dataloader, lightning_model)
    ds_out = get_ds_with_attributions(ds_out, lightning_model)
    dist_to_cyclone = get_distance_to_cyclones(ds_out)
    ds_out['distance_to_cyclone'] = dist_to_cyclone
    print(f"Saving file in /Data/gfi/users/rogui7909/data/NN_outputs/attributions/{run_id}_attributions_{samples_to_attribute}.nc")
    ds_out.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/{run_id}_attributions_{samples_to_attribute}.nc")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model attribution")
    parser.add_argument("--run_id", type=str, required=True, help="WandB run ID")
    parser.add_argument("--samples", type=str, default='all', help="Sample subset to attribute. Can be TP, all-extr or all")

    args = parser.parse_args()
    
    main(args.run_id, args.samples)