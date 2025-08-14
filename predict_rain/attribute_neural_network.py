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
    preds_da = xr.DataArray(preds, dims=["time", "timestep"], coords=ds_aligned.targets.coords)
    ds_aligned["predictions"] = preds_da

    return ds_aligned

def get_ds_with_attributions(dataloader, time_of_attributions, ds_prediction, lightning_model, ds_aligned):
    if torch.cuda.is_available():
        device="cuda"
    else:
        device=='cpu'
    steps = dataloader.config['num_timesteps_predicted']
    start_times = (time_of_attributions - pd.Timedelta(ds_prediction.timestep.size - 1, 'D')).dt.strftime("%Y-%m-%d %H:%M:%S").values
    end_times = time_of_attributions.dt.strftime("%Y-%m-%d %H:%M:%S").values

    lightning_model.to(device)
    lightning_model.eval()

    method = IntegratedGradients(lightning_model.model)

    all_multi_attrs = []
    all_multi_sens = []

    for k in tqdm(range(len(start_times))):
        # Select the time slice
        ds_test_extract = ds_aligned.sel(time=slice(start_times[k], end_times[k]))
        # Prepare input tensor on GPU
        tensor_in = torch.tensor(ds_test_extract.features.values, device=device, dtype=torch.float32, requires_grad=True)
        # Baseline on GPU
        baseline = torch.zeros_like(tensor_in, device=device)
        # Compute attributions
        attrs = method.attribute(
            tensor_in,
            baselines=baseline,
            target=[ds_prediction.timestep.size - 1 - t for t in range(ds_prediction.timestep.size)]
        )

        # Move results back to CPU once
        attrs_cpu = attrs.detach().cpu()

        da_attrs = xr.DataArray(
            attrs_cpu,
            dims=['timestep', 'var_name', 'latitude', 'longitude'],
            coords=dict(
                timestep=np.arange(1, steps + 1),
                var_name=ds_test_extract.var_name,
                latitude=ds_test_extract.latitude,
                longitude=ds_test_extract.longitude
            )
        )

        all_multi_attrs.append(da_attrs.assign_coords(time=time_of_attributions[k]))

        sens = da_attrs / ds_test_extract.features.rename(time='timestep').assign_coords(timestep=da_attrs.timestep)
        all_multi_sens.append(sens.assign_coords(time=time_of_attributions[k]))

    timesteps = np.arange(-steps + 1, 1)
    ds_attributions = xr.concat(all_multi_attrs, dim='time_of_event').assign_coords(timestep=timesteps)
    ds_sens = xr.concat(all_multi_sens, dim='time_of_event').assign_coords(timestep=timesteps)

    final_ds = xr.merge([xr.Dataset(dict(attributions=ds_attributions, sensitivity=ds_sens)), ds_prediction])
    return final_ds

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

def get_distance_to_cyclone(date, final_ds, df_features, cycs, grid):
    date_min = (pd.to_datetime(date) - pd.Timedelta(final_ds.timestep.size-1,'D')).strftime("%Y-%m-%d")
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
    min_distances_to_cyclone = min_distances_to_cyclone.rename(time='timestep').assign_coords(timestep=timesteps)
    return min_distances_to_cyclone

def get_distance_to_cyclones(final_ds):
    cycs = xr.open_dataset('/Data/gfi/spengler/kko033/cyclone_clustering/data/all_combined_NH_trcks.nc').to_dataframe().set_index('track_id').sort_index()
    df_features = pd.read_csv("/Data/gfi/users/rogui7909/data/attribute_precip_features/western_norway/features_attributed_to_rain_per_day_WN.csv",index_col=0, parse_dates=True)
    grid = xr.zeros_like(final_ds.attributions.isel(time_of_event=0, var_name=0, timestep=0, drop=True))
    all_dist = []
    days_str = final_ds.time_of_event.dt.strftime('%Y-%m-%d').values

    for date in tqdm(days_str):
        dist = get_distance_to_cyclone(date, final_ds, df_features, cycs, grid)
        if dist is not None:
            all_dist.append(dist)
    dist_to_cyclone = xr.concat(all_dist, dim='time_of_event')
    return dist_to_cyclone


def main(run_id=None, samples_to_attribute='extr'):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    dataloader, lightning_model = load_trained_model(run_id)
    ds_aligned = get_ds_with_predictions(dataloader, lightning_model)


    series_targets = ds_aligned.targets.to_series()
    series_predictions = ds_aligned.predictions.to_series()
    all_predictions = pd.DataFrame(dict(predictions=series_predictions, targets=series_targets)).reset_index()
    all_predictions.columns = ['time_of_prediction','timestep','prediction','target']
    all_predictions['time_of_event'] = all_predictions.time_of_prediction + all_predictions.timestep*pd.Timedelta(1,'D')
    valid_times = all_predictions.groupby("time_of_event").target.count().reset_index().query(f"target=={dataloader.config['num_timesteps_predicted']}").time_of_event # Only four predictions
    all_predictions = all_predictions.loc[all_predictions.time_of_event.isin(valid_times)]

    ds_prediction = all_predictions.pivot(index='time_of_event', values=['target','prediction'], columns='timestep').stack().to_xarray()
    ds_prediction = ds_prediction.assign_coords(timestep=np.arange(0,-ds_prediction.timestep.size,-1))

    time_of_attributions = ds_prediction.time_of_event

    match samples_to_attribute:
        case 'extr':
                time_of_attributions = ds_prediction.time_of_event.where(ds_prediction.isel(timestep=0, drop=True).target == 1, drop=True)
        case 'all':
            pass
        case _:
            print("Invalid selection of attributions")
    
    time_of_attributions = ds_prediction.time_of_event.where(ds_prediction.isel(timestep=0, drop=True).target == 1, drop=True)

    final_ds = get_ds_with_attributions(dataloader, time_of_attributions, ds_prediction, lightning_model, ds_aligned)

    final_ds['distance_to_cyclone'] = get_distance_to_cyclones(final_ds)

    final_ds.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/{run_id}_attributions_{samples_to_attribute}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model attribution")
    parser.add_argument("--run_id", type=str, required=True, help="WandB run ID")
    parser.add_argument("--samples", type=str, default='extr', help="Sample subset to attribute. Can be TP, all-extr or all")

    args = parser.parse_args()
    
    main(args.run_id, args.samples)