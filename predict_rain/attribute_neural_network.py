print('Running imports...')
import os
from captum.attr import IntegratedGradients

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch # long
from lightning.pytorch import seed_everything
from torch import nn 

import xarray as xr
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn.objects as so
import dask.bag as db
import argparse
import matplotlib.pyplot as plt
from ml_module_rain.models.trained_models import load_trained_model


def get_ds_with_predicitons(dataloader,lightning_model ):
    all_targets = []
    all_preds = []
    for batch in tqdm(dataloader.val_loader):
        x, y = batch
        lightning_model.eval()
        with torch.no_grad():
            all_preds.append(lightning_model(x))
        all_targets.append(y)
    targets = torch.cat(all_targets).cpu().numpy()
    preds = torch.nn.Sigmoid()(torch.cat(all_preds)).cpu().numpy()

    ds_aligned = dataloader.ds_val.isel(time=range(targets.shape[0]))
    preds_da = xr.DataArray(preds, dims=['time','timestep'], coords =ds_aligned.targets.coords)
    ds_aligned['predictions'] = preds_da

    return ds_aligned

def get_ds_with_attributions(dataloader, time_of_attributions, ds_prediction, lightning_model, ds_aligned):
    steps = dataloader.config['num_timesteps_predicted']
    start_times = (time_of_attributions-pd.Timedelta(ds_prediction.timestep.size-1,'D')).dt.strftime("%Y-%m-%d %H:%M:%S").values
    end_times = (time_of_attributions).dt.strftime("%Y-%m-%d %H:%M:%S").values
    method = IntegratedGradients(lightning_model.model)
    all_multi_attrs = []
    all_multi_sens = []
    for k in tqdm(range(len(start_times))):
        start = start_times[k]
        end = end_times[k]
        ds_test_extract =  ds_aligned.sel(time=slice(start ,end))
        tensor_in = torch.Tensor(ds_test_extract.features.values).requires_grad_()
        attrs = method.attribute(tensor_in, baselines=torch.Tensor([0]),target=[ds_prediction.timestep.size-1 -k for k in range(ds_prediction.timestep.size)]) 
        da_attrs = xr.DataArray(attrs.detach(), dims = ['timestep','var_name','latitude','longitude'], 
                                    coords= dict(timestep=np.arange(1,steps+1),var_name=ds_test_extract.var_name, latitude=ds_test_extract.latitude, longitude=ds_test_extract.longitude))
        all_multi_attrs.append(da_attrs.assign_coords(time=time_of_attributions[k]))
        sens = da_attrs/ds_test_extract.features.rename(time='timestep').assign_coords(timestep=da_attrs.timestep)
        all_multi_sens.append(sens.assign_coords(time=time_of_attributions[k]))
    timesteps = np.arange(-steps+1,1)
    ds_attributions = xr.concat(all_multi_attrs, dim='time_of_event').assign_coords(timestep=timesteps)
    ds_sens = xr.concat(all_multi_sens, dim='time_of_event').assign_coords(timestep=timesteps)
    final_ds = xr.merge([xr.Dataset(dict(attributions = ds_attributions, sensitivity=ds_sens)), ds_prediction])
    return final_ds


def main(run_id=None, samples_to_attribute='extr'):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    dataloader, lightning_model = load_trained_model(run_id)
    ds_aligned = get_ds_with_predicitons(dataloader, lightning_model)


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
    final_ds.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/{run_id}_attributions_{samples_to_attribute}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model attribution")
    parser.add_argument("--run_id", type=str, required=True, help="WandB run ID")
    parser.add_argument("--samples", type=str, default="TP", help="Sample subset to attribute. Can be TP, all-extr or all")

    args = parser.parse_args()
    
    main(args.run_id, args.samples)