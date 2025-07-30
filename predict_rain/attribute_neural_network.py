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
import numpy as np
import argparse

from ml_module_rain.models.trained_models import load_trained_model

def main(run_id=None, samples_to_attribute='all-extr'):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    dataloader, lightning_model = load_trained_model(run_id)

    all_targets = []
    all_probas = []
    for batch in tqdm(dataloader.val_loader):
        x, y = batch
        lightning_model.eval()
        with torch.no_grad():
            all_probas.append(lightning_model(x))
        all_targets.append(y)
    probas = torch.cat(all_probas).view(-1, 4, 3)
    targets = torch.cat(all_targets).cpu().numpy()
    preds = torch.argmax(probas, dim=2).cpu().numpy() 
    probas = probas.cpu().numpy()
    
    ds_aligned = dataloader.ds_val.isel(time=range(targets.shape[0]))
    preds_da = xr.DataArray(preds, dims=['time','timestep'], coords =ds_aligned.targets.coords)
    ds_aligned ['predictions'] = preds_da
    
    
    series_targets = ds_aligned.targets.to_series()
    series_predictions = ds_aligned.predictions.to_series()
    
    all_predictions = pd.DataFrame(dict(predictions=series_predictions, targets=series_targets)).reset_index()
    all_predictions.columns = ['time_of_prediction','timestep','prediction','target']
    all_predictions['time_of_event'] = all_predictions.time_of_prediction + all_predictions.timestep*pd.Timedelta(1,'D')
    valid_times = all_predictions.groupby("time_of_event").target.count().reset_index().query("target==4").time_of_event # Only four predictions
    all_predictions = all_predictions.loc[all_predictions.time_of_event.isin(valid_times)]
    
    match samples_to_attribute:
        case 'TP':
            time_of_attributions = all_predictions.query("(prediction==target) & (target==2)").groupby('time_of_event').time_of_prediction.count().reset_index().query("time_of_prediction==4").time_of_event.values
        case 'all-extr':
            time_of_attributions =all_predictions.query("prediction==2 | target==2").time_of_event.unique()
        case 'all':
            time_of_attributions = all_predictions.time_of_event.unique()
        case _:
            print("Invalid selection of attributions")

    predictions_attributions = all_predictions.loc[all_predictions.time_of_event.isin(time_of_attributions)].sort_values(['time_of_event','timestep'], ascending=[True, False])
    start_times = predictions_attributions.query("timestep==3").time_of_prediction.dt.strftime("%Y-%m-%d %H:%M:%S").values
    end_times = predictions_attributions.query("timestep==0").time_of_prediction.dt.strftime("%Y-%m-%d %H:%M:%S").values

    method = IntegratedGradients(lightning_model.model)
    all_multi_attrs = []
    all_multi_sens = []
    steps=4

    for k in tqdm(range(len(start_times))):
        start = start_times[k]
        end = end_times[k]
        # print(start,end)
        ds_test_extract =  ds_aligned.sel(time=slice(start ,end))
        tensor_in = torch.Tensor(ds_test_extract.features.values)
        tensor_out = torch.Tensor(ds_test_extract.targets.values)
        attrs = method.attribute(tensor_in, baselines=torch.Tensor([0]),target=[3*(steps-k)-1 for k in range(steps)]) 
        da_attrs = xr.DataArray(attrs, dims = ['timestep','var_name','latitude','longitude'], 
                                coords= dict(timestep=np.arange(1,steps+1),var_name=ds_test_extract.var_name, latitude=ds_test_extract.latitude, longitude=ds_test_extract.longitude))
        all_multi_attrs.append(da_attrs.assign_coords(time=time_of_attributions[k]))
        sens = da_attrs/ds_test_extract.features.rename(time='timestep').assign_coords(timestep=da_attrs.timestep)
        all_multi_sens.append(sens.assign_coords(time=time_of_attributions[k]))
        # break
    ds_attributions = xr.concat(all_multi_attrs, dim='time').assign_coords(timestep=np.arange(-3,1))
    ds_sens = xr.concat(all_multi_sens, dim='time').assign_coords(timestep=np.arange(-3,1))
    ds_predictions_attributions = predictions_attributions.set_index(['time_of_event','timestep'])[['prediction','target']].to_xarray().rename(time_of_event='time')
    ds_predictions_attributions = ds_predictions_attributions.assign_coords(timestep=-ds_predictions_attributions.timestep).sortby('timestep')
    final_ds = xr.merge([xr.Dataset(dict(attributions = ds_attributions, sensitivity=ds_sens)), ds_predictions_attributions])
    final_ds.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/{run_id}_attributions_{samples_to_attribute}.nc")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model attribution")
    parser.add_argument("--run_id", type=str, required=True, help="WandB run ID")
    parser.add_argument("--samples", type=str, default="TP", help="Sample subset to attribute. Can be TP, all-extr or all")

    args = parser.parse_args()
    
    main(args.run_id, args.samples)