print('Running imports...')
import os
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import wandb
import torch # long
from lightning.pytorch import loggers, seed_everything
import xarray as xr
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.neuralnetworks import get_neural_network
from data.datamodule import MyDataLoader
from models.lightning import ExtremeRainPredictor, AttributableTrainer
from models.losses import get_loss
from models.callbacks import LogF1Validation, get_checkpoint_callback
from models.trained_models import load_trained_model
from tqdm import tqdm
import pandas as pd
import glob


def main(run_id=None, samples_to_attribute='all-extr'):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    dataloader, lightroom_model = load_trained_model(run_id)
    all_targets = []
    all_probas = []
    for batch in tqdm(dataloader.val_loader):
        x, y = batch
        lightroom_model.eval()
        with torch.no_grad():
            all_probas.append(lightroom_model(x))
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
    
    all_extreme_pred = ((series_predictions==2)).reset_index()
    all_extreme_pred.columns = ['time_of_prediction','timestep','predicted_extreme']
    all_extreme_pred['time_of_event'] = all_extreme_pred.time_of_prediction + all_extreme_pred.timestep*pd.Timedelta(1,'D')
    all_extreme_pred = all_extreme_pred.groupby('time_of_event').predicted_extreme.sum()

    all_preds_correct = ((series_predictions == series_targets)&(series_targets==2)).reset_index()
    all_preds_correct.columns = ['time_of_prediction','timestep','TP_extreme']
    all_preds_correct['time_of_event'] = all_preds_correct.time_of_prediction + all_preds_correct.timestep*pd.Timedelta(1,'D')
    extreme_events_correct_predictions = all_preds_correct.groupby('time_of_event').TP_extreme.sum()

    all_targets = ((series_targets==2)).reset_index()
    all_targets.columns = ['time_of_prediction','timestep','extreme']
    all_targets['time_of_event'] = all_targets.time_of_prediction + all_targets.timestep*pd.Timedelta(1,'D')
    extreme_events = all_targets.groupby('time_of_event').extreme.sum()

    extreme_events_correct_predictions = extreme_events_correct_predictions.loc[extreme_events[extreme_events==4].index]
    
    match samples_to_attribute:
        case 'TP':
            time_of_attributions = extreme_events_correct_predictions.loc[extreme_events_correct_predictions==4].index
        case 'all_extr':
            time_of_attributions = extreme_events.loc[extreme_events==4].index
        case 'all':
            time_of_attributions = extreme_events.index
        case _:
            print("Invalid selection of attributions")
        
    # dataloader.attribute_integrated_gradients(lightroom_model, dataloader.ds_val, lightroom_model.predictions_test, lightroom_model.targets_test)
    # loss = MultiCrossEntropyLoss()


    
    # with torch.no_grad():
    #     trainer.test(lightroom_model, dataloaders=dataloader.val_loader)
    # if config['attribute_true_positives'] in ['TP','all_extr','all']:
    #     dataloader.attribute_integrated_gradients(lightroom_model, dataloader.ds_val, lightroom_model.predictions_test, lightroom_model.targets_test)
    #     dataloader.ds_attribution.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/attributions_TP_{wandb.run.id}.nc")
    #     # plot1 = plot_mean_attributions(loader.ds_attribution)
    #     # wandb.log({"attributions/mean_attribution_plot": wandb.Image(plot1.fig)})
    #     # plot2 = plot_top1pct_pixels(loader.ds_attribution)
    #     # wandb.log({"attributions/top1pct_attributions": wandb.Image(plot2.fig)})
    # wandb.finish()


if __name__ == "__main__":
    run_id = 'hylfeyaz'
    samples_to_attribute = 'all_extr'
    main(run_id, samples_to_attribute)