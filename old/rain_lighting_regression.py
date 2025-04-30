print('Running imports...')

import torch # long
import os
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import xarray as xr
from models.model import Wang2024
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import wandb
from models.lightning import LitCNN_regression, AttributableTrainer, LogPlotsRegression
from models.losses import get_loss
import random
import matplotlib
from models.data import get_input_data, get_input_data_small,get_input_data_small_extended
import pandas as pd
import cartopy.crs as ccrs
from models.attributions import log_attribution_regression
import argparse
# matplotlib.use('tkagg')



seed_everything(42, workers=True)

def main(args=None):
    attribute_test = True
    wandb_logger = WandbLogger(project="Predict-rain-WNorway_v2", 
                               save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                               dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb", )

    wandb_logger.experiment # Initialize wandb
    if args is not None:
        new_args = {k:vars(args)[k] for k in vars(args) if vars(args)[k] is not None}
        wandb_logger.experiment.config.update(new_args, allow_val_change=True)

    train_loader, valid_loader, test_loader, ds_test = get_input_data_small_extended(input_variable=wandb_logger.experiment.config['input_variable'], 
                                                                                    region_predicted=14,
                                                                                    type_prediction='regression', 
                                                                                    batch_size=wandb_logger.experiment.config['batch_size'], 
                                                                                    lags=wandb_logger.experiment.config['lags'], 
                                                                                    augment_train_set=False)

    image_size = train_loader.dataset.dataset.dataset.tensors[0].shape[-1]*train_loader.dataset.dataset.dataset.tensors[0].shape[-2]

    print('Data ready')

    callbacks=[LogPlotsRegression()]

    trainer = AttributableTrainer(limit_train_batches=100, max_epochs=wandb_logger.experiment.config['num_epochs'], 
                                logger=wandb_logger, 
                                log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                callbacks=callbacks, deterministic=True,
                                accelerator="gpu", devices=1,
                                gradient_clip_val=wandb_logger.experiment.config['gradient_clip_val'])

    num_classes = 1
    num_output_neurons = num_classes

    CNN = Wang2024(num_classes=num_output_neurons, 
                num_channels_in=len(wandb_logger.experiment.config['input_variable']), 
                image_size=image_size, 
                groups=wandb_logger.experiment.config['groups'],
                size_conv1_kernel=wandb_logger.experiment.config['conv1_kernel_size'],
                size_conv2_kernel=wandb_logger.experiment.config['conv2_kernel_size'],
                out_channels_conv1=wandb_logger.experiment.config['conv1_kernel_number'],
                out_channels_conv2=wandb_logger.experiment.config['conv2_kernel_number'],
                dropout_proba=wandb_logger.experiment.config['dropout_proba'],)


    loss = get_loss(wandb_logger.experiment.config['loss_fn'])


    model = LitCNN_regression(CNN, 
                            learning_rate=wandb_logger.experiment.config['learning_rate'], 
                            loss_fn = loss)

    print('Model init')
    trainer.fit(model, train_loader, valid_loader)


    model.eval()
    with torch.no_grad():
        trainer.test(model, dataloaders=test_loader)

    if attribute_test:
        print('Computing attributions...')
        ds_attr = trainer.get_multi_attribution_ds_regr(test_loader, ds_test, min_rain=0)
        netcdf_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.netcdf"
        ds_attr.to_netcdf(netcdf_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--input_variable")
    parser.add_argument("--groups", type=int)
    parser.add_argument("--loss_fn")
    parser.add_argument("--lags", type=int)
    parser.add_argument("--conv1_kernel_size",type=int)
    parser.add_argument("--conv2_kernel_size",type=int)
    parser.add_argument("--conv1_kernel_number",type=int)
    parser.add_argument("--conv2_kernel_number",type=int)
    args = parser.parse_args()
    main(args)

    # main(args)
    # predictions_test = trainer.predict(model, test_loader)


# if attribute_test:
#     print('Computing attributions...')
#     ds_attr = trainer.get_multi_attribution_ds_regr(test_loader, ds_test, min_rain=0)
#     netcdf_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.netcdf"
#     ds_attr.to_netcdf(netcdf_file)
#     # log_attribution_regression(ds_attr, wandb_logger, input_variable)

#     # baseline = torch.stack([torch.cat([batch[0] for batch in test_loader]).mean(dim=0)])
#     # from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
#     # ig = IntegratedGradients(model.model)
#     # nt = NoiseTunnel(ig)
#     # attributions = trainer.attribute(test_loader, attribution_method=nt, target=0, baselines=baseline)
#     # ds_attr = xr.DataArray(attributions, dims = ds_test.dims, coords=ds_test.coords)
#     # truth = torch.stack(model.test_step_true_values).cpu().numpy().squeeze()
#     # pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
#     # df = pd.DataFrame(dict(pred=pred, truth=truth), index=ds_test.time)
#     # precip = df.to_xarray().rename(index='time')
#     # attr_per_variable = ds_attr.sum(['longitude','latitude'])
#     # time_true30 = df.query("truth>=30 & pred>=30").index

#     # percent_importance = np.abs(ds_attr.sel(time=time_true30).stack(space=['latitude','longitude','var_name'])).rank('space', pct=True).unstack()

#     # important_areas = percent_importance.where(percent_importance>0.99).count('time')/time_true30.size*100
