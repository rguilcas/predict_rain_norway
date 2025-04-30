print('Running imports...')

import sys
import torch # long
import os
import numpy as np
import lightning as L
import xarray as xr
from models.model import Wang2024
from lightning.pytorch import loggers, seed_everything

import wandb
from models.lightning import LitCNN_regression, AttributableTrainer
from models.losses import get_loss
import random
import matplotlib
from models.data import get_input_data_from_wandb_logger
from models.model import get_model
from models.callbacks import LogIndividualScores
import argparse
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# matplotlib.use('tkagg')



seed_everything(42, workers=True)


def main(args=None):
    wandb_logger = loggers.WandbLogger(project="Predict-rain-WNorway_multidays", 
                                       save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                                       dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb", )

    wandb_logger.experiment # Initialize wandb

    if wandb_logger.experiment.config['num_days_predictant']>1:
        wandb_logger.experiment.config['LSTM_sequence'] = True

    train_loader, valid_loader, test_loader, ds_test = get_input_data_from_wandb_logger(wandb_logger)
               
    callbacks=[EarlyStopping(monitor="val/loss", mode="min"), LogIndividualScores()]

    trainer = AttributableTrainer(limit_train_batches=100, 
                                  max_epochs=wandb_logger.experiment.config['num_epochs'], 
                                  logger=wandb_logger, 
                                  log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                  callbacks=callbacks, deterministic=True,
                                  accelerator="gpu", devices=1,
                                  gradient_clip_val=wandb_logger.experiment.config['gradient_clip_val'])

    wandb_logger.experiment.config['num_classes'] = wandb_logger.experiment.config['num_days_predicted']
    wandb_logger.experiment.config['num_channels'] = len(wandb_logger.experiment.config['input_variable'])
    
    NN = get_model(wandb_logger)

    loss = get_loss(wandb_logger.experiment.config['loss_fn'])


    model = LitCNN_regression(NN, 
                              learning_rate=wandb_logger.experiment.config['learning_rate'], 
                              lr_scheduler =wandb_logger.experiment.config['lr_scheduler'],
                              loss_fn = loss)

    print('Model init')
    trainer.fit(model, train_loader, valid_loader)


    model.eval()
    with torch.no_grad():
        trainer.test(model, dataloaders=test_loader)


    if wandb_logger.experiment.config['attribute_test']:
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
