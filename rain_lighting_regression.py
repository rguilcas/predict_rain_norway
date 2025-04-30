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
from models.lightning import LitCNN_regression, AttributableTrainer,LogPlotsRegression
from models.losses import DistribLoss, PinballLoss, WeightedOrdinalLoss, logits_to_class_probs, SortedMSELoss, CompositeQuantileLoss, HybridMSEQuantileLoss, PinballLossSquare, WeightedMSELoss, LogMSE,WeightedLogMSELoss, WeightedSqrtMSELoss, AsymmetricMSELoss, AsymmetricMSEabveThreshLoss
import random
import matplotlib
from models.data import get_input_data, get_input_data_small,get_input_data_small_extended
import pandas as pd
import cartopy.crs as ccrs
from models.attributions import log_attribution_regression

# matplotlib.use('tkagg')



seed_everything(42, workers=True)

learning_rate = 0.001
batch_size = 512
num_epochs = 20
device = 'cuda:1'
input_variable = ['u850','v850','z500']
groups = 3
loss_fn = 'asymetric_mse_thresh'
region_predicted = 14
type_prediction='regression'
lags = 2
input_type='small_extended'
attribute_test = True
augment_train_set = False
data_type='anomaly'
log_rain_prediction = False

match loss_fn:
    case 'distrib':
        loss = SortedMSELoss()
    case 'mse':
        loss = torch.nn.MSELoss()
    case 'quantiles90':
        loss = PinballLoss(0.9)
    case 'quantiles70':
        loss = PinballLoss(0.7)
    case 'quantiles80':
        loss = PinballLoss(0.8)
    case 'quantiles75':
        loss = PinballLoss(0.75)
    case 'composite_quantiles':
        loss = CompositeQuantileLoss(quantiles = [0.5,0.75,0.9], lambdas = None)
    case 'hybrid_mse_quantiles':
        loss = HybridMSEQuantileLoss()
    case 'weighted_mse':
        loss = WeightedMSELoss()
    case 'weighted_sqrt_mse':
        loss = WeightedSqrtMSELoss()
    case 'quantile_extr':
        loss = CompositeQuantileLoss(quantiles=[0.9,0.95,0.99])
    case 'log_mse':
        loss = LogMSE()
    case 'weighted_log_mse':
        loss = WeightedLogMSELoss()
    case 'asymetric_mse':
        loss = AsymmetricMSELoss(alpha=2)
    case 'asymetric_mse_thresh':
        loss = AsymmetricMSEabveThreshLoss(alpha=4, thresh=20)

num_classes =  1


config = {
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "num_classes": num_classes,
    "learning_rate":learning_rate,
    "input_variable" :input_variable,
    "number_variables":len(input_variable),
    "groups" : groups,
    "lags":lags,
    "region_predicted":region_predicted,
    "type_prediction":type_prediction,
    "loss_fn":loss_fn,
    "input_type":input_type,
    "augment_train_set":augment_train_set,
    "data_type":data_type,
    "log_rain_prediction":log_rain_prediction
    }



wandb_logger = WandbLogger(project="Predict-rain-WNorway", 
                           save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                           dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb",
                           config=config, name=f"CNN-{type_prediction} {'-'.join(input_variable)} lag {lags}")
# wandb_logger = WandbLogger(project="Predict-rain-WNorway", config=config, name="Truth")

print('Loading data ...')
if input_type=='small':
    image_size = 32*32
    train_loader, valid_loader, test_loader, ds_test = get_input_data_small(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags, augment_train_set=augment_train_set)
if input_type=='small_extended':
    image_size = 32*64
    train_loader, valid_loader, test_loader, ds_test = get_input_data_small_extended(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags, augment_train_set=augment_train_set)
elif input_type=='big':
    image_size=100*256
    train_loader, valid_loader, test_loader = get_input_data(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)
 
print('Data ready')

callbacks=[LogPlotsRegression()]

trainer = AttributableTrainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                              log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                              callbacks=callbacks, deterministic=True,
                              accelerator="gpu", devices=1)


num_output_neurons = num_classes

CNN = Wang2024(num_classes=num_output_neurons, num_channels_in=len(config['input_variable']), image_size=image_size, 
               groups=config['groups'])

model = LitCNN_regression(CNN, 
                         learning_rate=config['learning_rate'], 
                         loss_fn = loss)

print('Model init')
trainer.fit(model, train_loader, valid_loader)


model.eval()
with torch.no_grad():
    trainer.test(model, dataloaders=test_loader)
    # predictions_test = trainer.predict(model, test_loader)


if attribute_test:
    print('Computing attributions...')
    ds_attr = trainer.get_multi_attribution_ds_regr(test_loader, ds_test, min_rain=0)
    netcdf_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.netcdf"
    ds_attr.to_netcdf(netcdf_file)
    # log_attribution_regression(ds_attr, wandb_logger, input_variable)

    # baseline = torch.stack([torch.cat([batch[0] for batch in test_loader]).mean(dim=0)])
    # from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
    # ig = IntegratedGradients(model.model)
    # nt = NoiseTunnel(ig)
    # attributions = trainer.attribute(test_loader, attribution_method=nt, target=0, baselines=baseline)
    # ds_attr = xr.DataArray(attributions, dims = ds_test.dims, coords=ds_test.coords)
    # truth = torch.stack(model.test_step_true_values).cpu().numpy().squeeze()
    # pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
    # df = pd.DataFrame(dict(pred=pred, truth=truth), index=ds_test.time)
    # precip = df.to_xarray().rename(index='time')
    # attr_per_variable = ds_attr.sum(['longitude','latitude'])
    # time_true30 = df.query("truth>=30 & pred>=30").index

    # percent_importance = np.abs(ds_attr.sel(time=time_true30).stack(space=['latitude','longitude','var_name'])).rank('space', pct=True).unstack()

    # important_areas = percent_importance.where(percent_importance>0.99).count('time')/time_true30.size*100
