import os
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import wandb
import numpy as np
import torch # long
import xarray as xr

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from models.model import CNN_Model, CNN_Model_ChannelAttention, CNN_Model_SpatialAttention
import pandas as pd
from torch import nn
from lightning.pytorch import loggers, seed_everything
from models.lightning import LitCNN_regression, AttributableTrainer
# from models.losses import get_loss
import matplotlib
from models.data import get_input_data_from_wandb_logger,get_input_data_from_wandb_logger_deciles
# from models.model import get_model
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# matplotlib.use('tkagg')

import seaborn as sns 
import argparse
from tqdm.notebook import tqdm

def main(args=None):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    wandb_logger = loggers.WandbLogger(project="Predict-rain-WNorway_backtoclassif", 
                                    save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                                    dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb", )

    wandb_logger.experiment # Initialize wandb

    train_loader, val_loader, test_loader, ds_test = get_input_data_from_wandb_logger_deciles(wandb_logger)

    callbacks=[EarlyStopping(monitor="val/loss", mode="min")]
    trainer = AttributableTrainer(limit_train_batches=100, 
                                    max_epochs=wandb_logger.experiment.config['num_epochs'], 
                                    logger=wandb_logger, 
                                    log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                    callbacks=callbacks, deterministic=True,
                                    accelerator="gpu", devices=1,)

    wandb_logger.experiment.config['num_classes'] = wandb_logger.experiment.config['num_timesteps_predicted']
    wandb_logger.experiment.config['num_channels'] = len(wandb_logger.experiment.config['input_variables'])

    NN = CNN_Model(input_channels = wandb_logger.experiment.config['num_channels'], 
               image_size=wandb_logger.experiment.config['image_size'], 
               num_classes=9,
               number_conv_layers=wandb_logger.experiment.config['num_conv_layer'],
               size_conv_kernel=wandb_logger.experiment.config['size_conv_kernel'],
               out_channels_conv1=wandb_logger.experiment.config['conv1_kernel_number'], 
               out_channel_factor_increase_per_layer = wandb_logger.experiment.config['factor_increase_kernels_per_conv_layer'],
               sigmoid=False, softmax=False)

    def loss(x,y):
        return corn_loss(x,y,10)

    model = LitCNN_regression(NN, 
                            learning_rate=wandb_logger.experiment.config['learning_rate'], 
                            lr_scheduler =wandb_logger.experiment.config['lr_scheduler'],
                            loss_fn = loss)

    trainer.fit(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        trainer.test(model, dataloaders=test_loader)

    y_test = torch.stack(model.test_true_values)#.cpu().numpy()
    y_scores = torch.stack(model.test_pred)#.cpu().numpy()

    # model.test_true_values

    pred = corn_label_from_logits(y_scores)

    df_ = pd.DataFrame(dict(targets=y_test.cpu().numpy(), pred=pred.cpu().numpy()))

    confusion = df_.reset_index().groupby(['targets','pred']).index.count().unstack().fillna(0)
    TP = confusion.loc[9,9]
    prec = TP/confusion.loc[:,9].sum()
    recall = TP/confusion.loc[9].sum()
    f1 = prec*recall*2/(prec+recall)

    wandb.log({"diagnostics/f1_90th": float(f1),
            "diagnostics/recall_90th": float(recall),
            "diagnostics/prec_90th": float(prec),
            })

    TP = confusion.loc[8:,8:].sum().sum()
    prec = TP/confusion.loc[:,8:].sum().sum()
    recall = TP/confusion.loc[8:].sum().sum()
    f1 = prec*recall*2/(prec+recall)

    wandb.log({"diagnostics/f1_80th": float(f1),
            "diagnostics/recall_80th": float(recall),
            "diagnostics/prec_80th": float(prec),
            })
    
    wandb.finish()

if __name__ == '__main__':
    main()