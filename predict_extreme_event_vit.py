# print('Running imports...')
import os


import wandb
import torch # long
import xarray as xr


from models.model import CNN_Model, CNN_Model_SpatialAttention, CNN_Model_CBAM, CNN_Model_ChannelAttention
from vit_pytorch import ViT

import pandas as pd
from torch import nn
from lightning.pytorch import loggers, seed_everything
from models.lightning import LitCNN_regression, AttributableTrainer
# from models.losses import get_loss
from models.data import get_input_data_from_wandb_logger
# from models.model import get_model

from lightning.pytorch.callbacks.early_stopping import EarlyStopping




def main(args=None):
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    wandb_logger = loggers.WandbLogger(project="Predict-rain-WNorway_test", 
                                        save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                                        dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb", )

    wandb_logger.experiment # Initialize wandb

    train_loader, val_loader, test_loader, ds_test = get_input_data_from_wandb_logger(wandb_logger)

    callbacks=[EarlyStopping(monitor="val/loss", mode="min")]
    trainer = AttributableTrainer(limit_train_batches=100, 
                                    max_epochs=wandb_logger.experiment.config['num_epochs'], 
                                    logger=wandb_logger, 
                                    log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                    callbacks=callbacks, deterministic=True,
                                    accelerator="gpu", devices=1,)

    wandb_logger.experiment.config['num_classes'] = wandb_logger.experiment.config['num_timesteps_predicted']
    wandb_logger.experiment.config['num_channels'] = len(wandb_logger.experiment.config['input_variables'])
        
        
    NN = ViT(
            image_size=wandb_logger.experiment.config['image_size'],            # your input H=W (e.g., 64×64)
            patch_size=wandb_logger.experiment.config['patch_size'],            # must divide image_size exactly
            num_classes= wandb_logger.experiment.config['num_classes'],         # binary classification
            channels=len(wandb_logger.experiment.config['input_variables']),    # number of input channels
            dim=wandb_logger.experiment.config['embedding_dimension'],          # embedding dimension
            depth=wandb_logger.experiment.config['depth_transformer'],          # number of transformer blocks
            heads=wandb_logger.experiment.config['attention_heads'],            # number of attention heads
            mlp_dim=wandb_logger.experiment.config['mlp_dim'],                  # feedforward layer size
            dropout=wandb_logger.experiment.config['dropout'],
            emb_dropout=wandb_logger.experiment.config['emb_dropout']
        )
    
    loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(10*wandb_logger.experiment.config['quantile_thresh']))
    model = LitCNN_regression(NN, 
                              learning_rate=wandb_logger.experiment.config['learning_rate'], 
                              lr_scheduler = wandb_logger.experiment.config['lr_scheduler'],
                              loss_fn = loss)
    
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        trainer.test(model, dataloaders=test_loader)

    # Diag
    # targets = xr.DataArray(torch.stack(model.test_true_values).cpu().numpy(), dims=['time','timestep'], coords = ds_test.targets.coords)
    # predictions = xr.DataArray(torch.sigmoid(torch.stack(model.test_pred)).cpu().numpy(), dims=['time','timestep'], coords = ds_test.targets.coords)
    # pred = xr.ones_like(predictions).where(predictions>.7,0)

    y_test = torch.cat(model.test_true_values)#.cpu().numpy()
    y_scores = torch.sigmoid(torch.cat(model.test_pred))#.cpu().numpy()
    from torcheval.metrics.functional import binary_auprc
    wandb.log({"diagnostics/auc_pr": float(binary_auprc(y_scores, y_test).cpu())})



    # TP = ((pred == targets) & (pred==1)).sum(['time','timestep'])
    # prec = TP/(pred == 1).sum(['time','timestep'])
    # recall = TP/(targets == 1).sum(['time','timestep'])
    # F1 =2*(recall*prec)/(recall+prec)
    # wandb.log({'diagnostics/F1_all':F1.values})

    # TP = ((pred == targets) & (pred==1)).sum(['time'])
    # prec = TP/(pred == 1).sum(['time'])
    # recall = TP/(targets == 1).sum(['time'])
    # F1 =2*(recall*prec)/(recall+prec)
    # F1
    # for timestep in F1.timestep.values:
    #     wandb.log({f'testF1/F1_+{timestep*6}H':F1.isel(timestep=timestep).values})

    # ds_ = xr.Dataset(dict(pred=pred, targets=targets))
    # df_ = ds_.to_dataframe().reset_index()
    # df_ = df_.query("pred==targets & targets==1").reset_index()
    # df_['rain_time'] = df_.time + pd.Series([pd.Timedelta(f'{k*6}h') for k in df_.timestep])
    # df_out = xr.open_dataset(wandb_logger.experiment.config['file_name_data_out']).tp.to_series()
    # steps = wandb_logger.experiment.config['num_timesteps_predicted']

    # df_count = df_.groupby(df_.rain_time).time.count()
    # events_predicted = df_count.loc[df_count==steps].index
    # start_times = events_predicted - pd.to_timedelta(6*steps-1,'h')

    # wandb.log({'diagnostics/percent_event_full_detected':(events_predicted.size/df_count.size)*100})
    wandb.finish()

if __name__ == "__main__":
    main()