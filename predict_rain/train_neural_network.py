print('Running imports...')
import os
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import wandb
import torch # long
from lightning.pytorch import loggers, seed_everything
import xarray as xr
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse


from ml_module_rain.models.neuralnetworks import get_neural_network, register_shape_hooks
from ml_module_rain.data.datamodule import MyDataLoader
from ml_module_rain.models.lightning import ExtremeRainPredictor, AttributableTrainer
from ml_module_rain.models.losses import get_loss
from ml_module_rain.models.callbacks import LogF1Validation, get_checkpoint_callback, BestF1Callback
from ml_module_rain.utils.config import load_config
from pathlib import Path

def main(config_path=None):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    init_config = load_config(config_path)
    wandb_logger = loggers.WandbLogger(project="Predict-rain-WNorway_v6", 
                                    save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                                    dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb",
                                    config=init_config )

    wandb_logger.experiment # Initialize wandb
    config = wandb_logger.experiment.config
    dataloader = MyDataLoader(config)
    dataloader.print_infos()
    NN = get_neural_network(config)

    callbacks=[EarlyStopping(monitor="val/loss", mode="min"), 
               get_checkpoint_callback(wandb_logger),
               BestF1Callback(multi_horizon=True)
            #    LogF1Validation(num_timesteps=config['num_timesteps_predicted']),
               ]
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    trainer = AttributableTrainer(limit_train_batches=100, 
                                max_epochs=config['num_epochs'], 
                                logger=wandb_logger, 
                                log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                callbacks=callbacks, deterministic=True,
                                accelerator=accelerator, devices=1)

    loss = get_loss(config)
    lightning_model = ExtremeRainPredictor(NN, 
                            learning_rate=config['learning_rate'], 
                            lr_scheduler =config['lr_scheduler'],
                            loss_fn = loss,
                            init_config=init_config)
    trainer.fit(lightning_model,dataloader.train_loader, dataloader.val_loader)
    lightning_model.eval()
    lightning_model.model.eval()
    with torch.no_grad():
        trainer.test(lightning_model, dataloaders=dataloader.val_loader)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model attribution")
    parser.add_argument("--config_path", type=str, default="/home/rogui7909/code/predict_rain_norway/predict_rain/ml_module_rain/configs/config-defaults.yaml", help="path to config file")

    args = parser.parse_args()
   
    main(config_path = args.config_path)