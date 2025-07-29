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
from models.neuralnetworks import CNN_MLP, register_shape_hooks
from models.losses import get_loss
from models.callbacks import LogF1Validation, get_checkpoint_callback
from utils.plotting import plot_mean_attributions, plot_top1pct_pixels
from utils.config import load_config, save_config

def main(args=None):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    config_path = 'configs/config-defaults.yaml'
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
               LogF1Validation(),
               ]
    trainer = AttributableTrainer(limit_train_batches=100, 
                                max_epochs=config['num_epochs'], 
                                logger=wandb_logger, 
                                log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                callbacks=callbacks, deterministic=True,
                                accelerator="cpu", devices=1)

    loss = get_loss(config['loss_function'], timesteps = config['num_timesteps_predicted'])
    # loss = MultiCrossEntropyLoss()
    lightroom_model = ExtremeRainPredictor(NN, 
                            learning_rate=config['learning_rate'], 
                            lr_scheduler =config['lr_scheduler'],
                            loss_fn = loss,
                            config = config)
    trainer.fit(lightroom_model,dataloader.train_loader, dataloader.val_loader)
    lightroom_model.eval()
    with torch.no_grad():
        trainer.test(lightroom_model, dataloaders=dataloader.val_loader)

    # if config['attribute_true_positives'] in ['TP','all_extr','all']:
    #     dataloader.attribute_integrated_gradients(lightroom_model, dataloader.ds_val, lightroom_model.predictions_test, lightroom_model.targets_test)
    #     dataloader.ds_attribution.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/attributions_TP_{wandb.run.id}.nc")
    #     # plot1 = plot_mean_attributions(loader.ds_attribution)
    #     # wandb.log({"attributions/mean_attribution_plot": wandb.Image(plot1.fig)})
    #     # plot2 = plot_top1pct_pixels(loader.ds_attribution)
    #     # wandb.log({"attributions/top1pct_attributions": wandb.Image(plot2.fig)})
    wandb.finish()


if __name__ == "__main__":
    main()