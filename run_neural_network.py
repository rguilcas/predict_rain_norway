print('Running imports...')
import os
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import wandb
import numpy as np
import torch # long
from lightning.pytorch import loggers, seed_everything
import xarray as xr
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.neuralnetworks import ConvLayerStride1_ConvLayerStride2, ConvLayerStride1MaxPool, ConvLayerStride2NoMaxPool
from src.data import MyDataLoader
from src.lightning import ExtremeRainPredictor, AttributableTrainer
from src.neuralnetworks import CNN_MLP
from src.losses import get_loss
from src.callbacks import LogF1Validation
from src.plotting import plot_mean_attributions,plot_top1pct_pixels

def main(args=None):
    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    wandb_logger = loggers.WandbLogger(project="Predict-rain-WNorway_v6", 
                                    save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                                    dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb", )

    wandb_logger.experiment # Initialize wandb
    loader = MyDataLoader(wandb_logger)
    loader.print_infos()

    callbacks=[EarlyStopping(monitor="val/loss", mode="min"), LogF1Validation()]
    trainer = AttributableTrainer(limit_train_batches=100, 
                                max_epochs=wandb_logger.experiment.config['num_epochs'], 
                                logger=wandb_logger, 
                                log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/",
                                callbacks=callbacks, deterministic=True,
                                accelerator="cpu", devices=1,)
    match wandb_logger.experiment.config['CNN_layer_module']:
        case 'ConvLayerStride1_ConvLayerStride2':
            CNN_module = ConvLayerStride1_ConvLayerStride2
        case 'ConvLayerStride1MaxPool':
            CNN_module = ConvLayerStride1MaxPool
        case 'ConvLayerStride2NoMaxPool':
            CNN_module = ConvLayerStride2NoMaxPool
            
    NN = CNN_MLP(feature_height=loader.feature_height, 
                feature_width=loader.feature_width,
                input_channels=loader.config['num_channels'],
                output_neurons=loader.config['num_classes'],
                CNN_number_of_layers=loader.config['num_conv_layer'],
                CNN_base_module = CNN_module,
                MLP_hidden_layers_neuron_number = wandb_logger.experiment.config['MLP_hidden_layers_neuron_number'], 
                use_residual = wandb_logger.experiment.config['use_skip_connections']
                )
    loss = get_loss(wandb_logger.experiment.config['loss_function'], timesteps = wandb_logger.experiment.config['num_timesteps_predicted'])
    # loss = MultiCrossEntropyLoss()
    model = ExtremeRainPredictor(NN, 
                            learning_rate=wandb_logger.experiment.config['learning_rate'], 
                            lr_scheduler =wandb_logger.experiment.config['lr_scheduler'],
                            loss_fn = loss)
    trainer.fit(model, loader.train_loader, loader.val_loader)
    model.eval()
    with torch.no_grad():
        trainer.test(model, dataloaders=loader.val_loader)
    if wandb_logger.experiment.config['attribute_true_positives']:
        loader.attribute_integrated_gradients(model, loader.ds_val, model.predictions_test, model.targets_test)
        loader.ds_attribution.to_netcdf(f"/Data/gfi/users/rogui7909/data/NN_outputs/attributions/attributions_TP_{wandb.run.id}.nc")
        # plot1 = plot_mean_attributions(loader.ds_attribution)
        # wandb.log({"attributions/mean_attribution_plot": wandb.Image(plot1.fig)})
        # plot2 = plot_top1pct_pixels(loader.ds_attribution)
        # wandb.log({"attributions/top1pct_attributions": wandb.Image(plot2.fig)})
    wandb.finish()


if __name__ == "__main__":
    main()