import wandb
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
from train_neural_network import main

def main_config():
    config_path = "/home/rogui7909/code/predict_rain_norway/predict_rain/ml_module_rain/configs/config-defaults.yaml"
    main(config_path=config_path)

def main_sweep():
    wandb.login()

    from train_neural_network import main

    sweep_configuration = {
    "name": "September 2nd 2025 sweep",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "test/best_f1"}, 
    "parameters": {
    "learning_rate": {"max": 1e-3,'min':1e-5, 'distribution':'log_uniform_values'},
    'use_skip_connections':{"values":[True, False]}, 
    'CNN_downsample_mode':{"values":['maxpool', 'avgpool', 'strideconv']}, 
    # 'CNN_conv_multiple':{"values":['single', 'double']},
    'num_conv_layer':{"values":[4,5,6,7]},
    'conv1_kernel_number':{"values":[8,16,24,32,64]},
    'dropout_CNN': {"min": 0., "max": 0.5},
    'batch_norm_CNN': {"values": [True, False]},
    'dropout_CNN': {"min": 0., "max": 0.5},
    'MLP_hidden_layers_neuron_number': {"values": [[128], [128,128]]},
    'dropout_MLP': {"min": 0., "max": 0.5},
    'activation_function':{"values":["ReLU","LeakyReLU","GELU"]}
    },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Predict-rain-WNorway_v7")
    wandb.agent(sweep_id, function=main_config, count=200)

if __name__=='__main__':
    main_sweep()