import wandb
print('Running imports...')
import os
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import wandb
from train_neural_network import main

def main_config():
    config_path = "/home/rogui7909/code/predict_rain_norway/predict_rain_norge/ml_module_rain/configs/config-defaults.yaml"
    main(config_path=config_path)

def main_sweep():
    wandb.login()
    from train_neural_network import main
    sweep_configuration = {
        "name": "Sweep1",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "test/best_f1"}, 
        "parameters": {
            "learning_rate": {"max": 3e-3,'min':1e-5, 'distribution':'log_uniform_values'},
            "batch_size": {"values": [128, 256, 512]},
            'CNN_num_conv_layers':{"values":[4,5,6]},
            'CNN_first_out_channels':{"values":[8, 16,24,32,]},
            'CNN_dropout': {"values": [0.0, 0.1, 0.2, 0.3]},
            'MLP_dropout': {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            'use_bn': {"values": [True, False]}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="WNorway_rainfall_attributions")
    wandb.agent(sweep_id, function=main_config, count=100)

if __name__=='__main__':
    main_sweep()