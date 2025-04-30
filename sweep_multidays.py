import wandb

wandb.login()

from rain_lighting_regression_multi_days import main

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "test/f1_95_day0"},
    "parameters": {
        "learning_rate": {"max": 0.01,'min':0.001, 'distribution':'log_uniform_values'},
        # "input_variable" :{"values":[['u850','v850','z500'],
        #                              ['u850','v850','z500','tcwv'],
        #                              ['u850','v850','w850','z500'], 
        #                              ['u850','tcwv','z500','w850','tcwv'], 
        #                              ]},
        "loss_fn" :{"values":["mse", "asymetric_mse", "asymetric_mse_thresh"]},
        # "conv1_kernel_number":{"values":[3,6,12,24]},
        "LSTM_hidden_size":{"values":[64,128,256,512]},
        # "model_choice":{"values":["CNN","CNN_LSTM"]},
        "num_conv_layer":{"values":[1,2,3,4]},
        "num_days_predictant":{"values":[1,2,3,4,5,6]},
        # "size_conv_kernel":{"values":[3,5,7,9]},
        # "gradient_clip_val":{"max": 1., "min": 0.1},


        # "dropout_proba":{"max": 0.01, "min": 0.}
    },
}



sweep_id = wandb.sweep(sweep=sweep_configuration, project="Predict-rain-WNorway_multidays")
wandb.agent(sweep_id, function=main, count=100)