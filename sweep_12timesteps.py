import wandb


def main():
    wandb.login()

    from predict_extreme_event import main

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "diagnostics/F1_all"},
        "parameters": {
            "learning_rate": {"max": 1e-2,'min':1e-4, 'distribution':'log_uniform_values'},
            "inputs" :{"values":[#'u850 v850',
                                 #'u850 v850 z500',
                                 'u850 v850 w850',
                                 'u850 v850 w850 tcwv',
                                #  'v850 z500 w850',
                                ]},
            "conv1_kernel_number":{"values":[12,24]},
            "num_conv_layer":{"values":[2,3,4]},
            "size_conv_kernel":{"values":[3,5,7,9]},
        },
    }



    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Predict-rain-WNorway_v4")
    wandb.agent(sweep_id, function=main, count=100)

if __name__=='__main__':
    main()