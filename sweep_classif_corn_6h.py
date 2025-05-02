import wandb


def main():
    wandb.login()

    from classif_with_corn import main

    sweep_configuration = {
        "name": "Sweep classif 6H for 1 timestep",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "diagnostics/f1_90th"},
        "parameters": {
            "learning_rate": {"max": 1e-2,'min':1e-5, 'distribution':'log_uniform_values'},
            "inputs" :{"values":['u850 v850',
                                 'u850 v850 z500',
                                 'u850 v850 w850',
                                 'u850 v850 w850 tcwv',
                                #  'v850 z500 w850',
                                ]},
            "conv1_kernel_number":{"values":[12,24]},
            "num_conv_layer":{"values":[2,3,4]},
            "size_conv_kernel":{"values":[3,5,7,9]},
        },
    }



    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Predict-rain-WNorway_backtoclassif")
    wandb.agent(sweep_id, function=main, count=100)

if __name__=='__main__':
    main()