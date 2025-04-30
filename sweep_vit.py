import wandb


def main():
    wandb.login()

    from predict_extreme_event_vit import main

    sweep_configuration = {
        "name": "ViT sweep",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "diagnostics/auc_pr"},
        "parameters": {
            "learning_rate": {"max": 1e-1,'min':1e-5, 'distribution':'log_uniform_values'},
            "inputs" :{"values":['u850 v850 w850',
                                 'u850 v850 w850 tcwv',
                                ]},
            'patch_size':{"values":[8,16,32]},            # must divide image_size exactly
            'embedding_dimension':{"values":[128,256,512,1024]},          # embedding dimension
            'depth_transformer':{"values":[4,5,6,7,8,9,10]},          # number of transformer blocks
            'attention_heads':{"values":[4,8,16,32]},            # number of attention heads
            'mlp_dim':{"values":[128,256,512,1024]},                  # feedforward layer size
            'dropout':{"min": 0., "max": 0.5},
            'emb_dropout':{"min": 0., "max": 0.5}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Predict-rain-WNorway_test")
    wandb.agent(sweep_id, function=main, count=100)

if __name__=='__main__':
    main()