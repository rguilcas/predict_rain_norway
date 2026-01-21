import yaml


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_values = extract_values(config)
    return config_values

def extract_values(config):
    if isinstance(config, dict):
        if "value" in config and set(config.keys()) == {"desc", "value"}:
            return extract_values(config["value"])
        else:
            return {k: extract_values(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [extract_values(i) for i in config]
    else:
        return config

def save_config(config, wandb_logger):
    file_name = f"{wandb_logger.experiment.id}-config.yaml"
    file_path = '/Data/gfi/users/rogui7909/data/NN_outputs/checkpoints/'
    with open(f"{file_path}/{file_name}", "w") as f:
        yaml.dump(config, f)