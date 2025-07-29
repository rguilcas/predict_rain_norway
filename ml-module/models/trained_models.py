import glob
from data.datamodule import MyDataLoader
from models.neuralnetworks import get_neural_network
from models.losses import get_loss
from models.lightning import ExtremeRainPredictor
import torch



checkpoints_path = '/Data/gfi/users/rogui7909/data/NN_outputs/checkpoints/'

def load_trained_model(run_id):
    run_id = 'hylfeyaz'
    weights_NN_file = glob.glob(f"{checkpoints_path}/{run_id}-best*")[0]
    state_dict = torch.load(weights_NN_file, weights_only=True)
    config = state_dict['hyper_parameters']

    dataloader = MyDataLoader(config)
    dataloader.print_infos()
    NN = get_neural_network(config)
    loss = get_loss(config['loss_function'], timesteps = config['num_timesteps_predicted'])
    lightroom_model = ExtremeRainPredictor(NN, 
                            learning_rate=config['learning_rate'], 
                            lr_scheduler =config['lr_scheduler'],
                            loss_fn = loss,
                            config = config)
    lightroom_model.load_state_dict(state_dict['state_dict'])
    lightroom_model.eval()
    return dataloader, lightroom_model

    
    