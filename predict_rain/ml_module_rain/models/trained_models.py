import glob
from ml_module_rain.data.datamodule import MyDataLoader
from ml_module_rain.models.neuralnetworks import get_neural_network
from ml_module_rain.models.losses import get_loss
from ml_module_rain.models.lightning import ExtremeRainPredictor
import torch




def load_trained_model(run_id):
    checkpoints_path = '/Data/gfi/users/rogui7909/data/NN_outputs/checkpoints/'
    weights_NN_file = glob.glob(f"{checkpoints_path}/{run_id}-best*")[0]
    state_dict = torch.load(weights_NN_file, weights_only=True)
    config = state_dict['hyper_parameters']

    dataloader = MyDataLoader(config)
    dataloader.print_infos()
    NN = get_neural_network(config)
    loss = get_loss(config)
    lightroom_model = ExtremeRainPredictor(NN, 
                            learning_rate=config['learning_rate'], 
                            lr_scheduler =config['lr_scheduler'],
                            loss_fn = loss,
                            init_config = config)
    lightroom_model.load_state_dict(state_dict['state_dict'])
    lightroom_model.eval()
    return dataloader, lightroom_model

    
    