print('Running imports...')

import torch # long
import os
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import xarray as xr
from models.model import Wang2024
from lightning.pytorch.loggers import WandbLogger
import wandb
from models.lightning import LitCNN
from models.losses import DistribLoss
import random
import matplotlib
from models.data import get_input_data


matplotlib.use('tkagg')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

learning_rate = 0.001
batch_size = 128
num_epochs = 10
device = 'cuda:1'
input_variable = ['u850']
groups = 3

region_predicted = 14
type_prediction='regression'






if type_prediction == 'quantiles':
    num_classes = 10
elif type_prediction == 'regression':
    num_classes =  1


config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "num_classes": num_classes,
    "learning_rate":learning_rate,
    "input_variable" :input_variable,
    "groups" : groups,
    "region_predicted":region_predicted,
    "type_prediction":type_prediction,
    # "loss_fn":'distib',

}

wandb_logger = WandbLogger(project="Predict-rain-WNorway", config=config, name=f"CNN-{type_prediction} {'-'.join(input_variable)}")

print('Loading data ...')

train_loader, valid_loader, test_loader = get_input_data(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'])
 
print('Data ready')


trainer = L.Trainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                    log_every_n_steps=1, default_root_dir="lightning_checkpoints/", devices=1,
                    # callbacks=[MyCallback()]
                    )


CNN = Wang2024(num_classes=config['num_classes'], num_channels_in=len(config['input_variable']), image_size=data_in.shape[2]*data_in.shape[3], 
               groups=config['groups'])
model = LitCNN(CNN, type_prediction='regression', learning_rate=config['learning_rate'])




print('Model init')
trainer.fit(model, train_loader, valid_loader)


model.eval()
with torch.no_grad():
    trainer.test(model, dataloaders=test_loader)
    # predictions_test = trainer.predict(model, test_loader)


# model.test_step_y =  torch.Tensor(model.test_step_y).type(torch.long).cpu()
# model.test_step_pred =  torch.stack(model.test_step_pred).cpu()


# wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
#                                 y_true=model.test_step_y.numpy(), 
#                                 preds=model.test_step_pred.numpy(),
#                                 class_names=[k for k in range(10)]
#                                 )})

if type_prediction=='regression':
    sorted_truth = np.sort(torch.stack(model.test_step_y).cpu().numpy().squeeze())
    sorted_pred = np.sort(torch.stack(model.test_step_pred).cpu().numpy().squeeze())

    table = wandb.Table(data=np.array([sorted_truth,sorted_pred]).T, columns = ["Truth", "Prediction"])

    wandb.log(
    {
        "Reliability_diagram": wandb.plot.line(
            table, "Truth", "Prediction", title="Reliability diagram"
        )
    }
    )
