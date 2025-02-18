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
from models.losses import DistribLoss, PinballLoss
import random
import matplotlib
from models.data import get_input_data, get_input_data_small
import pandas as pd

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
num_epochs = 20
device = 'cuda:1'
input_variable = ['u850_g']
groups = 1
loss_fn = 'mse'
region_predicted = 14
type_prediction='regression'
lags = 0
input_type='small'


match loss_fn:
    case 'distrib':
        loss = DistribLoss()
    case 'mse':
        loss = torch.nn.MSELoss()
    case 'cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
    case 'pinball90':
        loss = PinballLoss(.9)
   

if type_prediction == 'quantiles':
    num_classes = 10
elif type_prediction == 'regression':
    num_classes =  1


config = {
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "num_classes": num_classes,
    "learning_rate":learning_rate,
    "input_variable" :input_variable,
    "number_variables":len(input_variable),
    "groups" : groups,
    "lags":lags,
    "region_predicted":region_predicted,
    "type_prediction":type_prediction,
    "loss_fn":loss_fn,
    "input_type":input_type,
    }

wandb_logger = WandbLogger(project="Predict-rain-WNorway", 
                           save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                           config=config, name=f"CNN-{type_prediction} {'-'.join(input_variable)} lag {lags}")
# wandb_logger = WandbLogger(project="Predict-rain-WNorway", config=config, name="Truth")

print('Loading data ...')
if input_type=='small':
    image_size = 32*32
    train_loader, valid_loader, test_loader = get_input_data_small(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)
elif input_type=='big':
    imge_size=100*256
    train_loader, valid_loader, test_loader = get_input_data(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)
 
print('Data ready')


trainer = L.Trainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                    log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/", devices=1,
                    
                    # callbacks=[MyCallback()]
                    )



CNN = Wang2024(num_classes=config['num_classes'], num_channels_in=len(config['input_variable']), image_size=image_size, 
               groups=config['groups'])

model = LitCNN(CNN, 
               type_prediction='regression', 
               learning_rate=config['learning_rate'], 
               loss = loss)




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

# if type_prediction=='regression':
#     truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
#     pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()
#     quantile_truth = np.quantile(truth,np.arange(0,1.01,.01))
#     quantile_pred = np.quantile(pred,np.arange(0,1.01,.01))
#     sorted_truth = np.sort(truth)
#     sorted_pred = np.sort(pred)

#     table = wandb.Table(data=np.array([sorted_truth,sorted_pred]).T, columns = ["Truth", "Prediction"])
#     wandb.log({"Reliability Table": table})

#     table_q = wandb.Table(data=np.array([quantile_truth,quantile_pred, (np.arange(0,1.01,.01)*100).astype(int)]).T, columns = ["Truth", "Prediction", "Percentile"])
#     wandb.log({"Percentile Table": table_q})

# trainer.save_checkpoint("z500_lags1_regression.ckpt")
# run Truth

# run = wandb.init(project="Predict-rain-WNorway", name="Truth")

truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()
truth[truth<0]=0
# pred[pred<0]=0
quantile_truth = np.quantile(truth,np.arange(0,1.01,.01))
quantile_pred = np.quantile(pred,np.arange(0,1.01,.01))
sorted_truth = np.sort(truth)
sorted_pred = np.sort(pred)


df_sorted = pd.DataFrame(np.array([sorted_truth,sorted_pred]).T, columns = ['Truth','Prediction'])
table = wandb.Table(data=df_sorted)
wandb.log({"Reliability Table": table})

table_q = wandb.Table(data=np.array([quantile_truth,quantile_pred, (np.arange(0,1.01,.01)*100).astype(int)]).T, columns = ["Truth", "Prediction", "Percentile"])
wandb.log({"Percentile Table": table_q})

# wandb.log({'test/Histogram_test': wandb.plot.histogram(table, "Prediction", title='Histogram')})



