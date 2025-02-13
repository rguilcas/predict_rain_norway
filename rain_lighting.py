import os
import numpy as np
from torch import optim, nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import xarray as xr
from models.model import Wang2024
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import Callback
import random
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import matplotlib

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

learning_rate = 0.0001
batch_size = 128
num_epochs = 10
device = 'cuda:1'
input_variable = ['u850', 'v850']


region_predicted = 14
type_prediction='regression'






if type_prediction == 'quantiles':
    num_classes = 10
elif type_prediction == 'regression':
    num_classes =  1


variable_order = dict(z500=0,
                      u850=1,
                      v850=2,
                      tcwv=3)


process_variables = dict(z500='standard',
                         u850='std_only',
                         v850='std_only',
                         tcwv='standard',
                            )
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "batch_size": 256,
    "num_epochs": num_epochs,
    "num_classes": num_classes,
    "learning_rate":learning_rate,
    "input_variable" :input_variable,
    "groups" : 1,
    "region_predicted":region_predicted,
    "tpye_prediction":type_prediction

}

wandb_logger = WandbLogger(project="Predict-rain-WNorway", config=config, name=f'CNN-{type_prediction}')


lon_extent = slice(-90,89.5)
lat_extent = slice(20,90)

print('loading data ...')

all_data_in = []
for variable in config['input_variable']:
    with xr.open_dataset(f'/Data/gfi/users/rogui7909/data/ERA5/{variable}.nc') as ds:
        data_ = ds.sel(latitude=lat_extent, longitude=lon_extent)[variable].values
        time = ds.time
        if process_variables[variable] == 'standard':
            data_ = (data_ - ds[f'mean_{variable}'].values)/ds[f'std_{variable}'].values
        elif process_variables[variable] == 'std_only':
            data_ = (data_)/ds[f'std_{variable}'].values
        all_data_in.append(data_)
        print('   --> ',variable,'loaded')
data_in = np.array(all_data_in)
data_in = data_in.transpose(1,0,2,3)
print('Input data ready')

tensor_in = torch.Tensor(data_in).type(torch.float32)
# tensor_in = torch.load("/Data/gfi/users/rogui7909/data/ERA5/tensor_era5_in.pt", weights_only=True)


with xr.open_dataarray('/Data/gfi/users/rogui7909/era5_rain_norge/ERA5land_daily_TP_indices.nc', chunks = dict(longitude=20, latitude=20)) as ds_out:
    data_out = ds_out.sel(mask_id=region_predicted)
    if type_prediction=='quantiles':
        data_out = (data_out.rank('time', pct=True)//.1).astype(int)
    data_out = data_out.values

tensor_out = torch.Tensor(data_out).type(torch.float32)
if type_prediction=='quantiles':
    tensor_out = tensor_out.type(torch.long)

dataset = TensorDataset(tensor_in, tensor_out)

# use 20% of data for test
trainvalid_set_size = int(len(dataset) * 0.8)
test_set_size = len(dataset) - trainvalid_set_size
# split the train set into two
seed = torch.Generator().manual_seed(42)
trainvalid_set, test_set = random_split(dataset, [trainvalid_set_size, test_set_size], generator=seed)

# use 20% of training data for validation
train_set_size = int(len(trainvalid_set) * 0.8)
valid_set_size = len(trainvalid_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(43)
train_set, valid_set = random_split(trainvalid_set, [train_set_size, valid_set_size], generator=seed)


train_loader = DataLoader(train_set, batch_size=batch_size)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


print('Data ready')


# Define lightning class
class LitCNN(L.LightningModule):
    def __init__(self, model, type_prediction='quantiles'):
        super().__init__()
        self.model = model
        self.test_step_pred = []
        self.test_step_y = []
        if type_prediction=='quantiles':
            self.loss_fn = nn.CrossEntropyLoss()
        elif type_prediction == 'regression':
            self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.test_step_y += y
        self.test_step_pred += pred

        self.log("test/loss", loss)
        return x, pred
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    

# print('Imports done')



class MyCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        pl_module.test_step_y =  torch.Tensor(pl_module.test_step_y).type(torch.long)
        pl_module.test_step_pred =  torch.stack(pl_module.test_step_pred)
        # print(pl_module.test_step_pred)
        # do something with all training_step outputs, for example:
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                y_true=pl_module.test_step_y.numpy(), 
                                preds=pl_module.test_step_pred.numpy(),
                                class_names=[k for k in range(10)]
                                )})

        # df_ =  pd.DataFrame(dict(truth=pl_module.test_step_y, pred=pl_module.test_step_pred))

        # fig, ax = plt.subplots(figsize=(6,6))
        # sns.heatmap(df_.groupby(['truth','pred']).count().unstack(),
        #             square=True, cbar=False, annot=True, fmt = '.0f', ax=ax)
        # wandb.log({ 'confusion_matrix' : wandb.Image(fig) })
        

trainer = L.Trainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                    log_every_n_steps=1, default_root_dir="lightning_checkpoints/", devices=1,
                    # callbacks=[MyCallback()]
                    )


CNN = Wang2024(num_classes=config['num_classes'], num_channels_in=len(config['input_variable']), image_size=data_in.shape[2]*data_in.shape[3], 
               groups=config['groups'])
model = LitCNN(CNN, type_prediction='regression')




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
    sorted_truth = np.sort(torch.stack(model.test_step_y).cpu().numpy())
    sorted_pred = np.sort(torch.stack(model.test_step_pred).cpu().numpy()).squeeze()

    table = wandb.Table(data=np.array([sorted_truth,sorted_pred]).T, columns = ["Truth", "Prediction"])

    wandb.log({"Reliability diagram" : wandb.plot.scatter(table, "Truth", "Prediction",
                                    title="Reliability diagram")})




