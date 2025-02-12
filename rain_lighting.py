import os
import numpy as np
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import xarray as xr
from models.model import Wang2024
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb



learning_rate = 0.001
batch_size = 128
num_epochs = 10
device = 'cuda:1'
input_variable = ['z500']

config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
    "input_variable" :['z500']
}

# Pass the config dictionary when you initialize W&B
run = wandb.init(project="el-testo", config=config, entity='coucou', name='recoucou')

# start a new wandb run to track this script



linear = nn.Sequential(nn.Linear(128*64, 64), nn.ReLU(), nn.Linear(64,10))

# Define lightning class
class LitCNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        wandb.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        wandb.log("test/loss", loss)
        
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        wandb.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


print('Imports done')




ds_ml= xr.open_dataset('/Data/gfi/users/rogui7909/era5_rain_norge/DL_era5_rain_regression_in_out.nc').sel(mask_id=[14]).sel(time=slice(None,'2021'))
ds_ml['data_in'] = ds_ml.data_in.sel(var_name=config['input_variable'])

ds_ml = ds_ml.where(ds_ml.data_in.count(['longitude', 'latitude'])==24576//3, drop=True)
ds_ml['data_out'] = (ds_ml.data_out.rank('time', pct=True)//.1).astype(int)
time = ds_ml.time 

shuffle=True
split = .8
indices = np.arange(ds_ml.time.size)
if shuffle:
    np.random.shuffle(indices)
split = int(split*ds_ml.time.size)
indices_train, indices_test = indices[:split], indices[split:]

ds_train_prevalid = ds_ml.isel(time=indices_train)
ds_test = ds_ml.isel(time=indices_test)

shuffle=True
split = .8
indices_for_valid = np.arange(ds_train_prevalid.time.size)
if shuffle:
    np.random.shuffle(indices)
split = int(split*ds_train_prevalid.time.size)
indices_train_true, indices_valid = indices_for_valid[:split], indices_for_valid[split:]

ds_train = ds_train_prevalid.isel(time=indices_train_true)
ds_valid = ds_train_prevalid.isel(time=indices_valid)


X_train = Tensor(ds_train.data_in.values).type(torch.float32)
X_test = Tensor(ds_test.data_in.values).type(torch.float32)
X_valid = Tensor(ds_valid.data_in.values).type(torch.float32)
y_train = Tensor(ds_train.data_out.values[:,0,0]).type(torch.long)
y_test = Tensor(ds_test.data_out.values[:,0,0]).type(torch.long)
y_valid = Tensor(ds_valid.data_out.values[:,0,0]).type(torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,num_workers=71)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False,num_workers=71)
valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False,num_workers=71)

print('Data ready')

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="el-testo",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learning_rate,
#     "architecture": "CNN",
#     "dataset": "ERA",
#     "epochs": num_epochs,
#     }
# )
# wandb.stop

trainer = L.Trainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, log_every_n_steps=1, default_root_dir="lightning_checkpoints/")


CNN = Wang2024(num_classes=10, num_channels_in=1, image_size=128*64, groups=1)
model = LitCNN(CNN)




print('Model init')
trainer.fit(model, train_loader, valid_loader)


model.eval()
with torch.no_grad():
    # trainer.test(model, dataloaders=test_loader)
    predictions_test = trainer.predict(model, test_loader)