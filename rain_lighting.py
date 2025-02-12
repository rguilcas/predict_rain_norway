import os
import numpy as np
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
from models.model import Wang2024
import torch


# Define lightning class
class LitCNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = nn.CrossEntropyLoss(pred, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

print('Imports done')



model = LitCNN(Wang2024, num_classes=10, num_channels_in=1, image_size=128*64, groups=1)
print('Model init')
learning_rate = 0.0001
batch_size = 128
num_epochs = 100
device = 'cuda:1'

ds_ml= xr.open_dataset('/Data/gfi/users/rogui7909/era5_rain_norge/DL_era5_rain_regression_in_out.nc').sel(mask_id=[14]).sel(time=slice(None,'2021'))
ds_ml['data_in'] = ds_ml.data_in.sel(var_name=['z500'])

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

ds_train = ds_ml.isel(time=indices_train)
ds_test = ds_ml.isel(time=indices_test)


X_train = Tensor(ds_train.data_in.values).type(torch.float32)
X_test = Tensor(ds_test.data_in.values).type(torch.float32)
y_train = Tensor(ds_train.data_out.values[:,0,0]).type(torch.long)
y_test = Tensor(ds_test.data_out.values[:,0,0]).type(torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)

