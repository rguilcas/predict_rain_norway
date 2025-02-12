import matplotlib.backends
import torch
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import cartopy.crs as ccrs 
import matplotlib 
from tqdm.notebook import tqdm
from models.model import CNNQuantilesClassifier, CNNQuantilesClassifierDepthWise
from models.data import load_output_data, load_input_data, get_rain_bool, xarray_to_dataloaders
from models.training import train_loop, eval_loop, evaluate_model

from captum.attr import Occlusion, IntegratedGradients


matplotlib.use('TkAgg')
plt.style.use('robin')

# Load data


####### HYPERPARAMETERS #############

learning_rate = 0.001
batch_size = 256
num_epochs = 20
device = 'cuda:1'

ds_ml= xr.open_dataset('/Data/gfi/users/rogui7909/era5_rain_norge/DL_era5_rain_regression_in_out.nc').sel(mask_id=[14]).sel(time=slice(None,'2021'))
ds_ml = ds_ml.where(ds_ml.data_in.count(['longitude', 'latitude']).sum('var_name')==24576, drop=True)

time = ds_ml.time 
# time_winter = time.where(time.dt.month.isin([12,1,2]), drop=True)
# ds_ml = ds_ml.sel(time=time_winter)
# ds_ml['data_out'] = ds_ml.data_out.where(ds_ml.data_out>0,0)

# ds_ml['data_out'] = np.log(1e-5+ds_ml.data_out)
# ds_ml_mean = ds_ml.mean(['time','longitude','latitude'])
# ds_ml_std = ds_ml.std(['time','longitude','latitude'])
# ds_ml_standard = (ds_ml-ds_ml_mean)/ds_ml_std

# ds_out = xr.open_dataset('/Data/gfi/users/rogui7909/era5_rain_norge/era5_daily_rainfall_data_290125.nc').total_precipitation_day

# Make tensors


shuffle=True
split = .8
indices = np.arange(ds_ml.time.size)
if shuffle:
    np.random.shuffle(indices)
split = int(split*ds_ml.time.size)
indices_train, indices_test = indices[:split], indices[split:]

ds_train = ds_ml.isel(time=indices_train)
ds_test = ds_ml.isel(time=indices_test)


X_train = torch.tensor(ds_train.data_in.values).type(torch.float32)
X_test = torch.tensor(ds_test.data_in.values).type(torch.float32)
y_train = torch.tensor(ds_train.data_out.values.T).type(torch.float32).T
y_test = torch.tensor(ds_test.data_out.values.T).type(torch.float32).T

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)


# Make dataloader
print('Data ready')


class LeNET(nn.Module):
    def __init__(self, quantiles,num_channels_in, image_size, groups=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels_in, out_channels=12, kernel_size=3, stride=1, padding=1, groups=3)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_channels = 24
        max_pools = 4*4
        self.linear_size = image_size * cnn_channels // max_pools 
        self.fc1 = nn.Linear(self.linear_size, 128)
        self.fc2 = nn.Linear(128, quantiles)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = nn.Dropout()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = x.view((-1,self.linear_size))
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return nn.ReLU()(x)
    
class Wang2024(nn.Module):
    def __init__(self, num_classes,num_channels_in, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, groups=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, stride=1, padding=1, groups=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        cnn_channels = 48
        max_pools = 4*4
        self.linear_size = image_size * cnn_channels // max_pools 
        self.fc1 = nn.Linear(self.linear_size, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, num_classes)

    def forward(self, x):
        # print(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # print(x)
        x = self.pool1(x)
        # print(x)
        x = self.conv2(x)
        # print(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        # print(x)
        x = x.view((-1,self.linear_size))
        # print(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)

        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)

        x = self.fc3(x)
        # x = nn.ReLU()(x)
        
        return x
    


def train_loop(model, train_loader, loss_fn, optimizer, device):
    model.train()
    for X_, y_ in (train_loader):
        X_ = X_.to(device)
        y_ = y_.to(device)
        pred = model(X_)
        loss = loss_fn(pred, y_)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loop(model, test_loader, loss_fn, device):
    model.eval()
    all_pred = []
    all_y = []
    with torch.no_grad():
        for X_, y_ in test_loader:
            X_ = X_.to(device)
            y_ = y_.to(device)
            pred = model(X_)
            all_pred.append(pred)
            all_y.append(y_)
        pred = torch.cat(all_pred)
        y_ = torch.cat(all_y)
        loss = loss_fn(pred, y_)
    return loss.cpu()


class DistribLoss(nn.Module):
    def __init__(self):
        """
        Initialize the quantile loss module.
        :param quantile: The quantile to estimate (e.g., 0.5 for median, 0.95 for 95th percentile).
        """
        super(DistribLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the quantile loss.
        :param predictions: Predicted values (torch.Tensor).
        :param targets: Ground truth values (torch.Tensor).
        :return: The quantile loss (torch.Tensor).
        """
        # errors = targets - predictions
        # loss = torch.maximum(
        #     self.quantile * errors,
        #     (self.quantile - 1) * errors
        # )
        targets_sorted, _ = torch.sort(targets)
        predictions_sorted, _ = torch.sort(predictions)
        errors_cdf = torch.abs(targets_sorted - predictions_sorted)[batch_size//2:]**2
        errors = torch.abs(targets - predictions)**2
        loss = torch.mean(errors_cdf) + torch.mean(errors)
        return loss.mean()

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        """
        Initialize the quantile loss module.
        :param quantile: The quantile to estimate (e.g., 0.5 for median, 0.95 for 95th percentile).
        """
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, predictions, targets):
        """
        Compute the quantile loss.
        :param predictions: Predicted values (torch.Tensor).
        :param targets: Ground truth values (torch.Tensor).
        :return: The quantile loss (torch.Tensor).
        """
        errors = targets - predictions
        loss = torch.maximum(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
      
        return loss.mean()

distrib_loss = DistribLoss()
quant_loss = QuantileLoss(0.95)


torch.manual_seed(12)
torch.use_deterministic_algorithms(False)


model = Wang2024(num_classes=1, num_channels_in=3, image_size=128*64).to(device)


loss_fn = distrib_loss.to(device)
# loss_fn = quant_loss.to(device)
# loss_fn = nn.MSELoss().to(device) 


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

history = []
for epoch in range(num_epochs):
    train_loop(model, train_loader, loss_fn, optimizer, device)
    loss_train = eval_loop(model, train_loader, loss_fn, device)
    loss_test = eval_loop(model, test_loader, loss_fn, device)
    print(f'Epoch {epoch+1}:    train loss = {loss_train:>8.02f}    test loss = {loss_test:>8.02f}')
    history.append([loss_train, loss_test])
    # if epoch == num_epochs//2:
    #     loss_fn = distrib_loss.to(device)
history = np.array(history)

# torch.save(model.state_dict(), 'trained_models/wang2024_ERA5_norwaeyW_precip.torch')
# Evaluate model

model.eval()
with torch.no_grad():
    model = model.cpu()
    data_in = ds_test.data_in.values.squeeze()
    X = torch.tensor(data_in).type(torch.float32)
    pred = model(X).detach().numpy()

    pred_ds = xr.DataArray(pred, dims = ['time','mask_id'], coords=ds_test.data_out.coords)
    ds_eval = xr.Dataset(dict(truth=ds_test.data_out, pred=pred_ds)).sortby('time')

# ds_eval_destandard = ds_eval*ds_ml_std + ds_ml_mean
# ds_eval_destandard = np.exp(ds_eval) + 1e-5
ds_eval.truth.isel(time=slice(0,100)).plot()
ds_eval.pred.isel(time=slice(0,100)).plot()
plt.show()



import seaborn.objects as so
df_ = ds_eval.to_dataframe()
ds_bins = df_.groupby([pd.cut(df_.truth, np.arange(0,41,.5)),
             pd.cut(df_.pred, np.arange(0,41,.5)),
             ], observed=True).count().truth.to_xarray()

from matplotlib.colors import LogNorm

norm =  LogNorm(vmin=1, vmax=ds_bins.max())
fig, axs =plt.subplots(1,2,figsize=(6,3), dpi=200)
ds_bins.where(ds_bins>0).T.plot(norm = norm, ax=axs[0], add_colorbar=False)

sorted_df_ = df_.transform(np.sort)
sorted_df_.plot.scatter(y='pred',x='truth', ax=axs[1], s=5, color='.5', zorder=100)

for ax in axs:
    ax.set_aspect('equal')
    ax.set(xlim=(0,30), ylim=(0,30))
    ax.axline((0,0), slope=1, zorder=8, color='C3', lw=2)

axs[0].set_title(f"Individual predictions\nr={df_.corr().truth.pred:.03f}  RMSE={np.sqrt(((df_.truth - df_.pred)**2).mean()):.02f}mm/day")
axs[1].set_title(f"Reliability diagram\nr={sorted_df_.corr().truth.pred:.03f}")

plt.show()