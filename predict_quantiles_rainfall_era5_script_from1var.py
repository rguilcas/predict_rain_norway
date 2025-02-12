import matplotlib.backends
import torch
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib 
from tqdm.notebook import tqdm
from models.model import Wang2024
from models.data import load_output_data, load_input_data, get_rain_bool, xarray_to_dataloaders
from models.training import train_loop, eval_loop, evaluate_model



matplotlib.use('TkAgg')
plt.style.use('robin')

# Load data


####### HYPERPARAMETERS #############

learning_rate = 0.0001
batch_size = 128
num_epochs = 100
device = 'cuda:1'

ds_ml= xr.open_dataset('/Data/gfi/users/rogui7909/era5_rain_norge/DL_era5_rain_regression_in_out.nc').sel(mask_id=[14]).sel(time=slice(None,'2021'))
ds_ml['data_in'] = ds_ml.data_in.sel(var_name=['z500'])

ds_ml = ds_ml.where(ds_ml.data_in.count(['longitude', 'latitude'])==24576//3, drop=True)
ds_ml['data_out'] = (ds_ml.data_out.rank('time', pct=True)//.1).astype(int)
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
y_train = torch.tensor(ds_train.data_out.values[:,0,0]).type(torch.long)
y_test = torch.tensor(ds_test.data_out.values[:,0,0]).type(torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)


# Make dataloader
print('Data ready')

torch.manual_seed(12)
torch.use_deterministic_algorithms(False)


model = Wang2024(num_classes=10, num_channels_in=1, image_size=128*64, groups=1).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

history = []
for epoch in range(num_epochs):
    train_loop(model, train_loader, loss_fn, optimizer, device)
    loss_train = eval_loop(model, train_loader, loss_fn, device)
    loss_test = eval_loop(model, test_loader, loss_fn, device)
    print(f'Epoch {epoch+1}:    train loss = {loss_train:>8.02f}    test loss = {loss_test:>8.02f}')
    history.append([loss_train, loss_test])

history = np.array(history)

# torch.save(model.state_dict(), 'trained_models/wang2024_z500_ERA5_norwaeyW_precip_10quantiles.torch')
# torch.save(model.state_dict(), 'trained_models/wang2024_z500_ERA5_norwaeyW_precip_10quantiles.torch')

# Evaluate model

model.eval()
with torch.no_grad():
    model = model.cpu()
    data_in = ds_test.data_in.values
    X = torch.tensor(data_in).type(torch.float32)
    pred = torch.softmax(model(X), dim=1).detach().numpy()

    pred_ds = xr.DataArray(pred, dims = ['time','quantile'], coords=dict(time=ds_test.time, quantile=np.arange(0,10)))
    ds_eval = xr.Dataset(dict(truth=ds_test.data_out.squeeze(), pred_proba=pred_ds, pred=pred_ds.idxmax('quantile'))).sortby('time')


ds_eval.truth.isel(time=slice(0,100)).plot()
ds_eval.pred.isel(time=slice(0,100)).plot()
plt.show()

df_ = ds_eval[['truth','pred']].to_dataframe()

sns.heatmap(df_.groupby(['truth','pred']).count().unstack().mask_id,
            square=True, cbar=False, annot=True, fmt = '.0f')
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

axs[0].set_title(f"Individual predictions\nr={df_.corr(numeric_only=True).truth.pred:.03f}  RMSE={np.sqrt(((df_.truth - df_.pred)**2).mean()):.02f}mm/day")
axs[1].set_title(f"Reliability diagram\nr={sorted_df_.corr(numeric_only=True).truth.pred:.03f}")

plt.show()