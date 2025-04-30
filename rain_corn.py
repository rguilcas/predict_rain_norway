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
from models.lightning import LitCNN, AttributableTrainer
from models.losses import DistribLoss, PinballLoss
import random
import matplotlib
from models.data import get_input_data, get_input_data_small
import pandas as pd
import cartopy.crs as ccrs
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

from torch import optim
from captum.attr import IntegratedGradients

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
batch_size = 512
num_epochs = 40
device = 'cuda:1'
input_variable = [ 'u850', 'v850']
groups = 1
loss_fn = 'corn'
region_predicted = 14
type_prediction='quantiles'
lags = 1
input_type='small'
attribute_test = True

num_classes = 10

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

image_size = 32*32
train_loader, valid_loader, test_loader, ds_test = get_input_data_small(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)

wandb_logger = WandbLogger(project="Predict-rain-WNorway", 
                           save_dir="/Data/gfi/users/rogui7909/wanbd_logs/",
                           dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb",
                           config=config, name=f"CNN-{type_prediction} {'-'.join(input_variable)} lag {lags}")

trainer = AttributableTrainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                              log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/", 
                              devices=1 )

CNN = Wang2024(num_classes=config['num_classes']-1, num_channels_in=len(config['input_variable']), image_size=image_size, 
               groups=config['groups'])



class LitCNN(L.LightningModule):
    def __init__(self, model=None, 
                 type_prediction='quantiles',
                 learning_rate=1e-3,
                 loss=DistribLoss(),
                 corn_loss=False,):
        super().__init__()
        self.model = model
        self.test_step_pred = []
        self.test_step_y = []
        self.learning_rate = learning_rate
        self.corn_loss = corn_loss
        self.loss_fn = loss
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _shared_forward_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self.model(features)
        loss = corn_loss(logits, true_labels, num_classes=self.model.num_classes+1)
        predicted_labels = corn_label_from_logits(logits)
        return loss, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_forward_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x,y = batch
        loss, pred = self._shared_forward_step( batch, batch_idx)
        self.test_step_y += y
        self.test_step_pred += pred
        self.log("test/loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, pred = self._shared_forward_step( batch, batch_idx)
        self.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def forward(self, batch):
        loss, pred = self._shared_forward_step( batch, 0)
        # x = x.view(x.size(0), -1)
        return pred
    
    def attribute_step(self, batch, batch_idx, attribution_method=None, target=9, baselines=0, method_kwargs = dict()):
        x, y = batch  # Assuming batch = (inputs, labels)
        x.requires_grad = True

        if attribution_method is None:
            attribution_method = IntegratedGradients(self.model, **method_kwargs)

        attr = attribution_method.attribute(x, target=target, baselines=baselines)
        return attr.detach()
    
model = LitCNN(CNN, 
               type_prediction='regression', 
               learning_rate=config['learning_rate'], 
               loss = corn_loss,
               corn_loss=(True))




print('Model init')
trainer.fit(model, train_loader, valid_loader)


model.eval()
with torch.no_grad():
    trainer.test(model, dataloaders=test_loader)

truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=truth, preds=pred,
                    class_names=[f"{10*k:.0f}-{10*(k+1):.0f}" for k in range(10)])})


recall_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((truth==9))[0].size
precision_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((pred==9))[0].size
wandb.log({"Scores/precision9" : precision_class9*100,
           "Scores/recall9" : recall_class9*100,
           "Scores/f1_9" : 2*(precision_class9*recall_class9)/(precision_class9+recall_class9),
           })



# truth_better = torch.stack(model.test_step_y).cpu().numpy().squeeze()
# pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
# wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
#                     y_true=truth, preds=pred,
#                     class_names=[f"{10*k:.0f}-{10*(k+1):.0f}" for k in range(10)])})
truth_better = np.copy(truth)
truth_better[(truth_better>6)&(truth_better<9)] = 7
truth_better[(truth_better>1)&(truth_better<7)] = 3
truth_better[truth_better==3] = 1
truth_better[truth_better==7] = 2
truth_better[truth_better==9] = 3

pred_better = np.copy(pred)
pred_better[(pred_better>6)&(pred_better<9)] = 7
pred_better[(pred_better>1)&(pred_better<7)] = 3
pred_better[pred_better==3] = 1
pred_better[pred_better==7] = 2
pred_better[pred_better==9] = 3

wandb.log({"better_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                       y_true=truth_better, preds=pred_better,
                       class_names=["0-10%",'10-70%', '70-90%', '90-100%'])})


from captum.attr import IntegratedGradients, NoiseTunnel, DeepLiftShap, DeepLift
import hvplot.xarray
import panel as pn
predicting_9 = num_classes - 1

truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
pred = torch.argmax(torch.stack(model.test_step_pred),dim=1).cpu().numpy()

true_9_indices = np.where((truth==predicting_9)&(pred==predicting_9))[0]
data_in_true_9 = torch.Tensor(ds_test.isel(time=true_9_indices).values)
data_out_true_9 = torch.Tensor(pred[true_9_indices])
dataset_attr = TensorDataset(data_in_true_9,data_out_true_9)
attribution_loader = DataLoader(dataset_attr, batch_size=10)

baseline = torch.stack([test_loader.dataset.dataset.tensors[0].mean(dim=0)])
ig = IntegratedGradients(model.model)
nt = NoiseTunnel(ig)
dl = DeepLift(model.model)

attribution_ig = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=ig)
attribution_nt = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=nt)
attribution_dl = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=dl)



attribution_da = xr.DataArray([attribution_ig,attribution_nt,attribution_dl], 
                                dims=['attr_method']+list(ds_test.isel(time=true_9_indices).dims), 
                                coords={'attr_method':['IntegratedGradients', 'IntegratedGradients+NoiseTunnel', 'DeepLift'], **ds_test.isel(time=true_9_indices).coords})

ds_attr = xr.Dataset(dict(data = ds_test.isel(time=true_9_indices)-ds_test.mean('time'),
                            attributions = attribution_da)).sortby('time')
ds_attr_9_per_season = ds_attr.groupby('time.season').mean()


import panel as pn 
html_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.html"



season = pn.widgets.RadioButtonGroup(name="Season",description='Season',options=['DJF','MAM','JJA','SON'])
attr_method = pn.widgets.Select(description="Attribution method",options=[ str(k) for k in ds_attr_9_per_season.attr_method.values])

ds_attr_9_per_season = ds_attr_9_per_season.astype('float32')
def make_plot_attrs(var_name,season, attr_method, num_var=1):
    return ds_attr_9_per_season.attributions.isel(var_name=var_name).sel(season=season, attr_method=attr_method)\
                .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                    project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(),
                                    cmap='PuOr_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False), 
                                    title=ds_attr_9_per_season.var_name.isel(var_name=var_name).values+' attributions')

def make_plot_anomaly(var_name,season,num_var=1):
    return ds_attr_9_per_season.data.isel(var_name=var_name).sel(season=season)\
                .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
                                    project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(), 
                                    cmap='RdBu_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False),
                                    title=ds_attr_9_per_season.var_name.isel(var_name=var_name).values+' anomalies')

def plot_ones_season_method(season, method):
    plot_anomaly = make_plot_anomaly(0,season, num_var=len(input_variable))
    plot_attrs = make_plot_attrs(0,season,method,num_var=len(input_variable))
    for k in range(1, ds_attr_9_per_season.data.var_name.size):
        plot_anomaly = plot_anomaly + make_plot_anomaly(k,season, num_var=len(input_variable))
        plot_attrs =  plot_attrs +make_plot_attrs(k,season, method,num_var=len(input_variable))
    all_plots = (plot_anomaly+plot_attrs).cols(len(input_variable))  
    return all_plots 

interactive_plot = pn.bind(plot_ones_season_method, season, attr_method)

pn_layout = pn.Column(
                pn.WidgetBox(season, attr_method, horizontal=True),  
                interactive_plot
                    ).servable()
pn_layout.save(html_file,embed=True)
wandb.log({"Attribution/AttributingClass9": wandb.Html(html_file)})
