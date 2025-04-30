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
from models.lightning import LitCNN_quantiles, AttributableTrainer, ConfusionMatrix, BetterConfusionMatrix, PrecisionRecallClass9
from models.losses import DistribLoss, PinballLoss, WeightedOrdinalLoss, logits_to_class_probs
import random
import matplotlib
from models.data import get_input_data, get_input_data_small
import pandas as pd
import cartopy.crs as ccrs
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from models.attributions import log_attributions

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
batch_size = 256
num_epochs = 20
device = 'cuda:1'
input_variable = [ 'u850','v850','z500','tcwv']
groups = 2
loss_fn = 'corn'
region_predicted = 14
type_prediction='quantiles'
lags = 0
input_type='small'
attribute_test = True

match loss_fn:
    case 'distrib':
        loss = DistribLoss()
    case 'mse':
        loss = torch.nn.MSELoss()
    case 'cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
    case 'weighted_cross_entropy':
        loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,1,1,1,1,1,1,1,1,1]))
    case 'pinball90':
        loss = PinballLoss(.9)
    case 'corn':
        loss = corn_loss
    case 'weighted_ordinal':
        loss = WeightedOrdinalLoss(num_classes=10, extreme_weight=1)

   

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
                           dir="/Data/gfi/users/rogui7909/wanbd_logs/wandb",
                           config=config, name=f"CNN-{type_prediction} {'-'.join(input_variable)} lag {lags}")
# wandb_logger = WandbLogger(project="Predict-rain-WNorway", config=config, name="Truth")

print('Loading data ...')
if input_type=='small':
    image_size = 32*32
    train_loader, valid_loader, test_loader, ds_test = get_input_data_small(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)
elif input_type=='big':
    image_size=100*256
    train_loader, valid_loader, test_loader = get_input_data(config['input_variable'], config['region_predicted'], config['type_prediction'], config['batch_size'], lags=lags)
 
print('Data ready')

callbacks=[ConfusionMatrix(),
        #    BetterConfusionMatrix(), 
           PrecisionRecallClass9()]

trainer = AttributableTrainer(limit_train_batches=100, max_epochs=num_epochs, logger=wandb_logger, 
                              log_every_n_steps=1, default_root_dir="/Data/gfi/users/rogui7909/lightning_checkpoint/", devices=1,
                              callbacks=callbacks)

if 'corn' in str(loss) or 'WeightedOrdinalLoss' in str(loss):
    num_output_neurons = config['num_classes'] - 1
else:
    num_output_neurons = config['num_classes'] 

CNN = Wang2024(num_classes=num_output_neurons, num_channels_in=len(config['input_variable']), image_size=image_size, 
               groups=config['groups'])

model = LitCNN_quantiles(CNN, 
                         learning_rate=config['learning_rate'], 
                         loss_fn = loss)

print('Model init')
trainer.fit(model, train_loader, valid_loader)


model.eval()
with torch.no_grad():
    trainer.test(model, dataloaders=test_loader)
    # predictions_test = trainer.predict(model, test_loader)

if attribute_test:
    target_max = num_output_neurons-1
    ds_attr = trainer.get_multi_attribution_ds(test_loader, ds_test, target=target_max)
    log_attributions(ds_attr, wandb_logger, input_variable)


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
#     truth[truth<0]=0
#     # pred[pred<0]=0
#     quantile_truth = np.quantile(truth,np.arange(0,1.01,.01))
#     quantile_pred = np.quantile(pred,np.arange(0,1.01,.01))
#     sorted_truth = np.sort(truth)
#     sorted_pred = np.sort(pred)


#     df_sorted = pd.DataFrame(np.array([sorted_truth,sorted_pred]).T, columns = ['Truth','Prediction'])
#     table = wandb.Table(data=df_sorted)
#     wandb.log({"Reliability Table": table})

#     table_q = wandb.Table(data=np.array([quantile_truth,quantile_pred, (np.arange(0,1.01,.01)*100).astype(int)]).T, columns = ["Truth", "Prediction", "Percentile"])
#     wandb.log({"Percentile Table": table_q})

# elif type_prediction=='quantiles':
#     truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
#     pred = torch.stack(model.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
#     wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
#                         y_true=truth, preds=pred,
#                         class_names=[f"{10*k:.0f}-{10*(k+1):.0f}" for k in range(10)])})
#     recall_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((truth==9))[0].size
#     precision_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((pred==9))[0].size
#     wandb.log({"Scores/precision9" : precision_class9*100,
#             "Scores/recall9" : precision_class9*100,
#             "Scores/precision9" : (precision_class9*recall_class9)/(precision_class9+recall_class9),
#             })


# if attribute_test and type_prediction=='quantiles':
#     from captum.attr import IntegratedGradients, NoiseTunnel, DeepLiftShap, DeepLift
#     import hvplot.xarray
#     import panel as pn
#     predicting_9 = num_classes - 1
    
#     truth = torch.stack(model.test_step_y).cpu().numpy().squeeze()
#     pred = torch.argmax(torch.stack(model.test_step_pred),dim=1).cpu().numpy()
    
#     true_9_indices = np.where((truth==predicting_9)&(pred==predicting_9))[0]
#     data_in_true_9 = torch.Tensor(ds_test.isel(time=true_9_indices).values)
#     data_out_true_9 = torch.Tensor(pred[true_9_indices])
#     dataset_attr = TensorDataset(data_in_true_9,data_out_true_9)
#     attribution_loader = DataLoader(dataset_attr, batch_size=10)

#     baseline = torch.stack([test_loader.dataset.dataset.tensors[0].mean(dim=0)])
#     ig = IntegratedGradients(model.model)
#     nt = NoiseTunnel(ig)
#     dl = DeepLift(model.model)

#     attribution_ig = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=ig)
#     attribution_nt = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=nt)
#     attribution_dl = trainer.attribute(attribution_loader, baselines=baseline, target=predicting_9, attribution_method=dl)



#     attribution_da = xr.DataArray([attribution_ig,attribution_nt,attribution_dl], 
#                                   dims=['attr_method']+list(ds_test.isel(time=true_9_indices).dims), 
#                                   coords={'attr_method':['IntegratedGradients', 'IntegratedGradients+NoiseTunnel', 'DeepLift'], **ds_test.isel(time=true_9_indices).coords})
    
#     ds_attr = xr.Dataset(dict(data = ds_test.isel(time=true_9_indices)-ds_test.mean('time'),
#                               attributions = attribution_da)).sortby('time')
#     ds_attr_9_per_season = ds_attr.groupby('time.season').mean()


#     import panel as pn 
#     html_file = f"/Data/gfi/users/rogui7909/wanbd_logs/{wandb_logger.experiment.id}_attributions.html"
    


#     season = pn.widgets.RadioButtonGroup(name="Season",description='Season',options=['DJF','MAM','JJA','SON'])
#     attr_method = pn.widgets.Select(description="Attribution method",options=[ str(k) for k in ds_attr_9_per_season.attr_method.values])

#     ds_attr_9_per_season = ds_attr_9_per_season.astype('float32')
#     def make_plot_attrs(var_name,season, attr_method, num_var=1):
#         return ds_attr_9_per_season.attributions.isel(var_name=var_name).sel(season=season, attr_method=attr_method)\
#                     .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
#                                         project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(),
#                                         cmap='PuOr_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False), 
#                                         title=ds_attr_9_per_season.var_name.isel(var_name=var_name).values+' attributions')

#     def make_plot_anomaly(var_name,season,num_var=1):
#         return ds_attr_9_per_season.data.isel(var_name=var_name).sel(season=season)\
#                     .hvplot.quadmesh(x='longitude', y='latitude', geo=True, coastline=True,
#                                      project=True, projection=ccrs.PlateCarree(0), crs=ccrs.PlateCarree(), 
#                                      cmap='RdBu_r', symmetric=True,frame_width=200, colorbar=(True if var_name==num_var-1 else False),
#                                      title=ds_attr_9_per_season.var_name.isel(var_name=var_name).values+' anomalies')

#     def plot_ones_season_method(season, method):
#         plot_anomaly = make_plot_anomaly(0,season, num_var=len(input_variable))
#         plot_attrs = make_plot_attrs(0,season,method,num_var=len(input_variable))
#         for k in range(1, ds_attr_9_per_season.data.var_name.size):
#             plot_anomaly = plot_anomaly + make_plot_anomaly(k,season, num_var=len(input_variable))
#             plot_attrs =  plot_attrs +make_plot_attrs(k,season, method,num_var=len(input_variable))
#         all_plots = (plot_anomaly+plot_attrs).cols(len(input_variable))  
#         return all_plots 

#     interactive_plot = pn.bind(plot_ones_season_method, season, attr_method)
    
#     pn_layout = pn.Column(
#                     pn.WidgetBox(season, attr_method, horizontal=True),  
#                     interactive_plot
#                       ).servable()
#     pn_layout.save(html_file,embed=True)
#     wandb.log({"Attribution/AttributingClass9": wandb.Html(html_file)})

    # pn_layout.servable(title="Attribution_plot", filename=html_file)

    # wandb.log({'test/Histogram_test': wandb.plot.histogram(table, "Prediction", title='Histogram')})


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



