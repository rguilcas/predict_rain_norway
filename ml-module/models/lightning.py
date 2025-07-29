import lightning as L
from utils.losses import DistribLoss
from torch import optim
from captum.attr import IntegratedGradients, NoiseTunnel
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


class ExtremeRainPredictor(L.LightningModule):
    def __init__(self, model=None, 
                 learning_rate=1e-3,
                 loss_fn=DistribLoss(), 
                 lr_scheduler = 'exponential', 
                 config=dict()):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.predictions_validation = []
        self.targets_validation = []
        self.predictions_test = []
        self.targets_test = []
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        if lr_scheduler=='exponential':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        elif lr_scheduler=='step':
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=90, gamma=0.1)
        else:
            lr_scheduler = None
        self.scheduler = lr_scheduler

    
    def compute_loss(self, predictions, true_values):
        return self.loss_fn(predictions, true_values)
        
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    
    def _shared_forward_step(self, batch, batch_idx):
        features, true_values = batch
        predictions = self.model(features)
        loss = self.compute_loss(predictions, true_values)
        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, *_ = self._shared_forward_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, predictions = self._shared_forward_step( batch, batch_idx)
        features, true_values = batch
        loss, predictions = self._shared_forward_step( batch, batch_idx)
        self.targets_test += true_values
        self.predictions_test += predictions
        self.log("test/loss", loss)
        return loss
    def on_validation_start(self):
        self.predictions_validation = []
        self.targets_validation = []

    def validation_step(self, batch, batch_idx):
        features, true_values = batch
        loss, predictions = self._shared_forward_step( batch, batch_idx)
        self.targets_validation += true_values
        self.predictions_validation += predictions
        self.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def forward(self, features):
        predictions = self.model(features)
        return predictions
    
    def attribute_step(self, batch, batch_idx, attribution_method=None, target=9, baselines=0, method_kwargs = dict()):
        features, true_values = batch
        features.requires_grad = True
        if attribution_method is None:
            attribution_method = IntegratedGradients(self.model, **method_kwargs)
        attribution = attribution_method.attribute(features, target=target, baselines=baselines)
        return attribution.detach()
    
    def on_test_start(self):
        self.predictions_test = []
        self.targets_test = []
        # self.test_true_values = torch.stack(self.test_true_values).cpu().numpy()
        # self.test_pred = torch.stack(self.test_pred).cpu().numpy()
    


    

class AttributableTrainer(L.Trainer):
    def attribute(self, dataloader, attribution_method=None, target=0, baselines=0, method_kwargs = dict()):
        """
        Adds attribute method to trainer
        """
        self.model.eval()
        attributions = []
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Attributions")
        with torch.no_grad():
            for batch_idx, batch in progress_bar:
                attr = self.model.attribute_step(batch, batch_idx, attribution_method, target, baselines = baselines, method_kwargs=method_kwargs)
                attributions.append(attr.cpu())
        return torch.cat(attributions)
    
    def get_multi_attribution_ds_regr(self, test_loader, ds_test, min_rain=30):
        truth = self.model.test_true_values
        pred = self.model.test_pred
        true_indices = np.where((truth>=min_rain)&(pred>=min_rain))[0]
        data_in_true_target = torch.cat([batch[0] for batch in test_loader])[true_indices]
        data_out_true_target = torch.Tensor(pred[true_indices])
        dataset_attr = TensorDataset(data_in_true_target,data_out_true_target)
        attribution_loader = DataLoader(dataset_attr, batch_size=10)

        baseline = torch.stack([torch.cat([torch.zeros_like(batch[0]) for batch in test_loader]).mean(dim=0)])
        ig = IntegratedGradients(self.model.model)
        nt = NoiseTunnel(ig)
        # dl = DeepLift(self.model.model)

        attribution_ig = self.attribute(attribution_loader, baselines=baseline, target=0, attribution_method=ig)
        # attribution_nt = self.attribute(attribution_loader, baselines=baseline, target=0, attribution_method=nt)
        # attribution_dl = self.attribute(attribution_loader, baselines=baseline, target=0, attribution_method=dl)

        attribution_da = xr.DataArray([attribution_ig], 
                                    dims=['attr_method']+list(ds_test.dims), 
                                    coords={'attr_method':['IntegratedGradients', 
                                                        #    'IntegratedGradients+NoiseTunnel'
                                                           ], 
                                            **ds_test.isel(time=true_indices).coords})
        pred_da = xr.DataArray(pred[true_indices], dims=['time'], coords=dict(time=attribution_da.time))
        truth_da = xr.DataArray(truth[true_indices], dims=['time'], coords=dict(time=attribution_da.time))

        ds_attr = xr.Dataset(dict(data = ds_test.isel(time=true_indices)-ds_test.mean('time'),
                                  attributions = attribution_da,
                                  pred=pred_da, truth = truth_da)).sortby('time').assign_coords(min_rain=min_rain)
        ds_attr.attrs['baseline_prediction'] = self.model(baseline).detach().numpy().squeeze()
        return ds_attr

        


# class Over95PrecisionRecall(Callback):
#     def on_test_end(self, trainer, pl_module):
#         truth = torch.cat(pl_module.test_true_values.cpu().numpy().squeeze()
#         pred = pl_module.test_pred.cpu().numpy().squeeze()#.argmax(axis=1)
#         df = pd.DataFrame(dict(truth=truth, pred=pred))
#         table = wandb.Table(dataframe=df)
#         sorted_table = wandb.Table(dataframe=df.transform(np.sort))

#         histogram_rain = wandb.plot.histogram(table, "pred",title="Predicted Rain Distribution")

#         r_value = float(df.corr().truth.pred)
#         rmse = float(np.sqrt(((df.truth - df.pred)**2).mean()))
#         #
#         wandb.log({'histogram_precip_predicted': histogram_rain,
#                    'sorted_predictions':sorted_table,
#                    'test/r_value' : r_value,
#                    'test/rmse' : rmse
#                    })
        

