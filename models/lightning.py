import lightning as L
from lightning.pytorch.callbacks import Callback
from models.losses import DistribLoss, logits_to_class_probs
from models.model import Wang2024
from torch import Tensor, nn, optim, stack, long
import wandb
from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import torch
from tqdm import tqdm
import numpy as np
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
import pandas as pd


class LitCNN_regression(L.LightningModule):
    def __init__(self, model=None, 
                 learning_rate=1e-3,
                 loss_fn=DistribLoss()):
        super().__init__()
        self.model = model
        self.test_step_pred = []
        self.test_step_true_values = []
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
    
    def compute_loss(self, predictions, true_values):
        return self.loss_fn(predictions, true_values)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer
    
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
        features, true_values = batch
        loss, predictions = self._shared_forward_step( batch, batch_idx)
        self.test_step_true_values += true_values
        self.test_step_pred += predictions
        self.log("test/loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, *_ = self._shared_forward_step( batch, batch_idx)
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
    
class LitCNN_quantiles(L.LightningModule):
    def __init__(self, model=None, 
                 learning_rate=1e-3,
                 loss_fn=DistribLoss()):
        super().__init__()
        self.model = model
        self.test_step_pred = []
        self.test_step_true_labels = []
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        if 'corn' in str(self.loss_fn):
            print('Corn loss detected')
            self.ordinal_loss='corn' 
        elif 'WeightedOrdinalLoss' in str(self.loss_fn):
            print('WeightedOrdinalLoss loss detected')
            self.ordinal_loss='WeightedOrdinal'
        else:
            self.ordinal_loss=False 

    def prediction_from_logits(self,logits):
        if self.ordinal_loss=='corn':
            return corn_label_from_logits(logits)
        elif self.ordinal_loss=='WeightedOrdinal':
            class_probs = logits_to_class_probs(logits)
            return torch.argmax(class_probs, dim=1)
        return torch.argmax(logits, dim=1)

    def compute_loss(self, logits, true_labels):
        if self.ordinal_loss=='corn':
            return corn_loss(logits, true_labels, num_classes=self.model.num_classes+1)
        return self.loss_fn(logits, true_labels)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _shared_forward_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self.model(features)
        loss = self.compute_loss(logits, true_labels)
        predicted_labels = self.prediction_from_logits(logits)
        return loss, logits, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, *_ = self._shared_forward_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        features, true_labels = batch
        loss, logits, predicted_labels = self._shared_forward_step( batch, batch_idx)
        self.test_step_true_labels += true_labels
        self.test_step_pred += predicted_labels
        self.log("test/loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, *_ = self._shared_forward_step( batch, batch_idx)
        self.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def forward(self, features):
        logits = self.model(features)
        predicted_labels = self.prediction_from_logits(logits)
        return predicted_labels
    
    def attribute_step(self, batch, batch_idx, attribution_method=None, target=9, baselines=0, method_kwargs = dict()):
        features, true_labels = batch
        features.requires_grad = True
        if attribution_method is None:
            attribution_method = IntegratedGradients(self.model, **method_kwargs)
        attribution = attribution_method.attribute(features, target=target, baselines=baselines)
        return attribution.detach()
    

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
    
    def get_multi_attribution_ds_quantiles(self, test_loader, ds_test, target=0):
        truth = torch.stack(self.model.test_step_true_labels).cpu().numpy().squeeze()
        pred = torch.stack(self.model.test_step_pred).cpu().numpy()
        
        true_indices = np.where((truth==target)&(pred==target))[0]
        data_in_true_target = torch.Tensor(ds_test.isel(time=true_indices).values)
        data_out_true_target = torch.Tensor(pred[true_indices])
        dataset_attr = TensorDataset(data_in_true_target,data_out_true_target)
        attribution_loader = DataLoader(dataset_attr, batch_size=10)

        baseline = torch.stack([test_loader.dataset.dataset.tensors[0].mean(dim=0)])
        ig = IntegratedGradients(self.model.model)
        nt = NoiseTunnel(ig)
        dl = DeepLift(self.model.model)

        self.model.eval()
        
        attribution_ig = self.attribute(attribution_loader, baselines=baseline, target=target, attribution_method=ig)
        attribution_nt = self.attribute(attribution_loader, baselines=baseline, target=target, attribution_method=nt)
        attribution_dl = self.attribute(attribution_loader, baselines=baseline, target=target, attribution_method=dl)

        attribution_da = xr.DataArray([attribution_ig,attribution_nt,attribution_dl], 
                                    dims=['attr_method']+list(ds_test.dims), 
                                    coords={'attr_method':['IntegratedGradients', 'IntegratedGradients+NoiseTunnel', 'DeepLift'], 
                                            **ds_test.isel(time=true_indices).coords})
        
        ds_attr = xr.Dataset(dict(data = ds_test.isel(time=true_indices)-ds_test.mean('time'),
                                  attributions = attribution_da)).sortby('time').assign_coords(target=target)
        return ds_attr
    
    def get_multi_attribution_ds_regr(self, test_loader, ds_test, min_rain=30):
        truth = torch.stack(self.model.test_step_true_values).cpu().numpy().squeeze()
        pred = torch.stack(self.model.test_step_pred).cpu().numpy().squeeze()
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


### Quantiles 

class ConfusionMatrix(Callback):
    def on_test_end(self, trainer, pl_module):
        truth = torch.stack(pl_module.test_step_true_labels).cpu().numpy().squeeze()
        pred = torch.stack(pl_module.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=truth, preds=pred,
                            class_names=[f"{10*k:.0f}-{10*(k+1):.0f}" for k in range(10)])})
        

class PrecisionRecallClass9(Callback):
    def on_test_end(self, trainer, pl_module):
        truth = torch.stack(pl_module.test_step_true_labels).cpu().numpy().squeeze()
        pred = torch.stack(pl_module.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
        recall_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((truth==9))[0].size
        precision_class9 = np.where((truth==9)&(pred==9))[0].size/np.where((pred==9))[0].size
        wandb.log({"Scores/precision9" : precision_class9*100,
                "Scores/recall9" : recall_class9*100,
                "Scores/f1_9" : 2*(precision_class9*recall_class9)/(precision_class9+recall_class9),
                })

class BetterConfusionMatrix(Callback):
    def on_test_end(self, trainer, pl_module):
        truth = torch.stack(pl_module.test_step_true_labels).cpu().numpy().squeeze()
        pred = torch.stack(pl_module.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
        
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
    

### Regression
class LogPlotsRegression(Callback):
    def on_test_end(self, trainer, pl_module):
        truth = torch.stack(pl_module.test_step_true_values).cpu().numpy().squeeze()
        pred = torch.stack(pl_module.test_step_pred).cpu().numpy().squeeze()#.argmax(axis=1)
        df = pd.DataFrame(dict(truth=truth, pred=pred))
        table = wandb.Table(dataframe=df)
        sorted_table = wandb.Table(dataframe=df.transform(np.sort))

        histogram_rain = wandb.plot.histogram(table, "pred",title="Predicted Rain Distribution")

        r_value = float(df.corr().truth.pred)
        rmse = float(np.sqrt(((df.truth - df.pred)**2).mean()))
        #
        wandb.log({'histogram_precip_predicted': histogram_rain,
                   'sorted_predictions':sorted_table,
                   'test/r_value' : r_value,
                   'test/rmse' : rmse
                   })
        



