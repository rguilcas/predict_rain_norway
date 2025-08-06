from lightning import Callback
import torch 
import pandas as pd
import wandb
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
from lightning.pytorch.callbacks import ModelCheckpoint

def get_checkpoint_callback(wandb_logger):
    run_id = wandb_logger.experiment.id 
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",         # 🔍 monitor your validation metric
        mode="min",                 # minimize val_loss
        save_top_k=1,               # only keep the best model
        save_weights_only=True,   # save full model (architecture + weights + config)
        dirpath="/Data/gfi/users/rogui7909/data/NN_outputs/checkpoints/",     # local directory to save
        filename=f"{run_id}-best-{{epoch:02d}}-{{val_loss:.2f}}"
        )
    return checkpoint_cb


class LogF1ValidationBoolean(Callback):
    def __init__(self, num_timesteps=4, target_value=1):
        self.num_timesteps=num_timesteps
        self.target_value = target_value
    def on_validation_start(self, trainer, model):
        self.predictions_validation = []
        self.targets_validation = []
    def on_validation_end(self, trainer, model):
        y_validation = torch.stack(model.targets_validation)#.cpu().numpy()
        y_scores = torch.stack(model.predictions_validation).view(-1, self.num_timesteps, self.number_of_classes).cpu()
        preds = torch.argmax(y_scores, dim=2) 
        targets = y_validation.cpu().numpy()
        model.predictions_validation = preds 
        model.targets_validation = targets
        TP = ((model.predictions_validation==model.targets_validation) &(model.targets_validation==2)).sum(axis=0)
        recall = TP / (model.targets_validation==2).sum(axis=0)
        precision = TP / (model.predictions_validation==2).sum(axis=0)
        f1 = 2*(precision*recall)/(precision+recall)
        model.f1_per_day = f1

        TP = ((model.predictions_validation==model.targets_validation) &(model.targets_validation==2)).sum()
        recall = TP / (model.targets_validation==2).sum()
        precision = TP / (model.predictions_validation==2).sum()
        f1 = 2*(precision*recall)/(precision+recall)

        model.precision_validation = precision
        model.recall_validation = recall
        model.f1_validation = f1
        wandb.log({"val/precision_epoch": model.precision_validation,
                   "val/recall_epoch": model.recall_validation,
                   "val/f1_epoch": model.f1_validation})
        

    def on_test_start(self, trainer, model):
        self.predictions_test = []
        self.targets_test = []

    def on_test_end(self, trainer, model):
        y_test = torch.stack(model.targets_test)#.cpu().numpy()
        y_scores = torch.stack(model.predictions_test).view(-1, self.num_timesteps, 3).cpu()
        preds = torch.argmax(y_scores, dim=2) 
        targets = y_test.cpu().numpy()
        model.predictions_test = preds 
        model.targets_test = targets
        TP = ((model.predictions_test==model.targets_test) &(model.targets_test==2)).sum(axis=0)
        recall = TP / (model.targets_test==2).sum(axis=0)
        precision = TP / (model.predictions_test==2).sum(axis=0)
        f1 = 2*(precision*recall)/(precision+recall)
        model.f1_per_day_test = f1

        # xr_targets=xr.DataArray(y_test.cpu().numpy(), dims=['time','timestep'], coords=dict(time=trainer.ds_test.time[:y_test.shape[0]], timestep=np.arange(y_scores.shape[1])))
        # xr_preds=xr.DataArray(preds.cpu().numpy(), dims=['time','timestep'], coords=dict(time=trainer.ds_test.time[:y_test.shape[0]], timestep=np.arange(y_scores.shape[1])))
        trainer.targets_test  = targets
        trainer.predictions_test = preds
        table_out = wandb.Table(columns=["Day", "F1","Precision","Recall"],
                            data = np.array([[k for k in range(len(f1))], f1, precision, recall]).T)
        wandb.log({"test/results_per_day_plot":table_out})

class LogF1Validation(Callback):
    def __init__(self, num_timesteps=4, target_value=2, number_of_classes=3):
        self.num_timesteps=num_timesteps
        self.target_value = target_value
        self.number_of_classes = number_of_classes
    def on_validation_start(self, trainer, model):
        self.predictions_validation = []
        self.targets_validation = []
    def on_validation_end(self, trainer, model):
        y_validation = torch.stack(model.targets_validation)#.cpu().numpy()
        y_scores = torch.stack(model.predictions_validation).view(-1, self.num_timesteps, self.number_of_classes).cpu()
        preds = torch.argmax(y_scores, dim=2) 
        targets = y_validation.cpu().numpy()
        model.predictions_validation = preds 
        model.targets_validation = targets
        TP = ((model.predictions_validation==model.targets_validation) &(model.targets_validation==2)).sum(axis=0)
        recall = TP / (model.targets_validation==2).sum(axis=0)
        precision = TP / (model.predictions_validation==2).sum(axis=0)
        f1 = 2*(precision*recall)/(precision+recall)
        model.f1_per_day = f1

        TP = ((model.predictions_validation==model.targets_validation) &(model.targets_validation==2)).sum()
        recall = TP / (model.targets_validation==2).sum()
        precision = TP / (model.predictions_validation==2).sum()
        f1 = 2*(precision*recall)/(precision+recall)

        model.precision_validation = precision
        model.recall_validation = recall
        model.f1_validation = f1
        wandb.log({"val/precision_epoch": model.precision_validation,
                   "val/recall_epoch": model.recall_validation,
                   "val/f1_epoch": model.f1_validation})
        

    def on_test_start(self, trainer, model):
        self.predictions_test = []
        self.targets_test = []

    def on_test_end(self, trainer, model):
        y_test = torch.stack(model.targets_test)#.cpu().numpy()
        y_scores = torch.stack(model.predictions_test).view(-1, self.num_timesteps, 3).cpu()
        preds = torch.argmax(y_scores, dim=2) 
        targets = y_test.cpu().numpy()
        model.predictions_test = preds 
        model.targets_test = targets
        TP = ((model.predictions_test==model.targets_test) &(model.targets_test==2)).sum(axis=0)
        recall = TP / (model.targets_test==2).sum(axis=0)
        precision = TP / (model.predictions_test==2).sum(axis=0)
        f1 = 2*(precision*recall)/(precision+recall)
        model.f1_per_day_test = f1

        # xr_targets=xr.DataArray(y_test.cpu().numpy(), dims=['time','timestep'], coords=dict(time=trainer.ds_test.time[:y_test.shape[0]], timestep=np.arange(y_scores.shape[1])))
        # xr_preds=xr.DataArray(preds.cpu().numpy(), dims=['time','timestep'], coords=dict(time=trainer.ds_test.time[:y_test.shape[0]], timestep=np.arange(y_scores.shape[1])))
        trainer.targets_test  = targets
        trainer.predictions_test = preds
        table_out = wandb.Table(columns=["Day", "F1","Precision","Recall"],
                            data = np.array([[k for k in range(len(f1))], f1, precision, recall]).T)
        wandb.log({"test/results_per_day_plot":table_out})
        
class LogIndividualScoresThreeClasses(Callback):
    def on_test_end(self, trainer, model):
        model.test_true_values = torch.stack(model.test_true_values).cpu().numpy()
        model.test_pred = torch.stack(model.test_pred).cpu().numpy()
        rmse = np.sqrt(((model.test_pred-model.test_true_values)**2).mean(axis=0))
        corr = pearsonr(model.test_true_values, model.test_pred).statistic
        quantiles95_true = np.quantile(model.test_true_values, q=0.95, axis=0)
        predicted_over_q = (model.test_pred>quantiles95_true).sum(axis=0)
        true_over_q = (model.test_true_values>quantiles95_true).sum(axis=0)
        true_positives = ((model.test_pred>quantiles95_true)&(model.test_true_values>quantiles95_true)).sum(axis=0)
        precision = true_positives/predicted_over_q
        recall = true_positives/true_over_q 
        f1 = 2*precision*recall/(precision+recall)
        f1[np.isnan(f1)] = 0
        table_out = wandb.Table(columns=["Day","RMSE","Correlation", "F1_over95","Precision_over95","Recall_over95"],
                            data = np.array([[k for k in range(len(rmse))],rmse, corr, f1, precision, recall]).T)


        wandb.log({"test/results_per_day_plot":table_out})
        for k in range(len(corr)):
            wandb.log({f"test/f1_95_day{k}":f1[k],
                      f"test/rmse_day{k}":rmse[k],
                      f"test/corr_day{k}":corr[k],})
            
        