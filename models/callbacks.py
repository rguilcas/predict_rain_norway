from lightning import Callback
import torch 
import pandas as pd
import wandb
import numpy as np
from scipy.stats import pearsonr


class LogIndividualScores(Callback):
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
            
        
### Regression
class LogPlotsRegression(Callback):
    def on_test_end(self, trainer, pl_module):
        pl_module.test_true_values = torch.stack(pl_module.test_true_values).cpu().numpy().squeeze()
        pl_module.test_pred = torch.stack(pl_module.test_pred).cpu().numpy().squeeze()
        truth = pl_module.test_true_values
        pred = pl_module.test_pred
        df = pd.DataFrame(dict(truth=truth, pred=pred))
        table = wandb.Table(dataframe=df)
        sorted_table = wandb.Table(dataframe=df.transform(np.sort))

        histogram_rain = wandb.plot.histogram(table, "pred",title="Predicted Rain Distribution")

        r_value = float(df.corr().truth.pred)
        rmse = float(np.sqrt(((df.truth - df.pred)**2).mean()))
        #

        true_quantiles = df.quantile([0.90,0.95,0.99]).truth
        list_f1 = []
        for quantile in true_quantiles.index:    
            all_pred_overq = df.pred.loc[df.pred>true_quantiles.loc[quantile]].size
            all_truth_overq = df.truth.loc[df.truth>true_quantiles.loc[quantile]].size        
            all_both_overq = df.truth.loc[(df.truth>true_quantiles.loc[quantile])&(df.pred>true_quantiles.loc[quantile])].size
            precision = all_both_overq/all_pred_overq
            recall = all_both_overq/all_truth_overq
            f1 = 2*(precision*recall)/(precision+recall)
            wandb.log({f'recall{quantile}': recall,
                       f'precision{quantile}': precision,
                       f'f1{quantile}': f1,
                   })
        
        
        df_f1 = pd.DataFrame(list_f1, columns = ['quantile','precision','accuracy','f1'])
        
        table_f1 = wandb.Table(dataframe=df_f1)

        wandb.log({'histogram_precip_predicted': histogram_rain,
                   'sorted_predictions':sorted_table,
                   'test/r_value' : r_value,
                   'test/rmse' : rmse})
        