import lightning as L
from lightning.pytorch.callbacks import Callback
from models.losses import DistribLoss

from torch import Tensor, nn, optim, stack, long
import wandb

class LitCNN(L.LightningModule):
    def __init__(self, model, 
                 type_prediction='quantiles',
                 learning_rate=1e-3,
                 loss=DistribLoss()):
        super().__init__()
        self.model = model
        self.test_step_pred = []
        self.test_step_y = []
        self.learning_rate = learning_rate

        self.loss_fn = loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.test_step_y += y
        self.test_step_pred += pred

        self.log("test/loss", loss)
        return x, pred
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("val/loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    def forward(self, batch):
        x, y = batch
        # x = x.view(x.size(0), -1)
        pred = self.model(x)
        return pred

class MyCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        pl_module.test_step_y =  Tensor(pl_module.test_step_y).type(long)
        pl_module.test_step_pred =  stack(pl_module.test_step_pred)
        # print(pl_module.test_step_pred)
        # do something with all training_step outputs, for example:
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                y_true=pl_module.test_step_y.numpy(), 
                                preds=pl_module.test_step_pred.numpy(),
                                class_names=[k for k in range(10)]
                                )})