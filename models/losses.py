import torch 
from torch import nn
from torch.nn import functional

"""
https://github.com/Javicadserres/wind-production-forecast/blob/28310d7dab7b47d7db3d690580505c1a456e471b/src/model/losses.py#L5
"""

class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        
    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        upper =  self.quantiles * error
        lower = (self.quantiles - 1) * error 

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss

class SmoothPinballLoss(nn.Module):
    """
    Smoth version of the pinball loss function.

    Parameters
    ----------
    quantiles : torch.tensor
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles, alpha=0.001):
        super(SmoothPinballLoss,self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        q_error = self.quantiles * error
        beta = 1 / self.alpha
        soft_error = functional.softplus(-error, beta)

        losses = q_error + soft_error
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss



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
        errors_cdf = torch.abs(targets_sorted - predictions_sorted)**2
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