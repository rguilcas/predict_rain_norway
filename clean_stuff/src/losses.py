import torch 
from torch import nn
from torch.nn import functional

"""
https://github.com/Javicadserres/wind-production-forecast/blob/28310d7dab7b47d7db3d690580505c1a456e471b/src/model/losses.py#L5
"""

def get_loss(loss_key):
    if loss_key == 'distrib':
        loss = SortedMSELoss()
    elif loss_key == 'mse':
        loss = torch.nn.MSELoss()
    elif loss_key == 'quantiles90':
        loss = PinballLoss(0.9)
    elif loss_key == 'quantiles70':
        loss = PinballLoss(0.7)
    elif loss_key == 'quantiles80':
        loss = PinballLoss(0.8)
    elif loss_key == 'quantiles75':
        loss = PinballLoss(0.75)
    elif loss_key == 'composite_quantiles':
        loss = CompositeQuantileLoss(quantiles = [0.5,0.75,0.9], lambdas = None)
    elif loss_key == 'hybrid_mse_quantiles':
        loss = HybridMSEQuantileLoss()
    elif loss_key == 'weighted_mse':
        loss = WeightedMSELoss()
    elif loss_key == 'weighted_sqrt_mse':
        loss = WeightedSqrtMSELoss()
    elif loss_key == 'quantile_extr':
        loss = CompositeQuantileLoss(quantiles=[0.9,0.95,0.99])
    elif loss_key == 'log_mse':
        loss = LogMSE()
    elif loss_key == 'weighted_log_mse':
        loss = WeightedLogMSELoss()
    elif loss_key == 'asymetric_mse':
        loss = AsymmetricMSELoss(alpha=2)
    elif loss_key == 'asymetric_mse_thresh':
        loss = AsymmetricMSEabveThreshLoss(alpha=4, thresh=20)
    elif loss_key == 'f1_mse':
        loss = F1MSELoss(threshold_mm=20)
    return loss

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, timesteps):
        super(MultiCrossEntropyLoss, self).__init__()
        self.nll = torch.nn.NLLLoss(weight = torch.Tensor([1,1,7]))
        self.timesteps = timesteps
        
    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        loss = []
        pred = pred.view(-1, self.timesteps, 3)
        proba = torch.softmax(pred, dim=2)
        for k,timestep in enumerate(range(self.timesteps)):
            res = self.nll(torch.log(proba[:,timestep]), target[:,timestep])
            loss.append(res)#*(k+1)/10)
        loss = torch.mean(torch.stack(loss))
        return loss
    

class PinballLoss(nn.Module):
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
    

class PinballLossSquare(nn.Module):
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
        super(PinballLossSquare, self).__init__()
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

        losses = torch.max(lower, upper)**2
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
        loss = (torch.mean(errors_cdf) + torch.mean(errors))/2
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
    
class SortedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        sorted_pred, _ = torch.sort(pred, dim=0)
        sorted_target, _ = torch.sort(target, dim=0)
        return (self.mse(sorted_pred, sorted_target) + self.mse(pred, target))/2

class F1MSELoss(nn.Module):
    def __init__(self, threshold_mm=20):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        TP = ((target>20)&(pred>20)).sum(axis=0)
        P_pred = ((pred>20)).sum(axis=0)
        P_target = ((target>20)).sum(axis=0)
        recall = (TP/P_target)
        precision = (TP/P_pred)
        f1 = 1-recall*precision*2/(precision+recall)
        f1[(P_pred==0)&(P_target>0)] = 1
        f1[(P_target==0)&(P_pred>0)] = 1
        f1[(P_pred==0)&(P_target==0)] = 0
        f1[(precision==0)&(recall==0)] = 1
        f1 = torch.mean(f1)*100

        return (f1) + self.mse(pred, target)


class LogMSE(nn.Module):
    def __init__(self, offset=.01):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target):
        return torch.mean((torch.log(pred+self.offset) - torch.log(target+self.offset)) ** 2)



class WeightedLogMSELoss(nn.Module):
    def __init__(self, offset=.1):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target):
        weights = target + self.offset
        return torch.mean(weights*((torch.log(pred+self.offset) - torch.log(target+self.offset)) ** 2))
        
class WeightedMSELoss(nn.Module):
    def __init__(self, offset=.1):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target):
        weights = target + self.offset
        return torch.mean((weights*((pred - target)) ** 2))

class WeightedSqrtMSELoss(nn.Module):
    def __init__(self, offset=.1):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target):
        weights = torch.sqrt(target + self.offset)
        return torch.mean((weights*((pred - target)) ** 2))

class AsymmetricMSELoss(torch.nn.Module):
    def __init__(self, alpha=2.0):  # alpha > 1 increases penalty on underestimation
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.where(errors > 0, self.alpha * errors**2, errors**2)
        return loss.mean()

class AsymmetricMSEabveThreshLoss(torch.nn.Module):
    def __init__(self, alpha=2.0,thresh=20):  # alpha > 1 increases penalty on underestimation
        super().__init__()
        self.alpha = alpha
        self.threshold = thresh

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        is_extreme = y_true > self.threshold  # Mask for extreme values
        loss = torch.where((errors > 0) & is_extreme, self.alpha * errors**2, errors**2)
        return loss.mean()
    
class WeightedOrdinalLoss(nn.Module):
    def __init__(self, num_classes, extreme_weight=5.0):
        super(WeightedOrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.extreme_weight = extreme_weight  # Weight for extreme class

    def forward(self, logits, y_true):
        # Convert y_true to cumulative format
        cum_probs = torch.sigmoid(logits)
        y_cum = torch.zeros_like(cum_probs)
        for i in range(self.num_classes - 1):
            y_cum[:, i] = (y_true > i-1).float()

        # Weighted BCE loss
        weights = torch.ones_like(y_cum)
        weights[:, -1] = self.extreme_weight  # Higher weight for last threshold (extreme rainfall)
        weights[:, -2] = 1+(self.extreme_weight-1)*2/3  # Higher weight for last threshold (extreme rainfall)
        weights[:, -3] = 1+(self.extreme_weight-1)*1/3  # Higher weight for last threshold (extreme rainfall)

        loss = nn.functional.binary_cross_entropy(cum_probs, y_cum, weight=weights)
        return loss
    
def logits_to_class_probs(logits):
    cum_probs = torch.sigmoid(logits)
    batch_size = cum_probs.shape[0]
    num_classes = cum_probs.shape[1] + 1
    
    # Convert cumulative probabilities to class probabilities
    class_probs = torch.zeros((batch_size, num_classes), device=cum_probs.device)
    class_probs[:, 0] = 1-cum_probs[:, 0]
    for i in range(1, num_classes - 1):
        class_probs[:, i] = cum_probs[:, i-1] - cum_probs[:, i]
    class_probs[:, -1] = cum_probs[:, -1]  # Last class probability
    return class_probs

class CompositeQuantileLoss(nn.Module):
    def __init__(self, lambdas=None, quantiles=None):
        """
        Composite quantile loss function.
        
        Args:
            lambdas (list): Weights for each quantile loss
            quantiles (list): List of quantiles to use
        """
        super().__init__()
        
        if quantiles is None:
            self.quantiles = [0.5, 0.75, 0.9]  # Default quantiles
        else:
            self.quantiles = quantiles
        
        if lambdas is None:
            self.lambdas = [1 / len(self.quantiles)] * len(self.quantiles)  # Equal weights by default
        else:
            self.lambdas = lambdas
        
        assert len(self.lambdas) == len(self.quantiles), "Lambdas and quantiles must have the same length."
    
    def forward(self, y_pred, y_true):
        loss = sum(
            self.lambdas[i] * PinballLoss(self.quantiles[i])(y_pred, y_true)
            for i in range(len(self.quantiles))
        )
        return loss

class HybridMSEQuantileLoss(nn.Module):
    def __init__(self, lambdas=None, quantile=0.9):
        """
        Hybrid loss combining MSE and quantile loss.
        
        Args:
            lambdas (list): Weights for MSE and quantile loss
            quantile (float): Quantile to use for loss computation
        """
        super().__init__()
        
        if lambdas is None:
            self.lambdas = [0.5, 0.5]  # Default equal weight for MSE and quantile loss
        else:
            self.lambdas = lambdas
        
        self.quantile = quantile
    
    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss()(y_pred, y_true)
        quantile_component = PinballLossSquare(self.quantile)(y_pred, y_true)
        loss = self.lambdas[0] * mse_loss + self.lambdas[1] * quantile_component
        return loss