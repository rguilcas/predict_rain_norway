import torch 
from torch import nn
from torch.nn import functional 
import torch.nn.functional as F

"""
https://github.com/Javicadserres/wind-production-forecast/blob/28310d7dab7b47d7db3d690580505c1a456e471b/src/model/losses.py#L5
"""

def get_loss(config, **kwargs):
    loss_key = config['loss_function']
    match loss_key:
        case 'crossentropy':
            timesteps = config['num_timesteps_predicted']
            loss = MultiCrossEntropyLoss(timesteps)
        case 'focal':
            timesteps = config['num_timesteps_predicted']
            loss = MultiFocalLoss(timesteps=timesteps)
        case 'FocalBCE':
            gamma = config['focal_gamma']
            if 'loss_weights' in config.keys():
                pos_weight =  torch.tensor(config['loss_weights'])
            else:
                pos_weight =  torch.tensor([12., 12., 14., 16.])
            loss = BCEFocalLoss(gamma=gamma, pos_weight=pos_weight)
        case 'BCEwithlogits':
            if 'loss_weights' in config.keys():
                pos_weight =  torch.tensor(config['loss_weights'])  # Boost positive class loss
            else:
                pos_weight =  torch.tensor([12., 12., 14., 16.])  # Boost positive class loss
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        case 'BCEwithlogits_per_horizon':
            if 'loss_weights' in config.keys():
                pos_weight =  torch.tensor(config['loss_weights'])  # Boost positive class loss
            else:
                pos_weight =  torch.tensor([12., 12., 14., 16.])  # Boost positive class loss
            # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = BCEWithLogitsHorizonWeighted(pos_weight=pos_weight, horizon_weight=[0.5,0.7,1.,1.3, 1.5, 2., 2.])
    return loss

class FocalBCEWithLogitsLossDynamicHorizonWeights(nn.Module):
    # Optional: drop-in replacement; set gamma=0 for plain BCE
    def __init__(self, pos_weight=None, gamma=0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits, targets: (B,)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        if self.gamma == 0:
            return bce.mean()
        # p_t = sigmoid(logits) when y=1 else 1-sigmoid(logits)
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal = (1 - p_t).pow(self.gamma) * bce
        return focal.mean()
    
class BCEWithLogitsHorizonWeighted(nn.Module):
    def __init__(self, pos_rate=None, pos_weight=None, horizon_weight=None, eps=1e-8):
        """
        Args:
            pos_rate (Tensor or list, optional): [H] base rates per horizon (0-1).
                                                 Used to compute pos_weight = (1-p)/p.
            pos_weight (Tensor or list, optional): [H] direct positive-class weights.
                                                    Ignored if pos_rate is given.
            horizon_weight (Tensor or list, optional): [H] extra weight per horizon.
            eps (float): numerical stability constant.
        """
        super().__init__()
        self.eps = eps

        if pos_rate is not None:
            pos_rate = torch.as_tensor(pos_rate, dtype=torch.float32)
            pw = (1.0 - pos_rate).clamp(eps, 1.0 - eps) / pos_rate.clamp(eps, 1.0 - eps)
            self.register_buffer("pos_weight", pw)
        elif pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

        if horizon_weight is not None:
            self.register_buffer("horizon_weight", torch.as_tensor(horizon_weight, dtype=torch.float32))
        else:
            self.horizon_weight = None

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: [B, H] raw logits.
            targets: [B, H] binary targets (0 or 1).
            mask: [B, H] binary mask where 1=use, 0=ignore (optional).
        Returns:
            Scalar loss.
        """
        # BCE per element, no reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            reduction='none',
            pos_weight=self.pos_weight
        )  # [B, H]

        # Apply horizon weighting
        if self.horizon_weight is not None:
            bce = bce * self.horizon_weight  # broadcast over batch dim

        # Masking (optional)
        if mask is not None:
            bce = bce * mask
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(logits.numel(), device=logits.device, dtype=logits.dtype)

        # Mean over valid elements
        loss = bce.sum() / denom
        return loss
    
class BCEFocalLoss(nn.Module):
    """
    Binary focal loss for logits.
    gamma > 0 focuses on hard examples
    pos_weight can still be used for class imbalance
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # targets: same shape as logits, 0/1
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
        else:
            pos_weight = None

        bce_loss = functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        loss = ((1 - pt) ** self.gamma) * bce_loss
        return loss.mean()



class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, timesteps):
        super(MultiCrossEntropyLoss, self).__init__()
        self.nll = torch.nn.NLLLoss(weight = torch.Tensor([0.07584753, 0.4, 0.4]))
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
    

class MultiFocalLoss(nn.Module):
    def __init__(self, timesteps, alpha=[.5,1,1], gamma=2):
        super(MultiFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.timesteps = timesteps

    def forward(self, pred, target):
        """
        Computes the focal loss over multiple timesteps.
        """
        device = pred.device
        self.alpha = self.alpha.to(device)
        pred = pred.view(-1, self.timesteps, 3)
        proba = torch.softmax(pred, dim=2).clamp(min=1e-8, max=1. - 1e-8)  # prevent log(0)

        loss = []
        for k, timestep in enumerate(range(self.timesteps)):
            pt = proba[:, timestep, :]
            target_t = target[:, timestep]
            log_pt = torch.log(pt)
            
            # Gather the probs for the true classes
            pt_true = pt.gather(1, target_t.unsqueeze(1)).squeeze(1)
            log_pt_true = log_pt.gather(1, target_t.unsqueeze(1)).squeeze(1)
            
            # Get alpha weighting for each sample
            alpha_t = self.alpha.gather(0, target_t)

            focal_term = (1 - pt_true) ** self.gamma
            loss_t = -alpha_t * focal_term * log_pt_true
            loss.append(loss_t.mean())

        return torch.mean(torch.stack(loss))
    
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