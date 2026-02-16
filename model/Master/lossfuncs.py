import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if not hasattr(RegressionLoss, loss_type):
            raise ValueError(f"Unsupported reg_loss: {loss_type}")
        self.func = getattr(RegressionLoss, loss_type)

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    @staticmethod
    def smoothl1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.func(pred, target)


class EntropyLoss(nn.Module):
    def __init__(self, from_logits: bool = True, reduction: str = "mean") -> None:
        super().__init__()
        self.from_logits = from_logits
        self.reduction = reduction

    @staticmethod
    def entropy(probs: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        epsilon = 1e-5
        entropy = -probs * torch.log(probs + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return EntropyLoss._reduce(entropy, reduction)

    @staticmethod
    def entropy_logits(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return EntropyLoss.entropy(probs, reduction=reduction)

    @staticmethod
    def _reduce(entropy: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction in (None, "none", "None"):
            return entropy
        if reduction == "mean":
            return entropy.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            return self.entropy_logits(inputs, reduction=self.reduction)
        return self.entropy(inputs, reduction=self.reduction)
