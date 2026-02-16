import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.autograd import Function


# Adver_network.py
# 梯度反转层
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# Adver_network.py
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# common_network.py
class FeatBottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, layer_type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.layer_type = layer_type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.layer_type == "bn":
            x = self.bn(x)
        return x


# common_network.py
class FeatClassifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, layer_type="linear"):
        super().__init__()
        if layer_type == "wn":
            self.fc = weight_norm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        return self.fc(x)