import torch
import torch.nn.functional as F
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Softmax()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.squeeze(x).view(N, C)
        y = self.excitation(y).view(N, C, 1, 1)
        return x * y.expand_as(x) + x


class TemporalGatedUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, stride):
        super().__init__()

        self.conv_0 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding)
        # torch.manual_seed(42)
        self.w_0 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        # print("log w_0", self.w_0)

        self.conv_gate_0 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding)
        self.v_0 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        # print("log v_0", self.v_0)

    def forward(self, x):
        A = self.conv_0(x)
        N, C, T, V = A.size()
        # print("size of tensor ", A.size())
        A += self.w_0.repeat(N, 1, T, V)
        B = self.conv_gate_0(x)
        B += self.v_0.repeat(N, 1, T, V)

        return A * torch.sigmoid(B)


class AttentionBlock(nn.Module):
    def __init__(self, alpha, in_channels=300):
        super().__init__()
        self.alpha = alpha
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels ** 2)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        avg_pooling = self.pooling(x).view(N, T)

        fcn = self.fcn(avg_pooling)
        x1 = self.softmax(fcn).view(N, T, T)

        x2 = x.permute(0,2,1,3).contiguous().view(N, T, C*V)
        x3 = torch.matmul(x1, x2).view(N, T, C, V)

        return (x + x3 * self.alpha).permute(0, 2, 1, 3).contiguous()


