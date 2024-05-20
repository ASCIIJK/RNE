import torch
from torch import nn
from torch.nn import functional as F


class CRL(nn.Module):
    def __init__(self, input_channels, img_size):
        super(CRL, self).__init__()
        if img_size > 8:
            self.conv_a = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_b = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.conv_a = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=False)
            self.conv_b = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ln_a = nn.LayerNorm(img_size)
        self.ln_b = nn.LayerNorm(img_size)

    def forward(self, x):
        # x = self.conv_a(x)
        # # x = self.relu(x)
        # x = self.ln(x)
        # # x = self.relu(x)

        x1 = self.conv_a(x)
        x1 = self.ln_a(x1)
        x1 = F.relu(x1, inplace=True)

        x2 = self.conv_b(x1)
        x2 = self.ln_b(x2)
        x2 = F.relu(x2, inplace=True)

        out = x + x2
        return out


class CRLs(nn.Module):
    def __init__(self, convtype):
        super(CRLs, self).__init__()
        self.CRL = nn.ModuleList()
        if convtype == "cifar_resnet18":
            self.CRL.append(CRL(input_channels=64, img_size=32))
            self.CRL.append(CRL(input_channels=64, img_size=32))
            self.CRL.append(CRL(input_channels=128, img_size=16))
            self.CRL.append(CRL(input_channels=256, img_size=8))
            self.CRL.append(CRL(input_channels=512, img_size=4))
        elif convtype == "cifar_resnet18_compress":
            self.CRL.append(CRL(input_channels=16, img_size=32))
            self.CRL.append(CRL(input_channels=32, img_size=16))
            self.CRL.append(CRL(input_channels=64, img_size=8))
            self.CRL.append(CRL(input_channels=128, img_size=4))
        else:
            raise "No addaptive CRL modules!"

    def forward(self, fmaps):
        for i in range(len(fmaps)):
            fmaps[i] = self.CRL[i](fmaps[i])
        return fmaps
