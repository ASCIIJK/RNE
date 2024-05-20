import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from incrementNet import get_convnet
from conv.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CenterLinear
from conv.Moudule_tool import CRL, CRLs
import timm


class RNEbaseNet(nn.Module):
    def __init__(self, args):
        super(RNEbaseNet, self).__init__()
        self.convnets = nn.ModuleList()
        self.CRLs = CRLs(convtype=args["convnet_type"])
        self.fc_list = nn.ModuleList()
        self.backbone = args["convnet_type"]
        self.out_dim = []
        self.classes = []
        self.known_classes = 0
        self.cur_task = -1
        self.aux_fc = None

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes):
        self.cur_task += 1
        if self.cur_task == 0:
            convnet = get_convnet(self.backbone)
        else:
            convnet = copy.deepcopy(self.convnets[-1])
        self.convnets.append(convnet)
        self.out_dim.append(convnet.out_dim)

        fc = self.generate_fc(self.feature_dim(), nb_classes - self.known_classes)

        self.fc_list.append(fc)
        del self.aux_fc
        aux_fc = self.generate_fc(self.convnets[-1].out_dim, nb_classes - self.known_classes + 1)
        self.aux_fc = aux_fc
        self.known_classes = nb_classes
        self.classes.append(nb_classes)

    def feature_dim(self):
        return sum(self.out_dim)

    def extract_vector(self, x):
        if len(self.convnets) == 1:
            with torch.no_grad():
                features = self.convnets[0](x)["features"]
        else:
            features = []
            with torch.no_grad():
                out = self.convnets[0](x)
            features.append(out["features"])
            fmaps = out["fmaps"]
            for i in range(len(self.convnets) - 1):
                with torch.no_grad():
                    fmaps = self.CRLs(fmaps)
                    out = self.convnets[i+1](x, fmaps)
                fmaps = out["fmaps"]
                features.append(out["features"])
            features = torch.cat(features, 1)
        return features

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, mode=None):
        if mode is None:
            features = []
            out = self.convnets[0](x)
            fmaps = out["fmaps"]
            features.append(out["features"])
            for i in range(len(self.convnets) - 1):
                fmaps = self.CRLs(fmaps)
                out = self.convnets[i + 1](x, fmaps)
                fmaps = out["fmaps"]
                features.append(out["features"])
            if len(features) == 1:
                features = features[0]
                logits = self.fc_list[0](features)["logits"]
                out = {"features": features,
                       "logits": logits}
            else:
                logits = []
                aux_logits = self.aux_fc(features[-1])["logits"]
                features = torch.cat(features, dim=1)
                for i in range(len(self.fc_list)):
                    logits.append(self.fc_list[i](features[:, :sum(self.out_dim[:(i+1)])])["logits"])
                logits = torch.cat(logits, dim=1)
                out = {
                    "features": features,
                    "logits": logits,
                    "aux_logits": aux_logits
                }
            return out
        else:
            logits = []
            for i in range(len(self.fc_list)):
                logits.append(self.fc_list[i](x[:, :sum(self.out_dim[:(i+1)])])["logits"])
            logits = torch.cat(logits, dim=1)
        return {
            "logits": logits
        }


class RNECompressBaseNet(nn.Module):
    def __init__(self, args):
        super(RNECompressBaseNet, self).__init__()
        self.convnets = nn.ModuleList()
        self.CRLs = CRLs(convtype=args["convnet_type"])
        self.fc_list = nn.ModuleList()
        self.backbone_name = args["convnet_type"]
        self.backbone = get_convnet(self.backbone_name)
        self.out_dim = []
        self.classes = []
        self.known_classes = 0
        self.cur_task = -1
        self.aux_fc = None

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes):
        self.cur_task += 1
        if self.cur_task == 0:
            if self.backbone_name == "cifar_resnet18_compress":
                convnet = get_convnet('cifar_resnet18_inc')
            else:
                raise "{} do noy exist!".format(self.backbone_name)
        else:
            convnet = copy.deepcopy(self.convnets[-1])
        self.convnets.append(convnet)
        self.out_dim.append(convnet.out_dim)

        fc = self.generate_fc(self.feature_dim(), nb_classes - self.known_classes)
        self.fc_list.append(fc)

        del self.aux_fc
        aux_fc = self.generate_fc(self.convnets[-1].out_dim, nb_classes - self.known_classes + 1)
        self.aux_fc = aux_fc

        self.known_classes = nb_classes
        self.classes.append(nb_classes)

    def feature_dim(self):
        return sum(self.out_dim)

    def extract_vector(self, x):
        features = []
        out = self.backbone(x)
        fmaps_backbone = out["fmaps"]
        for i in range(len(self.convnets)):
            if i == 0:
                out = self.convnets[i](x, fmaps_backbone)
            else:
                fmaps = self.CRLs(fmaps)
                out = self.convnets[i](x, fmaps)
            fmaps = out["fmaps"]
            features.append(out["features"])
        features = torch.cat(features, 1)
        return features

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, mode=None):
        if mode is None:
            features = []
            out = self.backbone(x)
            fmaps_backbone = out["fmaps"]
            for i in range(len(self.convnets)):
                if i == 0:
                    out = self.convnets[i](x, fmaps_backbone)
                else:
                    fmaps = self.CRLs(fmaps)
                    out = self.convnets[i](x, fmaps)
                fmaps = out["fmaps"]
                features.append(out["features"])
            logits = []
            if self.cur_task > 0:
                aux_logits = self.aux_fc(features[-1])["logits"]
            else:
                aux_logits = None
            features = torch.cat(features, dim=1)
            for i in range(len(self.fc_list)):
                logits.append(self.fc_list[i](features[:, :sum(self.out_dim[:(i + 1)])])["logits"])
            logits = torch.cat(logits, dim=1)
            out = {
                "features": features,
                "logits": logits,
                "aux_logits": aux_logits,
            }
            return out
        else:
            logits = []
            for i in range(len(self.fc_list)):
                logits.append(self.fc_list[i](x[:, :sum(self.out_dim[:(i + 1)])])["logits"])
            logits = torch.cat(logits, dim=1)
        return {
            "logits": logits
        }