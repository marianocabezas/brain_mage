import time
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel, ResConv3dBlock, DoubleResConv3dBlock
from base import Encoder, Random3DTransformer



def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


class FeatureNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            verbose=0,
    ):
        super(FeatureNet, self).__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.encoder = Encoder(self.conv_filters, self.device, n_images, block=DoubleResConv3dBlock)
        self.encoder.to(device)

        self.transformer = Random3DTransformer()
        self.transformer.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'triplet',
                'weight': 1,
                'f': lambda p, t: self._triplet(p[0], p[1])
            }
        ]

        self.val_functions = self.train_functions

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        features_a = self.encoder(data)

        data_b = self.transformer(data)
        features_b = self.encoder(data_b)

        return features_a, features_b

    def _contrastive(self, features_a, features_b):
        y_pos = torch.ones(len(features_a), dtype=torch.uint8).to(self.device)

        return F.cosine_embedding_loss(features_a, features_b, target=y_pos)

    def _triplet(self, features_a, features_b):
        loss = 0
        for i in range(len(features_a) - 1):
            loss += F.triplet_margin_loss(
                features_a[i].unsqueeze(0),
                features_b[i].unsqueeze(0),
                features_a[i + 1].unsqueeze(0)
            )
        loss += F.triplet_margin_loss(
                features_a[-1].unsqueeze(0),
                features_b[-1].unsqueeze(0),
                features_a[0].unsqueeze(0)
            )

        return loss


class ClassifierNet(FeatureNet):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            n_classes=2,
            verbose=0,
    ):
        super(ClassifierNet, self).__init__(conv_filters, device, n_images, verbose)
        self.heads = nn.Linear(self.conv_filters[-1], n_classes)
        self.heads.to(device)

    def forward(self, data):
        features = self.encoder(data)
        logits = self.heads(features)

        return logits