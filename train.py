#!/usr/bin/env python
import math
import os
from itertools import count, groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule, Trainer


class MnistBetaVAE(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.haparams = hparams
        self.z_dim = 20
        self.encoder = nn.Sequential(nn.Linear(784, 400),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(400, 100),
                                     nn.ReLU(),
                                     nn.Dropout(0.9))
        self.z_loc = nn.Linear(100, 20)
        self.z_scale = nn.Linear(100, 20)
        self.decoder = nn.Sequential(nn.Linear(20, 100),
                                     nn.ReLU(),
                                     nn.Dropout(0.9),
                                     nn.Linear(100, 400),
                                     nn.ReLU(),
                                     nn.Dropout(0.9))
        self.x_loc = nn.Linear(400, 784)
        self.x_scale = nn.Linear(400, 784)

    def prepare_data(self):
        # Downloads if necessary
        MNIST(os.getcwd(), train=True, download=True)

    def train_dataloader(self):
        # No download, get mnist from disk
        data = MNIST(os.getcwd(), transform=ToTensor())
        return DataLoader(data, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def encode(self, x):
        h = self.encoder(x)
        return self.z_loc(h), self.z_scale(h)

    def reparameterize(self, mu, scale):
        std = 1e-6 + F.softplus(scale)
        eps = torch.randn_like(std)
        return mu + std*eps

    def decode(self, z):
        h = self.decoder(z)
        return self.x_loc(h), self.x_scale(h)

    def loss(self, x, x_mu, x_scale, z_mu, z_scale):
        x_std = 1e-6 + F.softplus(x_scale)
        z_std = 1e-6 + F.softplus(z_scale)

        norm = -0.5 * math.log(2.0*np.pi) - torch.log(1e-8 + x_std)
        logp = (x - x_mu).pow(2).mul(-1) / x_std.pow(2).mul(2.0) + norm
        logp = logp.sum(1).mean()
        kld = z_mu.pow(2) + z_std.pow(2) - torch.log(1e-8 + z_std.pow(2)) - 1.0
        kld = kld.sum(1).mul(0.5).mean()
        elbo = logp - self.hparams.beta * kld
        return -elbo

    def forward(self, x):
        z_mu, z_scale = self.encode(x)
        z = self.reparameterize(z_mu, z_scale)
        x_mu, x_scale = self.decode(z)
        return z, z_mu, z_scale, x, x_mu, x_scale

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        z, z_mu, z_scale, x, x_mu, x_scale = self.forward(x)
        loss = self.loss(x, x_mu, x_scale, z_mu, z_scale)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--max-epochs', type=int, default=100)

    hparams = parser.parse_args()
    model = MnistBetaVAE(hparams)
    trainer = Trainer(max_epochs=hparams.max_epochs,
                      gradient_clip_val=1.0,
                      benchmark=args.gpu is not None,
                      gpus=args.gpu)
    trainer.fit(model)
