import math
import os
from itertools import count, groupby
from operator import itemgetter
from types import SimpleNamespace

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


@torch.no_grad()
def get_digit_samples():
    by_digit = itemgetter(1)

    mnist = MNIST(os.getcwd(), transform=ToTensor())
    mnist = sorted(mnist, key=by_digit)
    mnist = groupby(mnist, key=by_digit)

    samples = []
    for digit, grp in mnist:
        x, y = next(grp)
        samples.append(x.view(-1))

    return torch.stack(samples)


@torch.no_grad()
def sweep_variable_across_samples(vae, samples, i, sweep):
    """Sweeps a single latent variable
    
    Arguments
    ---------
    vae : torch.Module
        A VAE module; must have a decode method
    samples : n-by-z array-like
        Contains n samples of z latent variables
    i : int < z
        The latent variable to sweep
    sweep : array
        The values to use in sweeping z
    """
    # XXX dumb, unvectorized version
    recons = []
    for sample in samples:
        recons.append([])
        for val in sweep:
            sample[i] = val
            # Use just means as image
            img, _ = vae.decode(sample)
            recons[-1].append(img.detach().numpy())
    return np.array(recons)


@torch.no_grad()
def plot_sweep_grid(origs, recons, sweepvals):
    idx = count(1)
    fig = plt.figure(figsize=(15, 13))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(10):
        plt.subplot(10, 11, next(idx))
        plt.imshow(origs[i].reshape(28, 28))
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('Orig')
        for j in range(10):
            plt.subplot(10, 11, next(idx))
            plt.imshow(recons[i][j].reshape(28, 28))
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.title(f'{sweepvals[j]:.2f}')
    plt.show()
    
    
@torch.no_grad()
def plot_all_sweeps(model):
    digits = get_digit_samples()
    digit_encodings, *_ = model(digits)
    sweep_range = torch.linspace(-4, 4, steps=10)
    
    return digit_encodings, sweep_range
#     for i in range(20):
    for i in range(1):
        print(f'Sweeping reconstructions over latent variable no. {i}')
        recons_by_var = sweep_variable_across_samples(model,
                                                      digit_encodings.clone(),
                                                      i,
                                                      sweep_range)
        plot_sweep_grid(digits.detach().numpy(), recons_by_var, sweep_range)
    return digit_encodings, sweep_range


@torch.no_grad()
def zeroth_mu_sigma(enc, model):
    m, s = model.decode(enc)
    s = F.softplus(s)
    
    m0, s0 = m[0], s[0]
    
    plt.subplot(221)
    plt.imshow(m0.reshape(28, 28), norm=None, cmap='gray', vmin=0.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(222)
    plt.imshow(s0.reshape(28, 28), norm=None, cmap='gray', vmin=0.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(223)
    plt.imshow(m0.reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(224)
    plt.imshow(s0.reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    return m, s