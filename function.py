# ! /usr/bin/python
# -*- coding: utf-8 -*-
#
# File: funtions used in algorithms
# Created: Apr.22, 2020
# Last modified: Apr.22, 2020

__author__ = "Li Li"
__mail__ = "lili@mmlab.cs.tsukuba.ac.jp"
__license__ = __author__


import math
import os
import torch
import numpy as np
from scipy.io import wavfile
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f2omega(f):
    return 2 * math.pi * f


def ang2rad(x, coef=False):
    if coef:
        return x / 360 * 2
    else:
        return x / 360 * 2 * math.pi


def generate_sv(d, theta, omega):
    c = 343.
    if type(d) is list:
        d = np.asarray(d)
    if type(omega) is list:
        omega = np.asarray(omega)
    if type(theta) is list:
        theta = np.asarray(theta)
    elif type(theta) in [int, float]:
        theta = np.asarray([theta])
    if d.ndim == 1:
        d = d[None, :, None]
    if omega.ndim == 1:
        omega = omega[..., None, None]
    if theta.ndim == 1:
        theta = theta[None, None, :]

    return np.exp(-1.j * omega * d * np.cos(ang2rad(theta)) / c)

def generate_sv2(d, theta, omega):
    c = 343.
    if type(d) is list:
        d = np.asarray(d)
    if type(omega) is list:
        omega = np.asarray(omega)
    if d.ndim == 1:
        d = d[None, :, None]
    if omega.ndim == 1:
        omega = omega[..., None, None]
    if theta.ndim == 1:
        theta = theta[None, None, :]

    omega = torch.from_numpy(omega)
    dis = torch.from_numpy(d)

    return torch.exp(-1.j*omega*dis*torch.cos(ang2rad(theta))/c)


def load_multichannel(path):
    mix_path = os.listdir(path)
    n_ch = len(mix_path)
    fs, x_ = wavfile.read(path + mix_path[0])
    x = np.zeros((len(x_), n_ch))
    x[:, 0] = x_

    for i in range(1, n_ch):
        fs, x_ = wavfile.read(path + mix_path[i])
        x[:, i] = x_

    return fs, x


def directivity(W, f, d, dB=True):
    theta = ang2rad(np.linspace(0, 180, 181))[..., None, None, None]
    c = 343.
    f = f[None, :, None, None]
    if type(d) is list:
        d = np.asarray(d)[None, None, None]
    phase = np.exp(1j * 2 * math.pi * f * d * np.cos(theta) / c)
    pattern = np.sum(np.conj(W[None]) * phase, axis=3)

    if dB:
        pattern = 10 * np.log10(np.maximum(np.abs(pattern), 1.e-5))

    return pattern, np.squeeze(f, axis=(2, 3)), np.squeeze(theta, axis=(2, 3))


def plot_signal(sig, path=None):
    plt.clf()
    plt.plot(sig)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

    return None


def plot_directivity(pattern, vmin=-20, vmax=20, path=None):
    if path is None:
        plt.pcolormesh(pattern, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()
    else:
        plt.clf()
        plt.pcolormesh(pattern, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.savefig(path)
        plt.close()

    return None


def plot_directivity3D(pattern, f, path=None):
    theta = np.linspace(0, 180, 181)[..., None]
    X, Y = np.meshgrid(theta, f)
    Z = pattern.T

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_zlim(-30, 30)
    fig.colorbar(surf)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

    return None
