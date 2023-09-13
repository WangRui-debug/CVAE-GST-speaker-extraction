# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CVAE-GST algorithm (PyTorch ver.)

import os
import argparse
import sys
import math
from function import *
import torch
from net_GST import GST
import net_GST
import numpy as np
import numpy.linalg as LA
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from scipy import signal
import math
import utils

epsi = sys.float_info.epsilon
eps = 1.e-10
STFTPara = {'fs': 16000, 'window_size': 2048, 'window_shift': 1024, 'type': "hamming"}

def MVAE(X, AlgPara, NNPara, NNPara1, sv, gpu):
    # check errors and set default values
    I, J, M = X.shape
    N = M
    if N > I:
        sys.stderr.write('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n')

    W = np.zeros((I, M, N), dtype=np.complex)
    for i in range(I):
        W[i, :, :] = np.eye(N)

    # Parameter for ILRMA
    if AlgPara['nb'] is None:
        AlgPara['nb'] = np.ceil(J / 10)
    L = AlgPara['nb']
    T = np.maximum(np.random.rand(I, L, N), epsi)
    V = np.maximum(np.random.rand(L, J, N), epsi)

    R = np.zeros((I, J, N))  # variance matrix
    Y = np.zeros((I, J, N), dtype=np.complex)
    for i in range(0, I):
        Y[i, :, :] = (W[i, :, :] @ X[i, :, :].T).T
    P = np.maximum(np.abs(Y) ** 2, epsi)  # power spectrogram

    # ILRMA
    Y, W, R, P = ilrma(X, W, R, P, T, V, AlgPara['it0'], AlgPara['norm'])

    ####  CVAE ####
    # load trained networks
    n_freq = I - 1
    encoder = net_GST.Encoder(n_freq, NNPara['n_src'])
    encoder_1 = net_GST.Encoder(n_freq, NNPara1['n_src1'])

    decoder = net_GST.Decoder(n_freq, NNPara['n_src'])
    decoder_1 = net_GST.Decoder(n_freq, NNPara1['n_src1'])

    gst = GST()
    #gst = net_GST.GST(NNPara['n_src'])
    gst1 = GST()

    checkpoint = torch.load(NNPara['model_path'], map_location ='cuda:0')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    gst.load_state_dict(checkpoint['gst_state_dict'])


    checkpoint_1 = torch.load(NNPara1['model_path1'], map_location='cuda:0')
    encoder_1.load_state_dict(checkpoint_1['encoder_state_dict'])
    decoder_1.load_state_dict(checkpoint_1['decoder_state_dict'])
    gst1.load_state_dict(checkpoint_1['gst_state_dict'])

    if gpu >= 0:
        device = torch.device("cuda:{}".format(gpu))
        encoder.cuda(device)
        decoder.cuda(device)
        encoder_1.cuda(device)
        decoder_1.cuda(device)
        gst.cuda(device)
        gst1.cuda(device)
    else:
        device = torch.device("cpu")

    Q = np.zeros((N, I, J))  # estimated variance matrix
    P = P.transpose(2, 0, 1)
    R = R.transpose(2, 0, 1)

    phase = np.angle(Y, deg=0)

    # initial z and l
    Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
    gv = np.mean(np.power(Y_abs[:, 1:, :], 2), axis=(1, 2), keepdims=True)
    Y_abs_norm = Y_abs / np.sqrt(gv)
    eps = np.ones(Y_abs_norm.shape) * epsi
    Y_abs_array_norm = np.maximum(Y_abs_norm, eps)[:, None]

    zs, ls, models, optims = [], [], [], []
    zs_1, ls_1, models_1, optims_1 = [], [], [], []

    y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[0, None, :, 1:, :], dtype="float32")).to(device)
    # label = torch.from_numpy(np.ones((1, NNPara['n_src']), dtype="float32") / NNPara['n_src']).to(device)
    label = gst(y_abs)
    label = label.reshape([1, 128])

    plt.clf()
    plt.pcolormesh(y_abs[0, 0].detach().to("cpu").numpy()**.15)
    plt.savefig("./fig_aud_test/TwoGST/yfigure_ch1_TwoGST.png")
    plt.close()

    y_abs_t1 = y_abs.cpu().numpy()
    sep_y_t1 = signal.istft(np.sqrt(y_abs_t1[0, 0]), window=STFTPara['type'])[1]
    sep_y_t1 = sep_y_t1 / max(abs(sep_y_t1)) * 30000
    wavfile.write('./reconstruction/TwoGST/y_t_ch1_TwoGST.wav', STFTPara['fs'], sep_y_t1.astype(np.int16))

    # z is the output of encoder
    #label = gst(y_abs)
    #label = np.squeeze(label)
    z = encoder(y_abs, label)[0]
    zs.append(z)
    ls.append(label)
    # Q is the output of decoder
    Q[0, 1:, :] = np.squeeze(np.exp(decoder(z, label).detach().to("cpu").numpy()), axis=1)

    plt.clf()
    plt.pcolormesh(Q[0]**.3)
    plt.savefig("./fig_aud_test/TwoGST/Qfigure_ch1_TwoGST.png")
    plt.close()

    sep_q_t1 = signal.istft(np.sqrt(Q[0])*phase[:, :, 0], window=STFTPara['type'])[1]
    sep_q_t1 = sep_q_t1 / max(abs(sep_q_t1)) * 30000
    wavfile.write('./reconstruction/TwoGST/Q_t_ch1_TwoGST.wav', STFTPara['fs'], sep_q_t1.astype(np.int16))

    y_abs_1 = torch.from_numpy(np.asarray(Y_abs_array_norm[1, None, :, 1:, :], dtype="float32")).to(device)
    #这里可以尝试给一个正确的初始label
    #label_1 = torch.from_numpy(np.ones((1, NNPara1['n_src1']), dtype="float32") / NNPara1['n_src1']).to(device)
    #label_1 = torch.from_numpy(np.eye(NNPara1['n_src1'], dtype="float32")[0, :]).to(device)
    #label_1 = label_1[None]

    plt.clf()
    plt.pcolormesh(y_abs_1[0, 0].detach().to("cpu").numpy()**.15)
    plt.savefig("./fig_aud_test/TwoGST/yfigure_ch2_TwoGST.png")
    plt.close()

    y_abs_t2 = y_abs_1.cpu().numpy()
    sep_y_t2 = signal.istft(np.sqrt(y_abs_t2[0, 0]), window=STFTPara['type'])[1]
    sep_y_t2 = sep_y_t2 / max(abs(sep_y_t2)) * 30000
    wavfile.write('./reconstruction/TwoGST/y_t_ch2_TwoGST.wav', STFTPara['fs'], sep_y_t2.astype(np.int16))

    label_1 = gst1(y_abs_1)
    #label_1 = np.squeeze(label_1)
    label_1 = label_1.reshape([1, 128])
    z_1 = encoder_1(y_abs_1, label_1)[0]
    zs_1.append(z_1)
    ls_1.append(label_1)
    Q[1, 1:, :] = np.squeeze(np.exp(decoder_1(z_1, label_1).detach().to("cpu").numpy()), axis=1)

    plt.clf()
    plt.pcolormesh(Q[1]**.3)
    plt.savefig("./fig_aud_test/TwoGST/Qfigure_ch2_TwoGST.png")
    plt.close()

    sep_q_t2 = signal.istft(np.sqrt(Q[1])*phase[:, :, 1], window=STFTPara['type'])[1]
    sep_q_t2 = sep_q_t2 / max(abs(sep_q_t2)) * 30000
    wavfile.write('./reconstruction/TwoGST/Q_t_ch2_TwoGST.wav', STFTPara['fs'], sep_q_t2.astype(np.int16))

    Q = np.maximum(Q, epsi)
    gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2), keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat #Update Vj

    # Model construction
    for para in decoder.parameters():
        para.requires_grad = False

    for para1 in decoder_1.parameters():
        para1.requires_grad = False


    z_para = torch.nn.Parameter(zs[0].type(torch.float), requires_grad=True)
    l_para = torch.nn.Parameter(ls[0].type(torch.float), requires_grad=True)

    z_para_1 = torch.nn.Parameter(zs_1[0].type(torch.float), requires_grad=True)
    l_para_1 = torch.nn.Parameter(ls_1[0].type(torch.float), requires_grad=True)

    src_model = net_GST.SourceModel(decoder, z_para, l_para) # load different decoder
    src_model_1 = net_GST.SourceModel(decoder_1, z_para_1, l_para_1)
    if gpu >= 0:
       src_model.cuda(device)
       src_model_1.cuda(device)
    optimizer = torch.optim.Adam(src_model.parameters(), lr=0.01)
    models.append(src_model)
    optims.append(optimizer)

    optimizer_1 = torch.optim.Adam(src_model_1.parameters(), lr=0.01)
    models_1.append(src_model_1)
    optims_1.append(optimizer_1)

    # initialize z, l by running BP 100 iterations
    Q = np.zeros((N, I, J))
    # for n in range(N):
    y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[0, None, :, 1:, :], dtype="float32")).to(device)
    for iz in range(100):
        optims[0].zero_grad()#estimate the different source model of two channels
        loss = models[0].loss(y_abs)
        loss.backward()
        optims[0].step()
    Q[0, 1:I, :] = models[0].get_power_spec(cpu=True)

    plt.clf()
    plt.pcolormesh(y_abs[0, 0].detach().to("cpu").numpy()**.15)
    plt.savefig("./fig_aud_test/TwoGST/yfigure0_TwoGST.png")
    plt.close()

    plt.clf()
    plt.pcolormesh(Q[0]**.3)
    plt.savefig("./fig_aud_test/TwoGST/Qfigure0_TwoGST.png")
    plt.close()


    y_abs_1 = torch.from_numpy(np.asarray(Y_abs_array_norm[1, None, :, 1:, :], dtype="float32")).to(device)
    for iz in range(100):
        optims_1[0].zero_grad()
        loss_1 = models_1[0].loss(y_abs_1)
        loss_1.backward()
        optims_1[0].step()
    Q[1, 1:I, :] = models_1[0].get_power_spec(cpu=True)

    plt.clf()
    plt.pcolormesh(y_abs[0, 0].detach().to("cpu").numpy()**.15)
    plt.savefig("./fig_aud_test/TwoGST/yfigure1_TwoGST.png")
    plt.close()

    plt.clf()
    plt.pcolormesh(Q[1]**.3)
    plt.savefig("./fig_aud_test/TwoGST/Qfigure1_TwoGST.png")
    plt.close()


    Q = np.maximum(Q, epsi)
    gv = np.mean(np.divide(P[:, 1:I, :], Q[:, 1:I, :]), axis=(1, 2), keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat #Undate Vj

    # Algorithm for MVAE
    for it in range(AlgPara['it1']):
        Y_abs_array_norm = Y_abs / np.sqrt(gv)

        y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[0, None, None, 1:, :], dtype="float32")).to(device)
        for iz in range(100):
            optims[0].zero_grad()
            loss = models[0].loss(y_abs)
            loss.backward()
            optims[0].step()
        Q[0, 1:, :] = models[0].get_power_spec(cpu=True)

        plt.clf()
        plt.pcolormesh(y_abs[0, 0].detach().to("cpu").numpy()**.15)
        plt.savefig("./fig_aud_test/TwoGST/yfigure0_TwoGST_iteration" + str(it) + ".png")
        plt.close()

        plt.clf()
        plt.pcolormesh(Q[0]**.3)
        plt.savefig("./fig_aud_test/TwoGST/Qfigure0_TwoGST_iteration" + str(it) + ".png")
        plt.close()


        y_abs_1 = torch.from_numpy(np.asarray(Y_abs_array_norm[1, None, None, 1:, :], dtype="float32")).to(device)
        for iz in range(100):
            optims_1[0].zero_grad()
            loss_1 = models_1[0].loss(y_abs_1)
            loss_1.backward()
            optims_1[0].step()
        Q[1, 1:, :] = models_1[0].get_power_spec(cpu=True)

        plt.clf()
        plt.pcolormesh(y_abs[0, 0].detach().to("cpu").numpy()**.15)
        plt.savefig("./fig_aud_test/TwoGST/yfigure1_TwoGST_iteration" + str(it) + ".png")
        plt.close()

        plt.clf()
        plt.pcolormesh(Q[1]**.3)
        plt.savefig("./fig_aud_test/TwoGST/Qfigure1_TwoGST_iteration" + str(it) + ".png")
        plt.close()

        Q = np.maximum(Q, epsi)
        gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2), keepdims=True)
        Rhat = np.multiply(Q, gv)
        Rhat[:, 0, :] = R[:, 0, :]
        R = Rhat.transpose(1, 2, 0)

        # update W
        # W = update_w(X, R, W)
        W = update_w2(X, R, W, sv)
        # Y = X @ W.conj()
        Y = demix(Y, X, W)
        pattern = np.square(np.abs(np.einsum("km,ksm->ks",np.conjugate(sv[:, :, 0]),W)))
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)
        #label1 = models_1[0].get_label(cpu=True)

        if AlgPara['norm']:
            W, R, P = local_normalize(W, R, P, I, J)

        Y_abs = Y_abs.transpose(2, 0, 1)
        P = P.transpose(2, 0, 1)
        R = R.transpose(2, 0, 1)
        label1 = models[0].get_label(cpu=True)
        label2 = models_1[0].get_label(cpu=True)

    return Y, W, pattern, label1, label2


def gcav_iva(X, n_src=None, W=None, sv=None, lamb=None, c=0., maxiter=100, cost_flag=True, pb=True):
    """
    入力：
    X: サイズが周波数ｘフレーム数ｘチャンネル数の混合信号
    n_src: 音源数
    W: 初期分離行列（与えられない場合は自動的に単位行列で初期化する）
    sv: steering vector
    lamb: 正則化項のウェイト
    c: 正則化の度合いを調整するパラメータ
    maxiter: 更新回数
    pb: projection backを行うかどうか
    """

    # check input parameters
    n_freq, n_frame, n_ch = X.shape
    if n_src is None:
        n_src = X.shape[2]
    assert n_ch == n_src
    if sv.ndim == 2:
        sv = sv[..., None]
    n_sv = sv.shape[2]
    if type(lamb) is list:
        lamb = np.asarray(lamb)
    if not isinstance(c, list):
        c = [c] * n_src
    if W is None:
        W = np.array([np.eye(n_ch, n_src) for f in range(n_freq)], dtype=X.dtype)

    Y = np.zeros((n_freq, n_frame, n_src), dtype=X.dtype)
    Y = demix(Y, X, W)
    cost = np.zeros((3, maxiter))  # 全体コスト, IVAと正則化項それぞれのコスト

    for it in range(maxiter):
        # update W
        r = np.sqrt(np.sum(np.abs(Y) ** 2, axis=0))
        r_inv = 1. / np.maximum(r, eps)

        lamb = np.maximum(lamb * (1 - it / maxiter / 4), 0)

        for n in range(n_src):
            # G(r_k) = r_k
            V = np.matmul((X * r_inv[None, :, n, None]).swapaxes(1, 2), np.conj(X)) / n_frame

            # 出力チャネルの順番に正則化をかける
            # n_sv < n_srcの場合は、後ろのチャンネルに正則化をかけずにAuxIVAの更新式で更新する
            if n < n_sv:
                # 論文式(14)以下のD、及び(15)~(18)の計算
                D = V + lamb[n] * np.matmul(sv[:, :, n, None], np.conj(sv[:, :, n, None]).swapaxes(1, 2))
                D_inv = np.linalg.inv(D)
                u = np.conj(np.matmul(D_inv, np.linalg.inv(W))[:, :, n, None])
                u_hat = np.conj(lamb[n] * c[n] * np.matmul(D_inv, sv[:, :, n, None]))
                h = np.matmul(np.matmul(u.swapaxes(1, 2), D), np.conj(u))
                h_hat = np.matmul(np.matmul(u.swapaxes(1, 2), D), np.conj(u_hat))

                # 式(19) u前の係数の計算
                coef = np.zeros_like(h, dtype=h.dtype)
                coef[h_hat == 0] = 1 / np.sqrt(h[h_hat == 0])
                coef[h_hat != 0] = h_hat[h_hat != 0] * (-1 + np.sqrt(1 + (4 * h[h_hat != 0])
                                                                     / (np.abs(h_hat[h_hat != 0]) ** 2))) / (
                                               2 * h[h_hat != 0])
                w = coef * u + u_hat
                W[:, n, :] = w[:, :, 0]

            else:
                # AuxIVAのW更新式
                WV = np.matmul(W, V)
                W[:, n, :] = np.squeeze(np.conj(np.linalg.solve(WV, np.tile(np.eye(n_src)[:, n, None],
                                                                            (n_freq, 1, 1)))), axis=2)
                denom = np.matmul(np.matmul(W[:, n, None, :], V), np.conj(W[:, n, None, :].swapaxes(1, 2)))
                W[:, n, :] /= np.sqrt(denom[:, :, 0])

        Y = demix(Y, X, W)
        cost[:, it] = np.asarray(calc_loss(Y, W, sv, lamb, c))

    # projection back
    if pb:
        Y_pb = np.zeros_like(Y)
        W_inv = np.linalg.inv(W)
        for n in range(n_ch):
            image_Y = np.zeros_like(Y)
            image_Y[:, :, n] = Y[:, :, n]
            Y_pb[:, :, n] = np.squeeze(np.matmul(W_inv[:, n, None, :], image_Y.swapaxes(1, 2)).swapaxes(1, 2), axis=2)
        Y = Y_pb

    return Y, W, cost

def demix(Y, X, W):
    # Y, X: n_freq x n_frame x n_src/n_ch
    # W: n_freq x n_src x n_ch
    n_freq = Y.shape[0]
    for f in range(n_freq):
        Y[f] = np.dot(W[f], X[f].T).T  # 矩阵乘法运算

    return Y

def calc_iva_loss(Y, W):
    W_cost = np.sum(np.log(np.abs(np.linalg.det(W))))
    src_cost = np.sum(np.mean(np.sqrt(np.sum(np.abs(Y) ** 2, axis=0)), axis=0))
    cost_iva = src_cost - W_cost

    return cost_iva

def calc_loss(Y, W, sv, lamb, c=0):
    cost_iva = calc_iva_loss(Y, W)
    n_sv = sv.shape[2]

    penalty_n = 0
    for n in range(n_sv):
        w_n = W[:, n, None, :]
        penalty_n += lamb[n] * np.sum(np.abs(np.matmul(w_n, sv[:, :, n, None]) - c) ** 2)
    cost = cost_iva + penalty_n

    return cost, cost_iva, penalty_n

#### Local functions ####
def ilrma(X, W, R, P, T, V, iteration, normalise):
    I, J, N = X.shape
    for n in range(N):
        R[:, :, n] = T[:, :, n] @ V[:, :, n]  # low-rank source model

    # Iterative update
    for it in range(iteration):
        for n in range(N):
            # Update T
            T[:, :, n] = T[:, :, n] * np.sqrt(
                (P[:, :, n] * (R[:, :, n] ** -2)) @ V[:, :, n].T / (R[:, :, n] ** -1 @ V[:, :, n].T))
            T[:, :, n] = np.maximum(T[:, :, n], epsi)
            R[:, :, n] = T[:, :, n] @ V[:, :, n]
            # Update V
            V[:, :, n] = V[:, :, n] * np.sqrt(
                T[:, :, n].T @ (P[:, :, n] * R[:, :, n] ** -2) / (T[:, :, n].T @ (R[:, :, n] ** -1)))
            V[:, :, n] = np.maximum(V[:, :, n], epsi)
            R[:, :, n] = T[:, :, n] @ V[:, :, n]
        # Update W
        W = update_w(X, R, W)

        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)
        if normalise:
            W, R, P, T = local_normalize(W, R, P, I, J, T)

    if iteration == 0:
        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)
        R = P
        if normalise:
            W, R, P, T = local_normalize(W, R, P, I, J, T)

    return Y, W, R, P


def local_normalize(W, R, P, I, J, *args):
    lamb = np.sqrt(np.sum(np.sum(P, axis=0), axis=0) / (I * J))  # 1 x 1 x N

    W = W / np.squeeze(lamb)
    lambPow = lamb ** 2
    P = P / lambPow
    R = R / lambPow
    if len(args) == 1:
        T = args[0]
        T = T / lambPow
        return W, R, P, T
    elif len(args) == 0:
        return W, R, P


def update_w(s, r, w):
    L = w.shape[-1]
    _, N, M = s.shape
    sigma = np.einsum('fnp,fnl,fnq->flpq', s, 1 / r, s.conj())
    sigma /= N
    for l in range(L):
        w[..., l] = LA.solve(
            w.swapaxes(-2, -1).conj() @ sigma[:, l, ...],
            np.eye(L)[None, :, l])
        den = np.einsum(
            'fp,fpq,fq->f',
            w[..., l].conj(), sigma[:, l, ...], w[..., l])
        w[..., l] /= np.maximum(np.sqrt(np.abs(den))[:, None], 1.e-8)
    w += epsi * np.eye(M)
    return w


def update_w2(s, r, W, sv):
    L = W.shape[-1]
    lamb = [10, 10]
    c = [1, 0.02]
    #lamb = np.maximum(lamb * (1 - it / maxiter / 4), 0)
    _, N, M = s.shape
    sigma = np.einsum('fnp,fnl,fnq->flpq', s, 1 / r, s.conj())
    sigma /= N
    for l in range(L):
        # 論文式(14)以下のD、及び(15)~(18)の計算
        D = sigma[:, l, ...] + lamb[l] * np.matmul(sv[:, :, l, None], np.conj(sv[:, :, l, None]).swapaxes(1, 2))
        D_inv = np.linalg.inv(D)
        u = np.conj(np.matmul(D_inv, np.linalg.inv(W))[:, :, l, None])
        u_hat = np.conj(lamb[l] * c[l] * np.matmul(D_inv, sv[:, :, l, None]))
        h = np.matmul(np.matmul(u.swapaxes(1, 2), D), np.conj(u))
        h_hat = np.matmul(np.matmul(u.swapaxes(1, 2), D), np.conj(u_hat))

        # 式(19) u前の係数の計算
        coef = np.zeros_like(h, dtype=h.dtype)
        coef[h_hat == 0] = 1 / np.sqrt(h[h_hat == 0])
        coef[h_hat != 0] = h_hat[h_hat != 0] * (-1 + np.sqrt(1 + (4 * h[h_hat != 0])
                                                                     / (np.abs(h_hat[h_hat != 0]) ** 2))) / (
                                               2 * h[h_hat != 0])

        w = coef * u + u_hat
        W[:, l, :] = w[:, :, 0]
    return W
