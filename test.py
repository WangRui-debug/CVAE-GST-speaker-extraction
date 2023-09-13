# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Test file


import os
import argparse
import utils
import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile, savemat
from scipy import signal
from MVAE5_2GST import MVAE
from addnoise3 import SNR_mixed
from function import *
import sys
from mir_eval.separation import bss_eval_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for MVAE")
    parser.add_argument('--gpu', '-g', type=int, help="GPU ID (negative value indicates CPU)", default=-1)
    parser.add_argument('--input_root', '-i', type=str, help="path for input test data", default="./data/test_input/")
    parser.add_argument('--output_root', '-o', type=str, help="path for output data", default="./output/")
    parser.add_argument('--n_src', '-n', type=int, help="number of dimensions", default=128)
    parser.add_argument('--n_src1', '-n2', type=int, help="number of dimensions", default=128)

    parser.add_argument('--fs', '-r', type=int, help="Resampling frequency", default=16000)
    parser.add_argument('--fft_size', '-l', type=int, help="Frame length of STFT in sample points", default=2048)
    parser.add_argument('--shift_size', '-s', type=int, help="Frame shift of STFT in samplie points", default=1024)

    parser.add_argument('--nb', '-nb', type=int, help="Number of basis for initialization", default=1)
    parser.add_argument('--n_itr0', '-n0', type=int, help="Number of iterations for initialization using ILRMA",
                        default=0)
    parser.add_argument('--n_itr1', '-n1', type=int, help="Number of iterations for MVAE", default=30)

    parser.add_argument('--model_path', '-m', type=str, help="path for a trained encoder")
    parser.add_argument('--model_path1', '-m1', type=str, help="path for a trained encoder1")

    parser.add_argument('--noise_file', type=str, default='./ch01_office_cut1.wav', help='file of noise wave')
    parser.add_argument('--snr', type=float, default=30, help='SNR')

    args = parser.parse_args()

    # ============== parameter and path settings ================
    STFTPara = {'fs': args.fs, 'window_size': args.fft_size, 'window_shift': args.shift_size, 'type': "hamming"}
    AlgPara = {'it0': args.n_itr0, 'it1': args.n_itr1,
               'nb': args.nb, 'whitening': False, 'norm': True, 'RefMic': 0}
    NNPara = {'n_src': args.n_src, 'model_path': args.model_path}
    NNPara1 = {'n_src1': args.n_src1, 'model_path1': args.model_path1}

    # path
    # input_root = args.input_root
    # output_root = args.output_root
    # os.makedirs(output_root, exist_ok=True)
    # input_dir_names = sorted(os.listdir(input_root))

    d = np.asarray([5, 5.05])  # linearマイク座標(x軸のみ)
    win_len = 2048
    win_ol = 1024
    DOA = [90, 90]
    lamb = [10, 10]
    c = [1, 0.02]

    gpu = args.gpu

    wav_files = [
        ['./test_audio/clean/421/421a010g.wav'],
        ['./test_audio/clean/423/423a0111.wav'],
        ['./test_audio/clean/053/053a050v.wav'],
    ]

    signals = [np.concatenate([wavfile.read(f)[1].astype(np.float32)
                               for f in source_files])
               for source_files in wav_files]


    locations_0 = [[5, 1.7, 0.5], [5.87, 1.2, 0.5], [6, 0.7, 0.5]]
    locations_1 = [[5, 2.2, 0.5], [locations_0[1][0] + 0.43, locations_0[1][1] + 0.25, 0.5],
                   [locations_0[2][0] + 0.5, locations_0[2][1], 0.5]]
    locations_2 = [[5, 2.7, 0.5], [locations_1[1][0] + 0.43, locations_1[1][1] + 0.25, 0.5],
                   [locations_1[2][0] + 0.5, locations_1[2][1], 0.5]]
    locations_3 = [[5, locations_2[0][1] + 0.5, 0.5], [locations_2[1][0] + 0.43, locations_2[1][1] + 0.25, 0.5],
                   [locations_2[2][0] + 0.5, locations_2[2][1], 0.5]]
    locations_4 = [[5, locations_3[0][1] + 0.5, 0.5], [locations_3[1][0] + 0.43, locations_3[1][1] + 0.25, 0.5],
                   [locations_3[2][0] + 0.5, locations_3[2][1], 0.5]]
    locations_5 = [[5, locations_4[0][1] + 0.5, 0.5], [locations_4[1][0] + 0.43, locations_4[1][1] + 0.25, 0.5],
                   [locations_4[2][0] + 0.5, locations_4[2][1], 0.5]]
    locations_6 = [[5, locations_5[0][1] + 0.5, 0.5], [locations_5[1][0] + 0.43, locations_5[1][1] + 0.25, 0.5],
                   [locations_5[2][0] + 0.5, locations_5[2][1], 0.5]]

    location = [locations_0, locations_1, locations_2, locations_3, locations_4, locations_5, locations_6]

    for locations in location:

        if locations == locations_0:
            distance = 1
        elif locations == locations_1:
            distance = 1.5
        elif locations == locations_2:
            distance = 2
        elif locations == locations_3:
            distance = 2.5
        elif locations == locations_4:
            distance = 3
        elif locations == locations_5:
            distance = 3.5
        elif locations == locations_6:
            distance = 4

        input_root = args.input_root

        # create demo room
        corners = np.array([[0, 0], [0, 6], [10, 6], [10, 0]]).T  # [x,y]
        room = pra.Room.from_corners(corners)

        # Room 4m by 6m
        room_dim = [10, 6]

        # create an anechoic room with sources and mics
        # set max_order to a low value for a quick (but less accurate) RIR
        room = pra.Room.from_corners(corners, fs=16000, max_order=1, materials=pra.Material(0.99, 0.99),
                                     ray_tracing=True, air_absorption=True)
        room.extrude(1., materials=pra.Material(0.99, 0.99))

        for sig, loc in zip(signals, locations):
            room.add_source(loc, signal=sig, delay=0)

        # add two-microphone array
        R = np.array([[5, 5.05], [0.7, 0.7], [0.5, 0.5]])  # [[x], [y], [z]]
        room.add_microphone(R)

        # compute image sources
        room.image_source_model()
        room.simulate()
        fig, ax = room.plot()

        # visualize 3D polyhedron room and image sources
        # fig, ax = room.plot(img_order=3)
        # fig.set_size_inches(18.5, 10.5)
        # Simulate
        # The premix contains the signals before mixing at the microphones混合前麦克风接收到的信号，即卷积后混合前的信号
        # shape=(n_sources, n_mics, n_samples)
        separate_recordings = room.simulate(return_premix=True)  # premix即为混合前的卷积信号

        # Mix down the recorded signals (n_mics, n_samples)
        # i.e., just sum the array over the sources axis

        mics_signals = np.sum(separate_recordings, axis=0)

        input_root = os.path.join(input_root,
                                  'test_m3_' + str(distance) + 'm_3D1_GST_noisy' + '_snr' + str(args.snr))
        os.makedirs(input_root, exist_ok=True)
        clean001 = os.path.join(input_root, 'clean001')
        os.makedirs(clean001, exist_ok=True)
        test001 = os.path.join(input_root, 'test001')
        os.makedirs(test001, exist_ok=True)
        input_dir_names = ['test001']
        # test002 = os.path.join(input_root, 'test002')
        # os.makedirs(test002, exist_ok=True)

        # wavfile.write(os.path.join(test001, 'estimated_signal7.wav'), STFTPara['fs'], sep.astype(np.int16))
        wavfile.write(os.path.join(clean001, 'm3_3D_' + str(distance) + 'm_1.wav'), room.fs,
                      mics_signals[0].astype(np.int16))
        wavfile.write(os.path.join(clean001, 'm3_3D_' + str(distance) + 'm_2.wav'), room.fs,
                      mics_signals[1].astype(np.int16))

        # add noise
        clean_root = clean001
        clean_dir_names = sorted(os.listdir(clean_root))
        mix_root = test001
        # os.makedirs(mix_root, exist_ok=True)

        for idx, f in enumerate(clean_dir_names):
            clean_dir = os.path.join(clean_root, f)
            save_dir = mix_root
            # os.makedirs(save_dir, exist_ok=True)

            noisename = args.noise_file
            SNR = args.snr
            mix_name = f[0:-4] + '_snr' + str(SNR) + '_noisy_office.wav'
            SNR_mixed(clean_dir, noisename, os.path.join(save_dir, mix_name), SNR)

        mics_normal = mics_signals / np.max(np.abs(mics_signals))
        ref = separate_recordings[:, 0, :]
        SDR, SIR = [], []
        # ref_normal = preprocessing.scale(ref) # 归一化
        ref_normal = ref / np.max(np.abs(ref))  # 归一化

        # save_root = './beam_pattern_distance_90_45_20ea_1.5m_2D1'
        # wavfile.write('m3_90_45_20ea_3D1_1.5m_wav_1.wav', room.fs, mics_signals[0])
        # wavfile.write('m3_90_45_20ea_3D1_1.5m_wav_2.wav', room.fs, mics_signals[1])

        # ================ separation ================
        for idx, f in enumerate(input_dir_names):
            input_dir = os.path.join(input_root, f)
            print("Processing {}...".format(input_dir), end="")
            output_root = args.output_root
            output_root = os.path.join(output_root,
                                       'output_m3_' + str(distance) + 'm_SNR_' + str(SNR) + '_3D/')
            os.makedirs(output_root, exist_ok=True)
            save_dir = os.path.join(output_root, f)
            os.makedirs(save_dir, exist_ok=True)

            # Input data and resample
            mix = utils.load_wav(input_dir, STFTPara['fs'])
            ns = mix.shape[1]

            # STFT
            frames_ = np.floor((mix.shape[0] + 2 * STFTPara['window_shift']) / STFTPara['window_shift'])  # to meet NOLA
            frames = int(np.ceil(frames_ / 8) * 8)

            X = np.zeros((int(STFTPara['window_size'] / 2 + 1), int(frames), mix.shape[1]), dtype=np.complex)
            for n in range(mix.shape[1]):
                f, t, X[:, :int(frames_), n] = signal.stft(mix[:, n], nperseg=STFTPara['window_size'],
                                                           window=STFTPara['type'],
                                                           noverlap=STFTPara['window_size'] - STFTPara['window_shift'])

            # paraments for seering vector
            n_ch = mix.shape[1]
            n_src2 = n_ch
            for i in range(n_ch):
                f, _, cspec = signal.stft(mix[:, i], 16000, nperseg=win_len, noverlap=win_ol)
                # x2 = cspec[..., None] if i == 0 else np.concatenate((mix, cspec[..., None]), axis=2)

            # generate steering vector from DOA
            n_c = len(DOA)
            sv = np.zeros((f.size, n_c, n_src2), dtype="complex")
            for i in range(n_c):
                sv[:, :, i, None] = generate_sv(d, DOA[i], f2omega(f))

            # source separation
            Y, W, pattern, label1, label2 = MVAE(X, AlgPara, NNPara, NNPara1, sv, gpu)

            # beam pattern
            save_folder = './beampatter/beam_pattern_distance_' + str(distance) + 'm' + '_snr' + str(SNR)
            os.makedirs(save_folder, exist_ok=True)
            # save_folder = './beam_pattern_new_60_01ea'

            for i in range(n_ch):
                # plot signals and directivity patterns for investigation
                pattern, f_idx = directivity(W[:, i, None, :], f, d)[0:2]
                pattern = pattern[::-1]
                plot_directivity(pattern[:, :, 0], path=os.path.join(save_folder, "directivity_{}.png".format(i + 1)))
                plot_directivity3D(pattern[:, :, 0], f_idx,
                                   path=os.path.join(save_folder, "directivity3D_{}.png".format(i + 1)))

            # projection back
            XbP = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.complex)
            XbP[:, :, 0] = X[:, :, AlgPara['RefMic']]
            Z = utils.back_projection(Y, XbP)

            # TF-mask
            ps1 = np.abs(Z[:, :, 0]) ** 2
            ps2 = np.abs(Z[:, :, 1]) ** 2
            # ps3 = np.abs(X[:, :, 0]) ** 2
            liner = X[:, :, 1] - Z[:, :, 1]
            ps3 = np.abs(liner) ** 2
            mask = np.divide(ps1, np.maximum(ps1 + ps2, sys.float_info.epsilon))
            mask2 = 1 - np.divide(ps2, np.maximum(ps3 + ps2, sys.float_info.epsilon))
            mask3 = 1 - np.divide(ps2, np.maximum(ps1 + ps2, sys.float_info.epsilon))
            np.amax(mask)
            np.amax(mask2)
            Zmaked = np.multiply(Z[:, :, 0], mask)
            Zmaked2 = np.multiply(Z[:, :, 0], mask2)
            Zmaked3 = np.multiply(Z[:, :, 0], mask3)
            Zmaked4 = np.multiply(X[:, :, 0], mask3)
            Zmaked5 = np.multiply(liner, mask3)
            Zmaked6 = np.multiply(X[:, :, 0], mask2)
            Zmaked7 = np.multiply(liner, mask2)
            Zmaked8 = np.multiply(liner, mask)
            Zmaked9 = np.multiply(X[:, :, 0], mask)

            # iSTFT and save
            sep = signal.istft(Z[:, :, 1], window=STFTPara['type'])[1]
            sep = sep / max(abs(sep)) * 30000
            wavfile.write(os.path.join(save_dir, 'estimated_signal_interference.wav'), STFTPara['fs'], sep.astype(np.int16))

            # sep = signal.istft(Zmaked2, window=STFTPara['type'])[1]
            # sep = sep / max(abs(sep)) * 30000
            # wavfile.write(os.path.join(save_dir, 'estimated_signal2.wav'), STFTPara['fs'], sep.astype(np.int16))

            sep3 = signal.istft(Zmaked3, window=STFTPara['type'])[1]
            sep3 = sep3 / max(abs(sep3)) * 30000
            wavfile.write(os.path.join(save_dir, 'estimated_signal_target.wav'), STFTPara['fs'], sep3.astype(np.int16))
            sep3 = sep3 / max(abs(sep3))
            sep3 = sep3.reshape(len(sep3), 1)
            sig2 = np.append(sep3, sep3, axis=1)
            sig3 = np.append(sig2, sep3, axis=1)
            sig3 = sig3.T
            m = np.minimum(sig3.shape[1], ref_normal.shape[1])
            sdr, sir, sar, perm = bss_eval_sources(ref_normal[:, :m], sig3[:, :m])




            print("Done with sir, sdr and sar of " + str(distance) + 'm with ' + '_snr' + str(SNR) + ' of ' + str(
                sir[0]) + ", " + str(sdr[0]) + ", " + str(sar[0]))
            strcontent = "Done with sir, sdr and sar of " + str(distance) + 'm with' + '_snr' + str(SNR) + ' of ' + str(
                sir[0]) + ", " + str(sdr[0]) + ", " + str(sar[0])

            sir = 0
            sdr = 0
            sar = 0

            output_root = args.output_root

            f = open(os.path.join(output_root, 'results_m3_GST_noisy' + '_snr' + str(SNR) + '.txt'), 'a')
            f.write(strcontent)
            f.write('\n')

print("Done!")
