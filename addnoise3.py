import scipy.io.wavfile as wav
import scipy
import numpy as np
import librosa
import soundfile as sf
import math
import numpy as np
import random
import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--clean_file', type=str, default='', help='file of clean wave')
	parser.add_argument('--noise_file', type=str, default='', help='file of noise wave')
	parser.add_argument('--mix_file', type=str, default='', help='file of mix wave')
	parser.add_argument('--snr', type=float, default='', help='SNR')
	args = parser.parse_args()
	return args

def signal_by_db(x1, x2, snr, handle_method):
    from numpy.linalg import norm
    x1 = x1.astype(np.int16)
    x2 = x2.astype(np.int16)
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 != l2:
        print("l1!=l2")
        if handle_method == 'cut':
            ll = min(l1, l2)
            print(l1)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            x2 = x2[:250000]  # 让噪声小于纯净语音
            ll = max(l1, l2)
            if l2 < ll:
                x2_total = []
                for i in range(int(l1 // l2) * 2):  # "//"表示整数除法,返回不大于结果的一个最大的整数
                    x2_total.append(x2)  # 添加噪声
                x2_total = np.hstack(x2_total)

                ll2 = x1.shape[0]

                x2 = x2_total[:ll2]
            print(np.shape(x1))
            print(np.shape(x2))

    x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr))

    mix = x1 + x2
    return mix, x2


def SNR_mixed(audio_path, noise_path, out_path, SNR):

    #src, sr = librosa.core.load(audio_path, sr=sr)
    #clean = "clean_wsj.wav"
    #noise = "ch01_2.wav"
    fs, clean = scipy.io.wavfile.read(audio_path)
    fs, noise = scipy.io.wavfile.read(noise_path)

    noisy_factory_test, _ = signal_by_db(clean, noise, SNR, 'cut')
    noisy_factory_test = np.asarray(noisy_factory_test, dtype=np.int16)
    #wav.write("timit_speech_factory_snr%s_test.wav" % snr, fs, noisy_factory_test)
    #sf.write("timit_speech_factory_snr%s_test2.wav" % snr, noisy_factory_test, fs)
    sf.write(out_path, noisy_factory_test, fs)
    print("The snr is ：", SNR)

def main():
    args = get_args()
    #parameter and path settings
    path = {'clean_file': args.clean_file, 'noise_file': args.noise_file, 'mix_file': args.mix_file}
    condition = {'snr': args.snr}

    clean_root = args.clean_file
    clean_dir_names = sorted(os.listdir(clean_root))
    mix_root = args.mix_file
    os.makedirs(mix_root, exist_ok=True)

    for idx, f in enumerate(clean_dir_names):
        clean_dir = os.path.join(clean_root, f)
        save_dir = mix_root
        os.makedirs(save_dir, exist_ok=True)

        noisename = args.noise_file
        SNR = args.snr
        mix_name = f[0:-4] + ' noisy.wav'
        SNR_mixed(clean_dir, noisename, os.path.join(save_dir, mix_name), SNR)

if __name__=="__main__":
    main()