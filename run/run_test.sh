#!/bin/sh

# Running test.py

python3 test.py \
	-i ./data/distance/ \
	-o ./output_distance/ \
	-n0 0 \
	-n1 30 \
	-m ./model_TarCVAE/Noisy_TarCVAE_GST_2048/1000.model \
	-m1 ./model_mixture_new_gst_2-20_noisy_multiFolder_new_1000/mixture_gst_2-20_noisy_multiFolder_new_1000/1000.model \
	--noise_file ./noise_sample/ch01_office_cut1.wav\
	--snr 0


