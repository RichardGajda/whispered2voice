from __future__ import print_function
import numpy as np
import argparse
import soundfile as sf
import os


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number_of_pairs", required=True, type=int, default=1)
parser.add_argument("-f", "--frame_period", required=True, type=float, default=5.0)
parser.add_argument("-sr", "--speaker", required=True, type=str)

EPSILON = 1e-8

def windower(x, M, N):

    # M Advance between each window
    # N windowsize

    T = x.shape[0]
    m = np.arange(0, T - N + 1, M)          # Start of windowing
    L = m.shape[0]                          # N windows
    ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1, L)) + np.ones((N, 1)) * m
    X = x[ind.astype(int)]
    return X.transpose()


def silence_filter(x, frame_period, fs, th=0.0025):
    frame_samples = np.round(fs * frame_period / 1000).astype(int)

    N = (6 * frame_samples).astype(int)
    M = (2 * frame_samples).astype(int)

    x_w = windower(x, M, N)

    x_power = np.apply_along_axis(np.power, 0, np.absolute(x_w), 2)
    x_power = np.apply_along_axis(np.sum, 1, x_power)

    speech_th = np.array(np.where(x_power > th))
    speech_index = (speech_th * M).astype(int)
    noise_th = np.array(np.where(x_power <= th))
    noise_index = (noise_th * M).astype(int)
    x_fil = np.concatenate([x[offset:(offset + M)] for offset in speech_index[0]])
    return x_fil, speech_index, noise_index, M

### SILENCE REMOVAL ###
def main(args):

    pairs = args.number_of_pairs
    frame_period = args.frame_period
    speaker = args.speaker
    for i in range(pairs):

        print('Utterance {}'.format(i+1))

        ### LOAD THE NEUTRAL AND WHISPERED SPEECH ###
        x_n, fs_n = sf.read("{}/{:03d}_n.wav".format(speaker, i + 1))
        x_w, fs_w = sf.read("{}/{:03d}_w.wav".format(speaker, i + 1))

        ### REMOVE THE SILENCE ###
        x_n_rs, speech_index, noise_index, M = silence_filter(x_n, frame_period, fs_n)
        x_w_rs, speech_index, noise_index, M = silence_filter(x_w, frame_period, fs_w)

        MYDIR = ("{}_sr".format(speaker))
        CHECK_FOLDER = os.path.isdir(MYDIR)

        ### FOLDER CHECK ###
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        ### WRITE THE WAVEFORMS RID OF SILENCE ###

        sf.write(("{}_sr/{:03d}_n_sr.wav".format(speaker, i + 1)), x_n_rs, fs_n)
        sf.write(("{}_sr/{:03d}_w_sr.wav".format(speaker, i + 1)), x_w_rs, fs_w)


if __name__ == '__main__':
        args = parser.parse_args()
        main(args)
