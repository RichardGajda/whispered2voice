import argparse
import soundfile as sf
import pyworld as pw
import numpy as np
import math
import os
from mcd import dtw


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number_of_pairs", required=True, type=int, default=1)
parser.add_argument("-s", "--starting_fn", type=str, default="001")
parser.add_argument("-e", "--ending_fn", type=str, default="001")
parser.add_argument("-f", "--frame_period", type=float, default=5.0)
parser.add_argument("-sp", "--speed", type=int, default=1)
parser.add_argument("-m", "--mceps", type=int, default=50)
parser.add_argument("-sr", "--speaker", required=True, type=str)


def duration_str(duration):
    hours, rest = divmod(duration, 3600)
    minutes, seconds = divmod(rest, 60)
    hours = int(hours)
    minutes = int(minutes)

    duration = "{:02d}h {:02d}m {:02.03f}s".format(hours, minutes, seconds)
    return duration


def delta(feat, N):

    ## taken from https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py

    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def eucCost(x, y):
    return math.sqrt(np.inner(x - y, x - y))


def main(args):

    start = args.starting_fn
    pairs = args.number_of_pairs
    speaker = args.speaker
    m = args.mceps                                                  #order of MCEPs, default value set to 50
    total_duration = 0                                              #aux variable for counting total duration of material
    np.set_printoptions(edgeitems=0)
    for i in range(pairs):
        i = i + int(start)
        print("\nUtterance {}\n".format(i))   # DEB

        ###################################
        # PROCESSING NEUTRAL SPEECH FIRST #
        ###################################

        # Load the file
        x_n, fs_n = sf.read("{}_sr/{:03d}_n_sr.wav".format(speaker, i))

        # Extract fundamental freq, spectrum and aperiodicity lvls using harvest, optimizing with stonemask, ct and d4c

        _f0_n_h, t_n_h = pw.harvest(x_n, fs_n)
        f0_n_h = pw.stonemask(x_n, _f0_n_h, t_n_h, fs_n)            # F0 of neutral speech for BLSTM training
        sp_n_h = pw.cheaptrick(x_n, f0_n_h, t_n_h, fs_n)            # sp of neutral speech extraction
        ap_n_h = pw.d4c(x_n, f0_n_h, t_n_h, fs_n)                   # ap of neutral speech for BLSTM training

        # Extract MCEP's from spectrum #

        sp_mcep_n = pw.code_spectral_envelope(sp_n_h, fs_n, m)      # Reduced dimensionality of sp for BLSTM training
        print("{} frames detected".format(len(f0_n_h)))

        # Reduce dimensionality of AP levels

        ap_n_h_r = pw.code_aperiodicity(ap_n_h, fs_n)

        # DEB print section

        info_n = sf.info("{}_sr/{:03d}_n_sr.wav".format(speaker, i))
        print("Neutral speech duration: {}".format(info_n._duration_str))

        ###############################
        # PROCESSING WHISPERED SPEECH #
        ###############################

        # Load the file
        x_w, fs_w = sf.read("{}_sr/{:03d}_w_sr.wav".format(speaker, i))

        # Extract fundamental freq, spectrum and aperiodicity lvls using harvest, optimizing with stonemask, ct and d4c

        _f0_w_h, t_w_h = pw.harvest(x_w, fs_w)
        f0_w_h = pw.stonemask(x_w, _f0_w_h, t_w_h, fs_w)            # F0 of whispered speech for BLSTM training
        sp_w_h = pw.cheaptrick(x_w, f0_w_h, t_w_h, fs_w)            # sp of whispered speech extraction

        # Extract MCEP's from spectrum #

        sp_mcep_w = pw.code_spectral_envelope(sp_w_h, fs_w, m)      # Reduced dimensionality of sp for BLSTM training
        print("{} frames detected".format(len(f0_w_h)))

        sp_delta = delta(sp_mcep_w, 2)                              # counting delta and double delta features
        sp_doubledelta = delta(sp_delta, 2)
        matrix = []
        for j in range(len(sp_mcep_w)):
            tmp = np.append(sp_mcep_w[j], sp_delta[j])
            tmp = np.append(tmp, sp_doubledelta[j])
            matrix.append(tmp)

        matrix = np.asarray(matrix)                                 # MCEP + Delta Features matrix for DTW

        # DTW

        w_IndexSeq = dtw.findWarpingBestCost(sp_mcep_n, sp_mcep_w, eucCost)
        w_features = dtw.warpGeneral(matrix, w_IndexSeq)

        # DEB print section

        info_w = sf.info("{}_sr/{:03d}_w_sr.wav".format(speaker, i))
        print("Whispered speech duration: {}".format(info_w._duration_str))

        total_duration = (total_duration + info_n.duration + info_w.duration)

        dir = ("dataset/{}".format(speaker))
        CHECK_FOLDER = os.path.isdir(dir)

        ### FOLDER CHECK ###
        if not CHECK_FOLDER:
            os.makedirs(dir)
            print("created folder : ", dir)

        np.savez("{}/{:03d}_ds".format(dir, i), f0=f0_n_h, sp=sp_mcep_n, ap=ap_n_h_r, w_features=w_features)

    str = duration_str(total_duration)
    print("Total duration: {}".format(str))

if __name__ == '__main__':
        args = parser.parse_args()
        main(args)