import pyworld as pw
import soundfile as sf
import numpy as np
import argparse
import os.path
from lstm import MyBLSTM
import torch

mcep_n = 26
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", required=True)
parser.add_argument("-g", "--gender", required=True)
w_f_dim = 78
num_hidden = 256


def delta(feat, N):
    ## taken from https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N + 1),
                               padded[t: t + 2 * N + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def get_pred(gender, typ, w_f):
    if typ == "f0":
        input_dim = 1
    elif typ == "ap":
        input_dim = 5
    elif typ == "mcep":
        input_dim = 50
        w_f_dim_mcep = 150

    if typ == "mcep":
        model = MyBLSTM(w_f_dim_mcep, num_hidden, input_dim, bidirectional=True, num_layers=2)
        model.load_state_dict(
            torch.load('models/{}/{}_model_state_complete'.format(gender, typ), map_location=torch.device('cpu')))
    else:
        model = MyBLSTM(w_f_dim, num_hidden, input_dim, bidirectional=True, num_layers=2)
        model.load_state_dict(
            torch.load('models/{}/{}_model_state_complete'.format(gender, typ), map_location=torch.device('cpu')))

    model.cpu()
    model.eval()

    seq = torch.from_numpy(w_f).float()

    try:
        param_predicted = model(seq).data.numpy()
    except:
        h, c = model.init_hidden(batch_size=1)
        seq_len = len(seq)
        seq = seq.view(1, -1, seq.size(-1))
        param_predicted = model(seq, [seq_len], h, c).data.double().numpy()

        param_predicted = np.reshape(param_predicted, (seq_len, input_dim))

        print(typ, " conversion done") # DEB FOR SERVER LOGS
        return param_predicted


def get_params(file):

    # READ THE FILE, PARSE IT INTO WORLD, GET THE PARAMETERS (F0 IS NEEDED FOR SPECTRAL ENVELOPE)

    x, fs = sf.read(file)

    _f0_x, t_x_h = pw.harvest(x, fs)
    f0_x = pw.stonemask(x, _f0_x, t_x_h, fs)
    sp_x = pw.cheaptrick(x, f0_x, t_x_h, fs)
    fft = pw.get_cheaptrick_fft_size(fs)

    # CODE SPECTRAL ENVELOPE INTO MCEPS. 26 MELS FOR F0 AND AP PREDICTION, 50 FOR MCEP PREDICTION

    sp_x_mcep = pw.code_spectral_envelope(sp_x, fs, mcep_n)
    sp_x_mcep_new = pw.code_spectral_envelope(sp_x, fs, 50)

    # APPLY DELTAS ON BOTH MCEPS

    sp_delta = delta(sp_x_mcep, 2)
    sp_doubledelta = delta(sp_delta, 2)
    matrix = []

    # APPEND THE DELTAS TO MCEPS

    for j in range(len(sp_x_mcep)):
        tmp = np.append(sp_x_mcep[j], sp_delta[j])
        tmp = np.append(tmp, sp_doubledelta[j])
        matrix.append(tmp)

    w_features = np.asarray(matrix)

    sp_delta_mcep = delta(sp_x_mcep_new, 2)
    sp_doubledelta_mcep = delta(sp_delta_mcep, 2)
    matrix_mcep = []

    for j in range(len(sp_x_mcep_new)):
        tmp = np.append(sp_x_mcep_new[j], sp_delta_mcep[j])
        tmp = np.append(tmp, sp_doubledelta_mcep[j])
        matrix_mcep.append(tmp)




    w_features_mcep = np.asarray(matrix_mcep)
    print("Features extracted")                 # DEB FOR SERVER LOGS

    return w_features, w_features_mcep, fft, fs


def main(file, gender):
    global mcep_pred, ap_pred, f0_pred
    if gender != "male" and gender != "female":
        print("Invalid gender")
        return -1
    else:

        # GET WHISPERED FEATURES.

        w_features, w_features_mcep, fft, fs = get_params(file)

        # GET PREDICTED PARAMETERS FROM MODEL INFERENCE

        f0_pred = get_pred(gender, "f0", w_features)
        ap_pred = get_pred(gender, "ap", w_features)
        mcep_pred = get_pred(gender, "mcep", w_features_mcep)

        # RESTORE DIMENSIONALITY FOR AP AND SP #

        ap_restored = pw.decode_aperiodicity(ap_pred, fs, fft)
        sp_restored = pw.decode_spectral_envelope(mcep_pred, fs, fft)

        # RESHAPE F0 #
        f0_pred = np.reshape(f0_pred, len(f0_pred))

        x_pred = pw.synthesize(f0_pred, sp_restored, ap_restored, fs, 5)

        return (x_pred, fs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
