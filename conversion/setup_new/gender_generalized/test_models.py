import pyworld as pw
import soundfile as sf
import numpy as np
import argparse
import os.path
from lstm import MyBLSTM
import torch


mcep_n = 50
parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--speaker", required=True, type=str)
w_f_dim = 150
num_hidden = 256


def delta(feat, N):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def get_pred(speaker, typ, w_f):

    if typ == "f0":
        input_dim = 1
    elif typ == "ap":
        input_dim = 5
    elif typ == "mcep":
        input_dim = 50

    model = MyBLSTM(w_f_dim, num_hidden, input_dim, bidirectional=True, num_layers=2)
    model.load_state_dict(torch.load('training/{}/{}_model_state_complete'.format(speaker, typ)))

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

        return param_predicted


def get_params(file):

    x, fs = sf.read(file)

    _f0_x, t_x_h = pw.harvest(x, fs)
    f0_x = pw.stonemask(x, _f0_x, t_x_h, fs)
    sp_x = pw.cheaptrick(x, f0_x, t_x_h, fs)
    fft = pw.get_cheaptrick_fft_size(fs)
    sp_x_mcep = pw.code_spectral_envelope(sp_x, fs, mcep_n)
    sp_x_mcep_new = pw.code_spectral_envelope(sp_x, fs, 50)

    sp_delta = delta(sp_x_mcep, 2)
    sp_doubledelta = delta(sp_delta, 2)
    matrix = []

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

    return w_features, w_features_mcep, fft, fs


def main(args):

    speaker = args.speaker

    folder = '{}_sr'.format(speaker)
    files = os.listdir('{}_sr'.format(speaker))
    print(files)
    files = files[-19::2] ## LIST LAST 10 whispered utterances

    for file in files:
        abs_path = os.path.join(folder, file)
        w_features, w_features_mcep, fft, fs = get_params(abs_path)

        # GET PREDICTED PARAMETERS #
        f0_pred = get_pred(speaker, "f0", w_features)
        ap_pred = get_pred(speaker, "ap", w_features)
        mcep_pred = get_pred(speaker, "mcep", w_features_mcep)

        # RESTORE DIMENSIONALITY FOR AP AND SP #

        ap_restored = pw.decode_aperiodicity(ap_pred, fs, fft)
        sp_restored = pw.decode_spectral_envelope(mcep_pred, fs, fft)

        # RESHAPE F0 #
        f0_pred = np.reshape(f0_pred, len(f0_pred))

        # SPLIT SOURCE FILE AND EXTENSION FOR SAVING CONVERTED VERSION #
        filename, extension = os.path.splitext(file)

        x_pred = pw.synthesize(f0_pred, sp_restored, ap_restored, fs, 5)

        dir = 'training/{}/pred'.format(speaker)
        CHECK_FOLDER = os.path.isdir(dir)

        ### FOLDER CHECK ###
        if not CHECK_FOLDER:
            os.makedirs(dir)
            print("created folder : ", dir)

        sf.write('{}/{}_converted{}'.format(dir, filename, extension), x_pred, fs)

if __name__ == '__main__':
        args = parser.parse_args()
        main(args)