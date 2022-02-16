import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--speaker", required=True, type=str)

def main(args):

    speaker = args.speaker
    folder = 'dataset/{}'.format(speaker)
    files = os.listdir('dataset/{}'.format(speaker))
    max_length = 0

    for file in files:

        utterance, suffix = file.split('_')
        full_path = os.path.join(folder, file)

        utterance = int(utterance)

        data = np.load(full_path)
        f0 = data['f0']
        ap = data['ap']
        sp = data['sp']
        w_features = data['w_features']

        if len(f0) > max_length:
            max_length = len(f0)

        for dir in ["dataset/{}_prep/f0".format(speaker),
                    "dataset/{}_prep/ap".format(speaker),
                    "dataset/{}_prep/mcep".format(speaker),
                    "dataset/{}_prep/w_f".format(speaker)]:
            CHECK_FOLDER = os.path.isdir(dir)

            ### FOLDER CHECK ###
            if not CHECK_FOLDER:
                os.makedirs(dir)
                print("created folder : ", dir)

        np.savez("dataset/{}_prep/f0/{:03d}_f0".format(speaker, utterance), f0=f0)
        np.savez("dataset/{}_prep/ap/{:03d}_ap".format(speaker, utterance), ap=ap)
        np.savez("dataset/{}_prep/mcep/{:03d}_mcep".format(speaker, utterance), mcep=sp)
        np.savez("dataset/{}_prep/w_f/{:03d}_w_f".format(speaker, utterance), w_features=w_features)
        print("basic save done", utterance)

    for file in files:

        utterance, suffix = file.split('_')
        full_path = os.path.join(folder, file)

        utterance = int(utterance)

        data = np.load(full_path)
        f0 = data['f0']
        ap = data['ap']
        sp = data['sp']
        w_features = data['w_features']

        frame_length = len(f0)
        difference = max_length - frame_length

        ## PADDING F0

        # since f0 has 1 value per frame, we can use len(f0)
        if frame_length != max_length:
            dif_vals = difference
            f0_pad_res = np.pad(f0, (0, dif_vals), constant_values=0)

        else:
            f0_pad_res = f0

        ## PADDING AP

        # since AP has 5 values per frame, we multiply difference times 5

        if frame_length != max_length:

            dif_vals = difference * 5
            ap = np.reshape(ap, (-1))
            ap_padded = np.pad(ap, (0, dif_vals), constant_values=0)
            ap_pad_res = np.reshape(ap_padded, (max_length, 5))
            # print(ap_pad_res[max_length - difference - 1], ap_pad_res[max_length - difference])

        else:
            ap_pad_res = ap

        ## PADDING MCEP

        # since MCEP has 26 values per frame, we multiply difference times 26

        if frame_length != max_length:

            dif_vals = difference * 50
            sp = np.reshape(sp, (-1))
            sp_padded = np.pad(sp, (0, dif_vals), constant_values=0)
            sp_pad_res = np.reshape(sp_padded, (max_length, 50))
            # print(ap_pad_res[max_length - difference - 1], ap_pad_res[max_length - difference])

        else:
            sp_pad_res = sp

        ## PADDING W_FEATURES

        # since whispered speech features have 26 * 3 values per frame, we multiply difference times 78

        if frame_length != max_length:

            dif_vals = difference * 150
            w_features = np.reshape(w_features, (-1))
            w_features_padded = np.pad(w_features, (0, dif_vals), constant_values=0)
            w_features_pad_res = np.reshape(w_features_padded, (max_length, 150))
            # print(ap_pad_res[max_length - difference - 1], ap_pad_res[max_length - difference])

        else:
            w_features_pad_res = w_features

        ## PADDING DONE ##

        # print(f0_pad_res.shape, ap_pad_res.shape, sp_pad_res.shape, w_features_pad_res.shape)

        for dir in ["dataset/{}_prep/padded_sep/f0".format(speaker),
                    "dataset/{}_prep/padded_sep/ap".format(speaker),
                    "dataset/{}_prep/padded_sep/mcep".format(speaker),
                    "dataset/{}_prep/padded_sep/w_f".format(speaker),
                    "dataset/{}_prep/padded_sep/lengths".format(speaker),
                    "dataset/{}_prep/padded".format(speaker)]:
            CHECK_FOLDER = os.path.isdir(dir)

            ### FOLDER CHECK ###
            if not CHECK_FOLDER:
                os.makedirs(dir)
                print("created folder : ", dir)

        np.savez("dataset/{}_prep/padded_sep/f0/{:03d}_f0".format(speaker, utterance), f0=f0_pad_res)
        np.savez("dataset/{}_prep/padded_sep/ap/{:03d}_ap".format(speaker, utterance), ap=ap_pad_res)
        np.savez("dataset/{}_prep/padded_sep/mcep/{:03d}_mcep".format(speaker, utterance), mcep=sp_pad_res)
        np.savez("dataset/{}_prep/padded_sep/w_f/{:03d}_w_f".format(speaker, utterance), w_features=w_features_pad_res)
        np.savez("dataset/{}_prep/padded_sep/lengths/{:03d}_len".format(speaker, utterance), len=frame_length)

        np.savez("dataset/{}_prep/padded/{:03d}_pad".format(speaker, utterance), f0=f0_pad_res, ap=ap_pad_res,
                 mcep=sp_pad_res, w_features=w_features_pad_res, len=frame_length)

        print("padded save done", utterance)

if __name__ == '__main__':
        args = parser.parse_args()
        main(args)




