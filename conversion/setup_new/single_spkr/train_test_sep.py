import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--speaker", required=True, type=str)

def main(args):
    speaker = args.speaker

    folder = 'dataset/{}_prep/padded'.format(speaker)


    for dir in ["{}/test".format(folder),
                "{}/train".format(folder),
                "{}/True_test".format(folder)]:
        CHECK_FOLDER = os.path.isdir(dir)

        ### FOLDER CHECK ###
        if not CHECK_FOLDER:
            os.makedirs(dir)
            print("created folder : ", dir)

    files = os.listdir('dataset/{}_prep/padded'.format(speaker))
    utt_counter = 0
    utterances = (len(files)) - 3 ## Number of files minus the created folders

    for file in files:
        utt_counter = utt_counter + 1

        full_path = os.path.join(folder, file)
        #print(full_path)
        if os.path.isdir(full_path):
            continue

        if utt_counter <= ((utterances - 10) * 0.9): # TRAIN
            os.rename(full_path, "dataset/{}_prep/padded/train/{}".format(speaker, file))
        elif utt_counter <= ((utterances - 10)): # TEST
            os.rename(full_path, "dataset/{}_prep/padded/test/{}".format(speaker, file))
        else: #TRUETEST
            os.rename(full_path, "dataset/{}_prep/padded/True_test/{}".format(speaker, file))



if __name__ == '__main__':
        args = parser.parse_args()
        main(args)