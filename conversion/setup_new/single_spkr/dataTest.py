from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os


class TrainF0Dataset(Dataset):

    def __init__(self, mode, speaker):

        self.mode = mode

        self.f0 = []
        self.w_features = []
        self.utterances = []
        self.length = []

        if mode == "train":
            folder = 'dataset/{}_prep/padded/train'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/train'.format(speaker))

        else:
            folder = 'dataset/{}_prep/padded/test'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/test'.format(speaker))

        for file in files:

            utterance, suffix = file.split('_')
            full_path = os.path.join(folder, file)

            data = np.load(full_path)
            f0 = data['f0']
            w_features = data['w_features']
            length = data['len']
            w_features.astype(float)
            f0.astype(float)

            length = int(length)

            self.w_features.append(w_features)
            self.f0.append(f0)
            self.utterances.append(utterance)
            self.length.append(length)

        if isinstance(self.length, list):
            lengths = np.array(self.length, dtype=np.int32)[:, None]
            self.length = lengths

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):

        f = np.expand_dims(self.f0[idx], 1)
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #f = scaler.fit_transform(f)

        l = torch.from_numpy(self.length[idx])
        return self.w_features[idx], f, l


class TrainAPDataset(Dataset):

    def __init__(self, mode, speaker):

        self.mode = mode

        self.ap = []
        self.w_features = []
        self.utterances = []
        self.length = []

        if mode == "train":
            folder = 'dataset/{}_prep/padded/train'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/train'.format(speaker))

        else:
            folder = 'dataset/{}_prep/padded/test'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/test'.format(speaker))

        for file in files:

            utterance, suffix = file.split('_')
            full_path = os.path.join(folder, file)

            data = np.load(full_path)
            ap = data['ap']
            w_features = data['w_features']
            length = data['len']
            w_features.astype(float)
            ap.astype(float)

            self.w_features.append(w_features)
            self.ap.append(ap)
            self.utterances.append(utterance)
            self.length.append(length)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #f = scaler.fit_transform(f)

        l = torch.from_numpy(self.length[idx])
        return self.w_features[idx], self.ap[idx], l


class TrainMCEPDataset(Dataset):

    def __init__(self, mode, speaker):

        self.mode = mode

        self.mcep = []
        self.w_features = []
        self.utterances = []
        self.length = []

        if mode == "train":
            folder = 'dataset/{}_prep/padded/train'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/train'.format(speaker))

        else:
            folder = 'dataset/{}_prep/padded/test'.format(speaker)
            files = os.listdir('dataset/{}_prep/padded/test'.format(speaker))

        for file in files:

            utterance, suffix = file.split('_')
            full_path = os.path.join(folder, file)

            data = np.load(full_path)
            mcep = data['mcep']
            w_features = data['w_features']
            length = data['len']
            w_features.astype(float)
            mcep.astype(float)

            self.w_features.append(w_features)
            self.mcep.append(mcep)
            self.utterances.append(utterance)
            self.length.append(length)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):

        l = torch.from_numpy(self.length[idx])
        return self.w_features[idx], self.mcep[idx], l

