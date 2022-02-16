from dataTest import TrainF0Dataset, TrainAPDataset, TrainMCEPDataset
from torch import nn
from tqdm import tnrange, tqdm, notebook
import torch
import lstm
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


num_hidden = 256
num_hidden_layers = 2
batch_size = 5
n_workers = 8
epochs = 750
learning_rate = 0.0001
weight_decay = 1e-6

w_f_dim = 150
mcep_dim = 50
f0_dim = 1
ap_dim = 5

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--speaker", required=True, type=str)

def train_lstm(model, optimizer, train_loader, test_loader, speaker, ty):

    model.cuda()

    dataset_loaders = {"train": train_loader, "test": test_loader}

    loss_func = nn.MSELoss()
    model.train()
    print("Start utterance-wise training...")
    loss_history = {"train": [], "test": []}

    for dir in ["training/{}".format(speaker),
                "training/{}/backup".format(speaker)]:
        CHECK_FOLDER = os.path.isdir(dir)

        ### FOLDER CHECK ###
        if not CHECK_FOLDER:
            os.makedirs(dir)
            print("created folder : ", dir)
    ## TRAINING LOOP ##

    for epoch in notebook.tnrange(epochs):
        for phase in ["train", "test"]:
            running_loss = 0
            counter = 0
            for x, y, lengths in dataset_loaders[phase]:

                sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()

                x = x.float()
                y = y.float()

                x, y = x[indices], y[indices]
                y = y[:, :sorted_lengths[0]]

                h, c = model.init_hidden(len(sorted_lengths))

                x, y = x.cuda(), y.cuda()
                h, c = h.cuda(), c.cuda()

                optimizer.zero_grad()

                y_hat = model(x, sorted_lengths, h, c)
                loss = loss_func(y_hat, y)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                counter = counter + 1
                print(sorted_lengths)
                print("Speaker: {}, Epoch: {}, Phase: {}, Batch {}".format(speaker, epoch, phase, counter))
                print("Running loss: ", running_loss)
            loss_history[phase].append(running_loss / len(dataset_loaders[phase]))

        ## SAVING AFTER EVERY 20 EPOCHS

        if not epoch % 20:
            torch.save(model.state_dict(), 'training/{}/backup/{}_model_state_{}'.format(speaker, ty, epoch))
            torch.save(model, 'training/{}/backup/{}_model_{}'.format(speaker, ty, epoch))
            np.savez("training/{}/backup/{}_loss_{}".format(speaker, ty, epoch), loss_history=loss_history)
    return loss_history

def main(args):
    speaker = args.speaker
    models = {}
    input_dims = {"f0": f0_dim, "ap": ap_dim, "mcep": mcep_dim}
    for typ in ["f0", "ap", "mcep"]:
        models[typ] = lstm.MyBLSTM(w_f_dim, num_hidden, input_dims[typ], num_layers=num_hidden_layers,
                                   bidirectional=True)
        print("Model for {}\n".format(typ), models[typ])

        ty = typ

        if ty == "f0":
            dataset_train = TrainF0Dataset('train', speaker)
            dataset_test = TrainF0Dataset('test', speaker)
        elif ty == "ap":
            dataset_train = TrainAPDataset('train', speaker)
            dataset_test = TrainAPDataset('test', speaker)
        else:
            dataset_train = TrainMCEPDataset('train', speaker)
            dataset_test = TrainMCEPDataset('test', speaker)

        data_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        data_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        optimizer = optim.Adam(models[ty].parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_history = train_lstm(models[ty], optimizer, data_train, data_test, speaker, typ)

        torch.save(models[ty].state_dict(), 'training/{}/{}_model_state_complete'.format(speaker, typ))
        torch.save(models[ty], 'training/{}/{}_model_complete'.format(speaker, typ))
        np.savez("training/{}/{}_loss_complete".format(speaker, typ), loss_history=loss_history)

        plt.plot(loss_history["train"], linewidth=2, label="Train loss")
        plt.plot(loss_history["test"], linewidth=2, label="Test loss")
        plt.legend(prop={"size": 16})
        plt.savefig('training/{}/{}_loss.png'.format(speaker, typ))
        plt.clf()



if __name__ == '__main__':
        args = parser.parse_args()
        main(args)

