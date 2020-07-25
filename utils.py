import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import pdb


def distances(x, x_pred):
    mask = (x[:, :, 2] != 0) * 1.
    x = x[:, :, 1]
    x_norm = (x.pow(2) * mask).sum(1, keepdim=True)
    x_pred_norm = (x_pred.pow(2) * mask).sum(1, keepdim=True)
    euclidean_distance = ((x_pred - x).pow(2) * mask).sum(dim=1, keepdim=True) / x_norm
    cosine_distance = (x * x_pred * mask).sum(1, keepdim=True) / x_norm / x_pred_norm
    return euclidean_distance, cosine_distance


def plot_confusion_matrix(cm, labels, title, savename, normalize=False, dpi=200):
    nc = len(labels)
    plt.clf()
    fig, ax = plt.subplots()
    if normalize:
        cm = cm / cm.sum(1)
        vmax = 1
        my_format = "{:.2f}"
    else:
        vmax = cm.max()
        my_format = "{:.0f}"
    thr = vmax * 0.5
    im = ax.imshow(cm, cmap="YlGn", aspect="equal", vmin=0, vmax=vmax)
    for i in range(nc):
        for j in range(nc):
            if cm[i, j] > thr:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, my_format.format(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xticks(np.arange(nc), labels, rotation=45)
    plt.yticks(np.arange(nc), labels)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi)
    return


def plot_loss(loss, savename, dpi=200):
    epochs = np.arange(len(loss))
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(epochs, loss[:, 0, 0], color="navy", label="train")
    ax[0].plot(epochs, loss[:, 1, 0], color="red", label="val")
    ax[0].set_ylabel("total loss")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(epochs, loss[:, 0, 1], color="navy", label="train")
    ax[1].plot(epochs, loss[:, 1, 1], color="red", label="val")
    ax[1].set_ylabel("reconstruction")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(epochs, loss[:, 0, 2], color="navy", label="train")
    ax[2].plot(epochs, loss[:, 1, 2], color="red", label="val")
    ax[2].set_ylabel("cross entropy")
    ax[2].set_xlabel("epochs")
    ax[2].legend()
    ax[2].grid()
    
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi)
    return


def save_json(data, savename):
    with open(savename, "w") as fp:
        json.dump(data, fp)


def load_json(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


class WMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-10):
        super(WMSELoss, self).__init__()
        self.eps = eps

    def forward(self, x, x_pred, seq_len):
        # x_pred -> (mu, logvar)
        mask = (x[:, :, 2] != 0) * 1.
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse


class MyDataset(Dataset):
    def __init__(self, x, y, m, s, z, device="cpu"):
        self.n, _, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.m = torch.tensor(m, dtype=torch.float, device=device)
        self.s = torch.tensor(s, dtype=torch.float, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.s[index], self.z[index]

    def __len__(self):
        return self.n


def seed_everything(seed=1234):
    """
    Author: Benjamin Minixhofer
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def count_parameters(model):
    # TODO: add docstring
    """
    Parameters
    ----------
    model
    Returns
    -------
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    