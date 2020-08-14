import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import pdb


def compute_params(z, logits):
    # source https://github.com/danieltan07/dagmm/blob/master/model.py
    softmax = torch.nn.Softmax(dim=1)
    gamma = softmax(logits)

    N = gamma.size(0)
    # K
    sum_gamma = torch.sum(gamma, dim=0)

    # K
    phi = sum_gamma / N

    # K x D
    mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
    # z = N x D
    # mu = K x D
    # gamma N x K

    # z_mu = N x K x D
    z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

    # z_mu_outer = N x K x D x D
    z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

    # K x D x D
    cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

    return phi, mu, cov


def compute_energy(z, phi=None, mu=None, cov=None, logits=None):
    # source https://github.com/danieltan07/dagmm/blob/master/model.py
    if phi is None or mu is None or cov is None:
        phi, mu, cov = compute_params(z, logits)

    k, D, _ = cov.size()

    z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

    eps = 1e-12
    cte = D * np.log(2 * np.pi)
    eye = (torch.eye(D, device=cov.device) * eps).unsqueeze(0)
    eye = eye.repeat(k, 1, 1)
    cov = cov + eye
    cov_inverse = torch.inverse(cov)
    det_cov = (0.5 * (cte + torch.logdet(cov)))
    # if torch.isnan(det_cov.sum()):
        # det_cov = torch.ones_like(det_cov) * eps
    # det_cov = 0.5 * torch.logdet(cov)
    if torch.isnan(det_cov.sum()):
        mask = det_cov != det_cov
        det_cov[mask] = det_cov[~ mask].mean()

    # N x K
    exp_term_tmp = - 0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
    log_norm_prob_dens = exp_term_tmp - det_cov.unsqueeze(0)
    # for stability (logsumexp)
    # max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
    # exp_term = torch.exp(exp_term_tmp - max_val)
    # max_val, _ = exp_term_tmp.clamp(min=0).max(dim=1, keepdim=True)
    # exp_term = torch.exp(exp_term_tmp - max_val) / torch.sqrt(det_cov).unsqueeze(0)
    # pdb.set_trace()
    exp_term = log_norm_prob_dens.exp()

    arg = torch.sum(phi.unsqueeze(0) * exp_term, dim=1)
    # sample_energy = - max_val.squeeze() - torch.log(arg + eps)
    sample_energy = 1 / (arg + eps)
    sample_energy = sample_energy.clamp(max=1e4)
    # sample_energy = - torch.log(arg + eps)
    
    if torch.isnan(sample_energy.mean()):
        pdb.set_trace()

    return sample_energy


def distances(x, x_pred):
    mask = (x[:, :, 0] != 0) * 1.
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
        cm = cm / cm.sum(1)[:, np.newaxis]
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
    fig, ax = plt.subplots(4, 1, figsize=(20, 10))

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
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(epochs, loss[:, 0, 3], color="navy", label="train")
    ax[3].plot(epochs, loss[:, 1, 3], color="red", label="val")
    ax[3].set_ylabel("gmm energy")
    ax[3].set_xlabel("epoch")
    ax[3].legend()
    ax[3].grid()
    
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
    def __init__(self, eps=1e-10, nc=3):
        super(WMSELoss, self).__init__()
        self.eps = eps
        self.loss_fun = self.wmse_3c if nc == 3 else self.wmse_2c

    def wmse_3c(self, x, x_pred, seq_len):
        mask = (x[:, :, 2] != 0) * 1.
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def wmse_2c(self, x, x_pred, seq_len):
        mask = (x[:, :, 0] != 0) * 1.
        wmse = ((x_pred - x[:, :, 1]).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def forward(self, x, x_pred, seq_len):
        wmse = self.loss_fun(x, x_pred, seq_len)
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


class MyFoldedDataset(Dataset):
    def __init__(self, x, y, m, s, p, z, device="cpu"):
        self.n, _, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.m = torch.tensor(m, dtype=torch.float, device=device)
        self.s = torch.tensor(s, dtype=torch.float, device=device)
        self.p = torch.tensor(p, dtype=torch.float, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.s[index], self.p[index], self.z[index]

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
    