import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd


def plot_loss(train_loss, val_loss, savename):
    fs = 5
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    fig, ax = plt.subplots(5, figsize=(1.618 * fs * 2, fs * 5))

    axi = ax[0]
    axi.plot(train_loss[:, 0], label="train")
    axi.plot(val_loss[:, 0], label="val")
    axi.legend()
    axi.set_title("total")

    axi = ax[1]
    axi.plot(train_loss[:, 1], label="train")
    axi.plot(val_loss[:, 1], label="val")
    axi.legend()
    axi.set_title("reconstruction")

    axi = ax[2]
    axi.plot(train_loss[:, 2], label="train")
    axi.plot(val_loss[:, 2], label="val")
    axi.legend()
    axi.set_title("cross entropy")

    axi = ax[3]
    axi.plot(train_loss[:, 3], label="train")
    axi.plot(val_loss[:, 3], label="val")
    axi.legend()
    axi.set_title("gmm")

    axi = ax[4]
    axi.plot(train_loss[:, 4], label="train")
    axi.plot(val_loss[:, 4], label="val")
    axi.legend()
    axi.set_title("singularity")
    
    plt.tight_layout()
    plt.savefig(savename, dpi=400)
    return


def pad_sequence_with_lengths(data):
    x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=0., batch_first=True)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    seq_len = torch.tensor([d[2] for d in data], dtype=torch.long)
    return x, y, seq_len


def get_datasets(x_train, x_val, x_test, y_train, y_val, y_test, device):
    return MyDataset(x_train, y_train, device), MyDataset(x_val, y_val, device), MyDataset(x_test, y_test, device)


def get_data_loaders(dataset, batch_size, device, oc=None):
    if dataset == "asas_sn":
        pass
    elif "ztf" in dataset:
        x_train, x_val, x_test, y_train, y_val, y_test = get_ztf_data(dataset)
    elif dataset == "asas":
        x_train, x_val, x_test, y_train, y_val, y_test = get_asas_data(oc)
    elif dataset == "linear":
        x_train, x_val, x_test, y_train, y_val, y_test = get_linear_data(oc)
    trainset, valset, testset = get_datasets(x_train, x_val, x_test, y_train, y_val, y_test, device)
    trainloader = DataLoader(trainset, batch_size=int(batch_size), shuffle=True, collate_fn=pad_sequence_with_lengths)
    valloader = DataLoader(valset, batch_size=int(batch_size), shuffle=True, collate_fn=pad_sequence_with_lengths)
    testloader = DataLoader(testset, batch_size=int(batch_size), shuffle=False, collate_fn=pad_sequence_with_lengths)
    return trainloader, valloader, testloader


def get_linear_data(oc):
    df = pd.read_csv("../datasets/linear/metadata_split_1.csv")
    data = np.load("../datasets/linear/light_curves_pf.npz", allow_pickle=True)["light_curves"]

    mask = df["split"] == "train"
    x_train = [dat for dat, ma in zip(data, mask) if ma]
    y_train = df["target"][mask].values

    mask = df["split"] == "val"
    x_val = [dat for dat, ma in zip(data, mask) if ma]
    y_val = df["target"][mask].values

    mask = df["split"] == "test"
    x_test = [dat for dat, ma in zip(data, mask) if ma]
    y_test = df["target"][mask].values
    normalize_mag(x_train)
    normalize_mag(x_test)
    normalize_mag(x_val)
    return x_train, x_val, x_test, y_train, y_val, y_test


def get_asas_data(oc):
    df = pd.read_csv("../datasets/asas/metadata_split_1.csv")
    data = np.load("../datasets/asas/light_curves_pf_pro1.npz", allow_pickle=True)["light_curves"]
    normalize_mag(data)

    # data splitting
    mask = df["split"] == "train"
    x_train = [dat for dat, ma in zip(data, mask) if ma]
    y_train = df["target"][mask].values
    mask = df["split"] == "val"
    x_val = [dat for dat, ma in zip(data, mask) if ma]
    y_val = df["target"][mask].values
    mask = df["split"] == "test"
    x_test = [dat for dat, ma in zip(data, mask) if ma]
    y_test = df["target"][mask].values

    # od data removal
    mask = y_train != oc
    x_train = [dat for dat, ma in zip(data, mask) if ma]
    y_train = y_train[mask]
    mask = y_val != oc
    x_val = [dat for dat, ma in zip(data, mask) if ma]
    y_val = y_val[mask]
    return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_mag(x):
    for xi in x:
        m, s = xi[:, 1].mean(), xi[:, 1].std()
        xi[:, 1] = (xi[:, 1] - m) / s
        xi[:, 2] = xi[:, 2] / s
    return


def remove_time_offset(x):
    for xi in x:
        xi[:, 0] = xi[:, 0] -  xi[:, 0].min()
    return


def get_ztf_data(dataset_name):
    family = dataset_name.split('_')[1]
    data = np.load("../datasets/ztf/cl/{}/light_curves.npz".format(family), allow_pickle=True)["light_curves"]
    df = pd.read_csv("../datasets/ztf/cl/{}/metadata_pro1.csv".format(family))
    remove_time_offset(data)
    normalize_mag(data)
    mask = df["split2"] == "train"
    x_train = [dat for dat, ma in zip(data, mask) if ma]
    y_train = df["label"][mask].values
    mask = df["split2"] == "val"
    x_val = [dat for dat, ma in zip(data, mask) if ma]
    y_val = df["label"][mask].values
    mask = df["split2"] == "test"
    x_test = [dat for dat, ma in zip(data, mask) if ma]
    y_test = df["label"][mask].values
    return x_train, x_val, x_test, y_train, y_val, y_test


def od_metrics(scores, y, split=False, n_splits=None):
    aucpr = list()
    aucroc = list()
    scores_in = scores[y == 1]
    scores_out = scores[y == 0]
    if split:
        _, counts = np.unique(y, return_counts=True)
        if counts[0] < counts[1]:
            new_y = np.concatenate((np.ones(counts[0]), np.zeros(counts[0])))            
            for i in range(n_splits):
                new_scores = np.random.choice(scores_in, counts[0], replace=False)
                cat_scores = np.concatenate((new_scores, scores_out))
                precision, recall, _ = precision_recall_curve(new_y, cat_scores, pos_label=0)
                fpr, tpr, _ = roc_curve(new_y, cat_scores, pos_label=0)
                aucpr.append(auc(recall, precision))
                aucroc.append(auc(fpr, tpr))
        else:
            new_y = np.concatenate((np.ones(counts[1]), np.zeros(counts[1])))
            for i in range(n_splits):
                new_scores = np.random.choice(scores_out, counts[1], replace=False)
                cat_scores = np.concatenate((scores_in, new_scores))
                precision, recall, _ = precision_recall_curve(new_y, cat_scores, pos_label=0)
                fpr, tpr, _ = roc_curve(new_y, cat_scores, pos_label=0)
                aucpr.append(auc(recall, precision))
                aucroc.append(auc(fpr, tpr))
        return aucpr, None, None, aucroc, None, None
    else:
        precision, recall, _ = precision_recall_curve(y, scores, pos_label=0)
        fpr, tpr, _ = roc_curve(y, scores, pos_label=0)
        aucpr = auc(recall, precision)
        aucroc = auc(fpr, tpr)
        return aucpr, precision, recall, aucroc, fpr, tpr


def save_yaml(data, savename):
    with open(savename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    return


def load_yaml(filename):
    with open(filename, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def pad_seq(x, device):
    lc = [xi[0] for xi in x]
    sequences = torch.nn.utils.rnn.pad_sequence(lc, batch_first=True)
    y = torch.tensor([xi[1] for xi in x], dtype=torch.long, device=device)
    m = torch.tensor([xi[2] for xi in x], dtype=torch.float, device=device)
    s = torch.tensor([xi[3] for xi in x], dtype=torch.float, device=device)
    seq_len = torch.tensor([xi[4] for xi in x], dtype=torch.float, device=device)
    return sequences, y, m, s, seq_len


def plot_single_confusion_matrix(cm, labels, title, savename, normalize, ni=None):
    nc = len(labels)
    if not ni:
        ni = nc
    if normalize:        
        total = cm.sum(axis=1, keepdims=True)
        vmax = 1
        template = "{:.2f}"
    else:
        vmax = cm.max()
        template = "{:.0f}"
    thr = vmax * 0.5
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="YlGn", aspect="equal", vmin=0, vmax=vmax)
    for i in range(nc):
        for j in range(ni):
            if cm[i, j] > thr:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, template.format(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xticks(np.arange(ni), labels[:ni], rotation=45)
    plt.yticks(np.arange(nc), labels)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
    return


def plot_confusion_matrix(cms, labels, title, savename, normalize, ni=None):
    nc = len(labels)
    if not ni:
        ni = nc    
    cms_mean = cms.mean(axis=0)
    cms_std = cms.std(axis=0)
    if normalize:        
        total = cms[0].sum(axis=1, keepdims=True)
        cms_mean = cms_mean / total
        cms_std = cms_std / total
        vmax = 1
        template = "{:.2f} \n +- {:.2f}"
    else:
        vmax = cms_mean.max()
        template = "{:.0f} \n +- {:.0f}"
    thr = vmax * 0.5
    fig, ax = plt.subplots()
    im = ax.imshow(cms_mean, cmap="YlGn", aspect="equal", vmin=0, vmax=vmax)
    for i in range(nc):
        for j in range(ni):
            if cms_mean[i, j] > thr:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, template.format(cms_mean[i, j], cms_std[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xticks(np.arange(ni), labels[:ni], rotation=45)
    plt.yticks(np.arange(nc), labels)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
    return


def calculate_seq_len(x):
    sl = (x[:, :, 2] != 0).sum(1)
    return sl


def compute_energy(z, phi=None, mu=None, cov=None, size_average=True):
    k, D, _ = cov.size()
    z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

    eps = 1e-12
    cte = D * np.log(2 * np.pi)
    eye = (torch.eye(D, device=cov.device) * eps).unsqueeze(0)
    eye = eye.repeat(k, 1, 1)
    cov = cov + eye
    cov_inverse = torch.inverse(cov)
    log_det_cov = torch.logdet(cov)
    cte_2 = (0.5 * (log_det_cov + cte)).exp()
    cte_2 = torch.where(torch.isnan(cte_2), torch.ones_like(cte_2) * eps, cte_2)
    # cte_2 = torch.nan_to_num(input, nan=eps)  # only for pytorch > 1.8
    # N x K
    exp_arg = - 0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
    # for stability (logsumexp)
    max_val, _ = exp_arg.max(dim=1, keepdim=True)
    exp_term = torch.exp(exp_arg - max_val)
    density = exp_term / cte_2.unsqueeze(0)
    sample_energy = - max_val.squeeze() - (density * phi.unsqueeze(0)).sum(-1).log()
    if size_average:
        sample_energy = torch.mean(sample_energy)
    return sample_energy


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
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=1) / seq_len
        return wmse

    def wmse_2c(self, x, x_pred, seq_len):
        mask = (x[:, :, 0] != 0) * 1.
        wmse = ((x_pred - x[:, :, 1]).pow(2) * mask).sum(dim=1) / seq_len
        return wmse

    def forward(self, x, x_pred, seq_len):
        wmse = self.loss_fun(x, x_pred, seq_len)
        return wmse


class MyFoldedDataset(Dataset):
    def __init__(self, x, y, m, s, p, sl, device="cpu"):
        self.n, _, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.m = torch.tensor(m, dtype=torch.float, device=device)
        self.s = torch.tensor(s, dtype=torch.float, device=device)
        self.p = torch.tensor(p, dtype=torch.float, device=device)
        self.sl = torch.tensor(sl, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.s[index], self.p[index], self.sl[index]

    def __len__(self):
        return self.n


class MyDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.n = len(x)
        _, self.ndim = x[0].shape
        self.sl = [len(xi) for xi in x]
        self.x = [torch.tensor(xi, dtype=torch.float, device=device) for xi in x]
        self.y = y
        self.device = device

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sl[index]

    def __len__(self):
        return self.n


class MyFeatureDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.n, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


def seed_everything(seed=1234):
    # https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # True before
        torch.backends.cudnn.benchmark = True  # False before
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    