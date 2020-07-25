import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pdb

from utils import make_dir, save_json, MyDataset


def phase_fold(x, p, seq_len):
    x_copy = x.copy()
    t = x[:seq_len, 0]
    mag = x[:seq_len, 1]
    err = x[:seq_len, 2]    
    t = t - t[0]
    t = (t % p) / p
    index = np.argsort(t)
    t = t[index]
    mag = mag[index]
    err = err[index]
    x_copy[:seq_len, 0] = t
    x_copy[:seq_len, 1] = mag
    x_copy[:seq_len, 2] = err
    return x_copy


class LightCurveDataset(object):
    def __init__(self, name, fold, bs, device="cpu", val_size=0.2, eps=1e-10, eval=True):
        self.name = name
        self.fold = fold
        self.eps = eps
        self.val_size = val_size
        self.bs = bs
        self.device = device

        self.data_path = "processed_data/{}".format(name)
        make_dir("processed_data")
        make_dir(self.data_path)
        
        if not eval:
            self.load_data()
            self.fold_light_curves()
            self.normalize()
            self.compute_dt()
            self.time_series_features()
            self.train_val_split()
            self.save_processed_data()
        else:
            self.load_processed_data()
            self.compute_seq_len()

        self.define_datasets()

    def load_data(self):
        self.x = np.load("../datasets/{}/light_curves.npy".format(self.name))
        self.metadata = pd.read_csv("../datasets/{}/metadata.csv".format(self.name))
        return

    def fold_light_curves(self):
        self.x_folded = self.x.copy()
        for i in tqdm(range(len(self.x))):
            dfi = self.metadata.loc[i]
            self.x_folded[i] = phase_fold(self.x[i], dfi["p"], int(dfi["seq_len"]))
        return

    def normalize(self):
        mask = self.x[:, :, 2] != 0
        means = (self.metadata["mean"].values)[:, np.newaxis]
        stds = (self.metadata["std"].values)[:, np.newaxis]
        self.x[:, :, 1] = (self.x[:, :, 1] - means) / (stds + self.eps) * mask
        self.x[:, :, 2] = self.x[:, :, 2] / (stds + self.eps) * mask

        self.x_folded[:, :, 1] = (self.x_folded[:, :, 1] - means) / (stds + self.eps) * mask
        self.x_folded[:, :, 2] = self.x_folded[:, :, 2] / (stds + self.eps) * mask
        return

    def compute_dt(self):
        mask = self.x[:, :, 2] != 0
        self.x[:, 1:, 0] = self.x[:, 1:, 0] - self.x[:, :-1, 0]
        self.x[:, 0, 0] = 0
        self.x[:, :, 0] *= mask

        self.x_folded[:, 1:, 0] = self.x_folded[:, 1:, 0] - self.x_folded[:, :-1, 0]
        self.x_folded[:, 0, 0] = 0
        self.x_folded[:, :, 0] *= mask
        return

    def time_series_features(self):
        self.m = self.metadata["mean"].values
        self.s = self.metadata["std"].values
        self.p = self.metadata["p"].values
        self.p = np.log10(self.p)
        self.seq_len = self.metadata["seq_len"].values
        return

    def train_val_split(self):
        index = np.arange(len(self.x))
        lab2idx = {lab: i for i, lab in enumerate(self.metadata["label"].unique())}
        save_json(lab2idx, "{}/lab2idx.json".format(self.data_path))
        self.y  = [lab2idx[lab] for lab in self.metadata["label"].values]
        self.y = np.array(self.y)      
        train_idx, val_idx = train_test_split(index, test_size=self.val_size, stratify=self.y, shuffle=True)

        self.x_train = self.x[train_idx]
        self.x_train_folded = self.x_folded[train_idx]
        self.y_train = self.y[train_idx]
        self.m_train = self.m[train_idx]
        self.s_train = self.s[train_idx]
        self.p_train = self.p[train_idx]
        self.seq_len_train = self.seq_len[train_idx]

        self.x_val = self.x[val_idx]
        self.x_val_folded = self.x_folded[val_idx]
        self.y_val = self.y[val_idx]
        self.m_val = self.m[val_idx]
        self.s_val = self.s[val_idx]
        self.p_val = self.p[val_idx]
        self.seq_len_val = self.seq_len[val_idx]
        return

    def save_processed_data(self):
        np.save("{}/x_train.npy".format(self.data_path), self.x_train)
        np.save("{}/x_train_folded.npy".format(self.data_path), self.x_train_folded)
        np.save("{}/y_train.npy".format(self.data_path), self.y_train)
        np.save("{}/m_train.npy".format(self.data_path), self.m_train)
        np.save("{}/s_train.npy".format(self.data_path), self.s_train)
        np.save("{}/p_train.npy".format(self.data_path), self.p_train)
        
        np.save("{}/x_val.npy".format(self.data_path), self.x_val)
        np.save("{}/x_val_folded.npy".format(self.data_path), self.x_val_folded)
        np.save("{}/y_val.npy".format(self.data_path), self.y_val)
        np.save("{}/m_val.npy".format(self.data_path), self.m_val)
        np.save("{}/s_val.npy".format(self.data_path), self.s_val)
        np.save("{}/p_val.npy".format(self.data_path), self.p_val)
        return

    def load_processed_data(self):
        if self.fold:
            self.x_train = np.load("{}/x_train_folded.npy".format(self.data_path))
            self.p_train = np.load("{}/p_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val_folded.npy".format(self.data_path))
            self.p_val = np.load("{}/p_val.npy".format(self.data_path))
        else:
            self.x_train = np.load("{}/x_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val.npy".format(self.data_path))

        self.y_train = np.load("{}/y_train.npy".format(self.data_path))
        self.m_train = np.load("{}/m_train.npy".format(self.data_path))
        self.s_train = np.load("{}/s_train.npy".format(self.data_path))
        
        self.y_val = np.load("{}/y_val.npy".format(self.data_path))
        self.m_val = np.load("{}/m_val.npy".format(self.data_path))
        self.s_val = np.load("{}/s_val.npy".format(self.data_path))
        return

    def define_datasets(self):
        self.train_dataset = MyDataset(self.x_train, self.y_train, self.m_train, self.s_train, self.seq_len_train, device=self.device)
        self.val_dataset = MyDataset(self.x_val, self.y_val, self.m_val, self.s_val, self.seq_len_val, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=True)
        return

    def compute_seq_len(self):
        self.seq_len_train = (self.x_train[:, :, 2] != 0).sum(axis=1)
        self.seq_len_val = (self.x_val[:, :, 2] != 0).sum(axis=1)
        return


if __name__ == "__main__":
    dataset = LightCurveDataset("linear", bs=256, fold=True, eval=True)
    plt.errorbar(np.cumsum(dataset.x_train[0, :196, 0]), dataset.x_train[0, :196, 1], dataset.x_train[0, :196, 2], fmt=".")
    plt.show()
