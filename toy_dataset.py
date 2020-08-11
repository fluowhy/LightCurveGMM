import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pdb
from sklearn.model_selection import train_test_split


def read_data(seq_len=32, nclasses=3, uneven=False):
    if uneven:
        x_train = np.load("../datasets/toy/x_uneven_True_train_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        x_test = np.load("../datasets/toy/x_uneven_True_test_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        y_train = np.load("../datasets/toy/y_uneven_True_train_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        y_test = np.load("../datasets/toy/y_uneven_True_test_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
    else:
        x_train = np.load("../datasets/toy/x_train_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        x_test = np.load("../datasets/toy/x_test_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        y_train = np.load("../datasets/toy/y_train_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
        y_test = np.load("../datasets/toy/y_test_nsamples_{}_nclasses_{}.npy".format(seq_len, nclasses))
    return x_train, x_test, y_train, y_test


def filter_inlier_data(x_train, y_train, outlier_class):
    # filter by inlier class for training
    inlier_filter = y_train != outlier_class
    x_train = x_train[inlier_filter]
    y_train = y_train[inlier_filter]
    return x_train, y_train


def change_label(y_train, y_test, outlier_class):
    new2old = np.unique(y_train)
    old2new = {lab: i for i, lab in enumerate(new2old)}
    old2new[outlier_class] = - 99
    y_train = np.array([old2new[lab] for lab in y_train])
    mask = y_test == outlier_class
    y_test = np.array([old2new[lab] for lab in y_test])
    return y_train, y_test


def transform_data(x):
    aux_y_0 = np.zeros(len(x))

    aux_x_1 = amplitude_inverse_surrogates(x)
    aux_y_1 = np.full(len(aux_x_1), 1)
    aux_x_2 = time_inverse_surrogates(x)
    aux_y_2 = np.full(len(aux_x_2), 2)
    aux_x_3 = block_surrogates(x)
    aux_y_3 = np.full(len(aux_x_3), 3)
    aux_x_4 = shuffle_surrogates(x)
    aux_y_4 = np.full(len(aux_x_4), 4)
    x = np.concatenate((x, aux_x_1, aux_x_2, aux_x_3, aux_x_4))
    y = np.concatenate((aux_y_0, aux_y_1, aux_y_2, aux_y_3, aux_y_4))
    return x, y


def random_shuffle(x, y):
    index = np.arange(len(x))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    return x, y


def validation_split(x_train, y_train, val_size):
    index = np.arange(len(x_train))
    train_idx, val_idx = train_test_split(index, test_size=val_size, stratify=y_train)
    x_val = x_train[val_idx]
    x_train = x_train[train_idx]
    y_val = y_train[val_idx]
    y_train = y_train[train_idx]
    return x_train, x_val, y_train, y_val


def normalize_light_curves(x, eps=1e-10):
    seq_len = (x[:, :, 0] != 0).sum(axis=-1) + 1
    means = np.zeros(len(x)) 
    stds = np.zeros(len(x))
    for i in range(len(x)):
        xi = x[i, :seq_len[i], 1]
        mean = xi.mean()
        std = xi.std()
        x[i, :seq_len[i], 1] = (x[i, :seq_len[i], 1] - mean) / (std + eps)
        if x.shape[2] == 3:
            x[i, :seq_len[i], 2] = x[i, :seq_len[i], 2] / (std + eps)
        means[i] = mean
        stds[i] = std
    means = means[:, np.newaxis]
    stds = stds[:, np.newaxis]
    return x, means, stds


def time_norm(x):
    tmin = x[:, 0, 0][:, np.newaxis]
    x[:, :, 0] = x[:, :, 0] - tmin
    x[:, 1:, 0] = x[:, 1:, 0] - x[:, :-1, 0]    #tmax = np.max(x[:, :, 0], axis=1)[:, np.newaxis]
    # x[:, :, 0] = x[:, :, 0] / (tmax + 1e-10)
    return x


class MyAugDataset(Dataset):
    def __init__(self, x, y, m, s, z, device="cpu"):
        self.n, _, _ = x.shape
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.m = torch.tensor(m, dtype=torch.float, device=device)
        self.s = torch.tensor(s, dtype=torch.float, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.s[index], self.z[index]

    def __len__(self):
        return self.n


class ToyDataset(object):
    def __init__(self, args, val_size=0.1, sl=20):
        self.args = args
        self.val_size = val_size
        self.sl = sl
        self.bs = args.bs        
        self.device = args.d        
        self.nclasses = 5
        self.eps = 1e-10
        self.lab2idx = {"sawtooth": 0, "sine": 1, "gaussian": 2, "square": 3, "laplacian": 4}
        self.process()

    def process(self):        
        self.x_train, self.x_test, self.y_train, self.y_test = read_data(seq_len=self.sl, nclasses=5)

        # train outliers removal and put in test
        _, counts = np.unique(self.y_test, return_counts=True)
        for i in [3, 4]:
            mask = self.y_train == i
            sel_x = self.x_train[mask][:int(counts[0] * (3 / 2 - 1))]
            sel_y = self.y_train[mask][:int(counts[0] * (3 / 2 - 1))]
            self.x_test = np.concatenate((self.x_test, sel_x))
            self.y_test = np.concatenate((self.y_test, sel_y))
            self.x_train = self.x_train[~ mask]
            self.y_train = self.y_train[~ mask]
        
        # old processing
        self.x_train, self.y_train = random_shuffle(self.x_train, self.y_train)
        self.x_train, self.x_val, self.y_train, self.y_val = validation_split(self.x_train, self.y_train, self.val_size)

        # time series normalization
        self.x_train, self.m_train, self.s_train = normalize_light_curves(self.x_train)
        self.x_val, self.m_val, self.s_val = normalize_light_curves(self.x_val)
        self.x_test, self.m_test, self.s_test = normalize_light_curves(self.x_test)

        # time normalization
        self.x_train = time_norm(self.x_train)
        self.x_test = time_norm(self.x_test)
        self.x_val = time_norm(self.x_val)

        # sequence length calculation
        self.seq_len_train = self.calculate_seq_len(self.x_train)
        self.seq_len_val = self.calculate_seq_len(self.x_val)

        # dataset and dataloader definition
        self.train_dataset = MyAugDataset(self.x_train, self.y_train, self.m_train, self.s_train, self.seq_len_train, device=self.device)
        self.val_dataset = MyAugDataset(self.x_val, self.y_val, self.m_val, self.s_val, self.seq_len_val, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False)
        return

    def calculate_seq_len(self, x):
        return (x[:, :, 0] != 0).sum(1) + 1
