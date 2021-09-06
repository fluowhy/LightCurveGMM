import numpy as np
import torch

from torch.utils.data import DataLoader
from utils import load_json
from utils import MyDataset
from utils import MyFoldedDataset
from utils import calculate_seq_len


def pad_sequence_with_lengths(data):
    x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=0., batch_first=True)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    seq_len = torch.tensor([d[2] for d in data], dtype=torch.float)
    return x, y, seq_len


def read_and_normalize_data(data_path):
    # data
    lab2fam = load_json("../datasets/ztf/family.json")
    lab2idx = load_json("../datasets/ztf/lab2idx.json")
    idx2lab = load_json("../datasets/ztf/idx2lab.json")

    x = np.load(data_path, allow_pickle=True)
    x_train = x["x_train"]
    x_val = x["x_val"]
    y_train = x["y_train"]
    y_val = x["y_val"]
    x_train, _, _ = normalize_light_curves_w_list(x_train)
    x_val, _, _ = normalize_light_curves_w_list(x_val)
    x_train = time_norm_w_list(x_train)
    x_val = time_norm_w_list(x_val)
    x_test = x["x_test"]
    y_test = x["y_test"]
    x_test, _, _ = normalize_light_curves_w_list(x_test)
    x_test = time_norm_w_list(x_test)
    labels = np.unique(y_train)
    lab2newlab = {lab: idx for idx, lab in enumerate(labels)}
    for lab in np.unique(y_test):
        try:
            lab2newlab[lab]
        except KeyError:
            lab2newlab[lab] = len(lab2newlab)
    newlab2lab = [lab for lab in lab2newlab]
    y_train = [lab2newlab[lab] for lab in y_train]
    y_val = [lab2newlab[lab] for lab in y_val]
    y_test = [lab2newlab[lab] for lab in y_test]
    return x_train, x_val, x_test, y_train, y_val, y_test, lab2fam, lab2idx, idx2lab, lab2newlab, newlab2lab


def random_shuffle(x, y, m, s, p=None):
    index = np.arange(len(x))
    np.random.shuffle(index)
    if p is not None:
        return x[index], y[index], m[index], s[index], p[index]
    else:
        return x[index], y[index], m[index], s[index]


def read_data(fold=False):
    if fold:        
        x_train = np.load("../datasets/asas_sn/pf_train.npy")
        x_test = np.load("../datasets/asas_sn/pf_test.npy")
        x_val = np.load("../datasets/asas_sn/pf_val.npy")
        p_train = np.load("../datasets/asas_sn/p_train.npy")
        p_test = np.load("../datasets/asas_sn/p_test.npy")
        p_val = np.load("../datasets/asas_sn/p_val.npy")
    else:
        data = np.load("../datasets/asas_sn/train_data.npz", allow_pickle=True)
    y_train = np.load("../datasets/asas_sn/y_train.npy")
    y_test = np.load("../datasets/asas_sn/y_test.npy")
    y_val = np.load("../datasets/asas_sn/y_val.npy")
    if fold:
        return x_train, x_test, x_val, y_train, y_test, y_val, p_train, p_test, p_val
    else:
        return data["x_train"], data["x_test"], data["x_val"], y_train, y_test, y_val


def time_norm(x):
    for i in range(len(x)):
        x[i, 1:, 0] = x[i, 1:, 0] - x[i, :-1, 0]
        x[i, 0, 0] = 0.    
    return x


def time_norm_w_list(x):
    for i in range(len(x)):
        x[i][1:, 0] = x[i][1:, 0] - x[i][:-1, 0]
        x[i][0, 0] = 0.    
    return x


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


def normalize_light_curves(x, eps=1e-10, minmax=False):
    means = list()
    stds = list()
    for i in range(len(x)):
        mean = x[i, :, 1].mean()
        std = x[i, :, 1].std()
        if minmax:
            y1 = 1
            y0 = -1
            xmin = x[i, :, 1].min()
            xmax = x[i, :, 1].max()
            delta_y = y1 - y0
            delta_x = xmax - xmin
            x[i, :, 1] = delta_y / delta_x * (x[i, :, 1] - xmin) + y0
            if x.shape[2] == 3:
                x[i, :, 2] = delta_y / delta_x * x[i, :, 2]
        else:
            x[i, :, 1] = (x[i, :, 1] - mean) / (std + eps)
            if x[i].shape[-1] == 3:
                x[i, :, 2] = x[i, :, 2] / (std + eps)
        means.append(mean)
        stds.append(std)
    return x, means, stds


def normalize_light_curves_w_list(x, eps=1e-10, minmax=False):
    means = list()
    stds = list()
    for i in range(len(x)):
        mean = x[i][:, 1].mean()
        std = x[i][:, 1].std()
        if minmax:
            y1 = 1
            y0 = -1
            xmin = x[i, :, 1].min()
            xmax = x[i, :, 1].max()
            delta_y = y1 - y0
            delta_x = xmax - xmin
            x[i, :, 1] = delta_y / delta_x * (x[i, :, 1] - xmin) + y0
            if x.shape[2] == 3:
                x[i, :, 2] = delta_y / delta_x * x[i, :, 2]
        else:
            x[i][:, 1] = (x[i][:, 1] - mean) / (std + eps)
            if x[i].shape[-1] == 3:
                x[i][:, 2] = x[i][:, 2] / (std + eps)
        means.append(mean)
        stds.append(std)
    return x, means, stds


class ASASSNDataset(object):
    def __init__(self, args, self_adv=False, oe=False, geotrans=False):
        #seed_everything(args.seed)
        self.args = args
        self.eps = 1e-10
        if self.args["fold"]:
            self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val, self.p_train, self.p_test, self.p_val = read_data(True)
        else:
            self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = read_data(False)
        
        self.ndim = self.x_train[0].shape[-1]
        self.n_inlier_classes = len(np.unique(self.y_train))

        self.lab2idx = load_json("../datasets/asas_sn/lab2idx.json")

        # magnitude normalization
        self.x_train, self.m_train, self.s_train = normalize_light_curves(self.x_train, minmax=False)
        self.x_val, self.m_val, self.s_val = normalize_light_curves(self.x_val, minmax=False)
        self.x_test, self.m_test, self.s_test = normalize_light_curves(self.x_test, minmax=False)

        # time normalization
        self.x_train = time_norm(self.x_train)
        self.x_test = time_norm(self.x_test)
        self.x_val = time_norm(self.x_val)

        self.average_precision = (self.y_test == 8).sum() / len(self.y_test)

        self.seq_len_train = calculate_seq_len(self.x_train)
        self.seq_len_val = calculate_seq_len(self.x_val)
        self.seq_len_test = calculate_seq_len(self.x_test)

        if self.args["fold"]:
            self.train_dataset = MyFoldedDataset(self.x_train, self.y_train, self.m_train, self.s_train, self.p_train, self.seq_len_train, device="cpu")
            self.val_dataset = MyFoldedDataset(self.x_val, self.y_val, self.m_val, self.s_val, self.p_val, self.seq_len_val, device="cpu")
            self.test_dataset = MyFoldedDataset(self.x_test, self.y_test, self.m_test, self.s_test, self.p_test, self.seq_len_test, device="cpu")
        else:
            self.train_dataset = MyDataset(self.x_train, self.y_train, self.seq_len_train, device="cpu")
            self.val_dataset = MyDataset(self.x_val, self.y_val, self.seq_len_val, device="cpu")
            self.test_dataset = MyDataset(self.x_test, self.y_test, self.seq_len_test, device="cpu")

        # # balancing
        # labs, counts = np.unique(self.y_train, return_counts=True)
        # # mask = labs != -99
        # # weights = 1 / counts[mask]
        # # weights /= 2 * weights.sum()
        # # weights = np.insert(weights, 0, 0.5)

        # weights = 1 / counts
        # weights /= weights.sum()
        
        # sample_weight = np.zeros(len(self.y_train))
        # for i, lab in enumerate(labs):
        #     mask = self.y_train == lab
        #     sample_weight[mask] = weights[i]
        # sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.bs, sampler=sampler)
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.bs, shuffle=True)
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.bs, shuffle=False)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args["bs"], shuffle=True, collate_fn=pad_sequence_with_lengths)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args["bs"], shuffle=True, collate_fn=pad_sequence_with_lengths)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args["bs"], shuffle=False, collate_fn=pad_sequence_with_lengths)


def pad_sequence_with_lengths(data):
    x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=0., batch_first=True)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    seq_len = torch.tensor([d[2] for d in data], dtype=torch.float)
    return x, y, seq_len
