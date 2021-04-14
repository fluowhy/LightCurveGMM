import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import partial

from utils import load_json
from utils import calculate_seq_len
from utils import MyDataset

from utils_light_curves import remove_data_from_selected_class
from utils_light_curves import process_self_adversarial
from utils_light_curves import normalize_light_curves
from utils_light_curves import time_norm
from utils import pad_seq


class ZTFDataset(Dataset):
    def __init__(self, args, split, self_adv=False, cv_oc=[]):
        #seed_everything(args.seed)
        self.args = args
        self.eps = 1e-10
        self.read_data(split)
        self.lab2idx = load_json("../datasets/ztf/lab2idx.json")
        self.family = load_json("../datasets/ztf/family.json")

        assert isinstance(cv_oc, list), "cv_oc not a list"

        # remove outliers from train and val data only if there is some oc
        if cv_oc and ((split == "train") or (split == "val")):
            for key in self.family:
                flab = self.family[key]
                if flab == cv_oc[0]:
                    try:
                        print(key, flab, self.lab2idx[key])
                        self.x, self.y = remove_data_from_selected_class(self.x, self.y, self.lab2idx[key])
                    except:
                        print(key, flab, "not found")

        _, _, self.n_dim = self.x.shape

        # add transformations
        if self_adv and ((split == "train") or (split == "val")):
            self.x, self.y = process_self_adversarial(self.x, self.y, args)

        seq_len = (self.x[:, :, 2] != 0).sum(1)
        self.x = [torch.tensor(xi[:sl], dtype=torch.float, device=args["d"]) for xi, sl in zip(self.x, seq_len)]

        # temporal class shift
        if (split == "train") or (split == "val"):
            idx = 0
            self.temp_labels_dict = {}
            for lab in np.unique(self.y):
                if lab != -99:
                    self.temp_labels_dict[lab] = idx
                    idx += 1
                else:
                    self.temp_labels_dict[-99] = -99
            self.y = np.array([self.temp_labels_dict[lab] for lab in self.y])
            self.n_classes = len(np.unique(self.y)) - 1

        # preprocessing
        self.preprocess_data()
        self.n = len(self.x)
        self.ndim = self.x[0].shape[1]
        self.n_inlier_classes = len(np.unique(self.y))
        print("inlier classes", np.unique(self.y))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mean[idx], self.std[idx], self.seq_len[idx]

    def __len__(self):
        return self.n

    def read_data(self, split):
        data = np.load("../datasets/ztf/train_data.npz", allow_pickle=True)
        self.x = data["x_{}".format(split)]
        self.y = data["y_{}".format(split)]
        return
    
    def preprocess_data(self):
        self.mean = []
        self.std = []
        self.seq_len = []
        for lc in tqdm(self.x):
            seq_len = len(lc)
            self.seq_len.append(seq_len)
            std = lc[:, 1].std()
            mean = lc[:, 1].mean()
            tmin = lc[:, 0].min()
            lc[:, 1] -= mean
            lc[:, 1] /= std
            lc[:, 2] /= std
            lc[1:, 0] = lc[1:, 0] - lc[:-1, 0]
            lc[0, 0] = 0
            lc[seq_len:, 0] = 0
            self.mean.append(mean)
            self.std.append(std)
        return


if __name__ == "__main__":
    from code.utils import load_yaml
    from torch.utils.data import DataLoader
    from functools import partial

    import pdb

    args = load_yaml("configs/config_90.yaml")
    split = "val"   

    val_dataset = ZTFDataset(args, "val", self_adv=True, cv_oc=[args["outlier_fam"]])
    val_dataloader = DataLoader(val_dataset, batch_size=args["bs"], shuffle=True, collate_fn=partial(pad_seq, device=args["d"]))
    
    test_dataset = ZTFDataset(args, "test", self_adv=True, cv_oc=[args["outlier_fam"]])
    test_dataloader = DataLoader(test_dataset, batch_size=args["bs"], shuffle=True, collate_fn=partial(pad_seq, device=args["d"]))

    train_dataset = ZTFDataset(args, "train", self_adv=True, cv_oc=[args["outlier_fam"]])
    train_dataloader = DataLoader(train_dataset, batch_size=args["bs"], shuffle=True, collate_fn=partial(pad_seq, device=args["d"]))
    
    # balancing
    y_train = train_dataset[:][1]
    labs, counts = np.unique(np.unique(y_train), return_counts=True)
    weights = 1 / counts
    weights /= weights.sum()
    sample_weight = np.zeros(len(y_train))
    for i, lab in enumerate(labs):
        mask = y_train == lab
        sample_weight[mask] = weights[i]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))

    train_dataloader = DataLoader(train_dataset, batch_size=args["bs"], sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args["bs"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args["bs"], shuffle=False)
