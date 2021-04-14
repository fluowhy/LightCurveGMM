import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import load_json
from utils import calculate_seq_len
from utils import MyDataset

from utils_light_curves import remove_data_from_selected_class
from utils_light_curves import process_self_adversarial
from utils_light_curves import normalize_light_curves
from utils_light_curves import time_norm


class ZTFDataset(object):
    def __init__(self, args, self_adv=False, cv_oc=[]):
        #seed_everything(args.seed)
        self.args = args
        self.eps = 1e-10
        self.read_data()
        self.lab2idx = load_json("../datasets/ztf/lab2idx.json")
        self.family = load_json("../datasets/ztf/family.json")

        assert isinstance(cv_oc, list), "cv_oc not a list"

        # remove outliers from train and val data only if there is some oc
        if cv_oc:
            for key in self.family:
                flab = self.family[key]
                if flab == cv_oc[0]:
                    try:
                        print(key, flab, self.lab2idx[key])
                        self.x_train, self.y_train = remove_data_from_selected_class(self.x_train, self.y_train, self.lab2idx[key])
                        self.x_val, self.y_val = remove_data_from_selected_class(self.x_val, self.y_val, self.lab2idx[key])
                    except:
                        print(key, flab, "not found")

        # add transformations
        if self_adv:
            self.x_train, self.y_train = process_self_adversarial(self.x_train, self.y_train, args)
            self.x_val, self.y_val = process_self_adversarial(self.x_val, self.y_val, args)

        # magnitude normalization
        self.x_train, self.mean_train, self.std_train = normalize_light_curves(self.x_train, minmax=False)
        self.x_val, self.mean_val, self.std_val = normalize_light_curves(self.x_val, minmax=False)
        self.x_test, self.mean_test, self.std_test = normalize_light_curves(self.x_test, minmax=False)

        # time normalization
        self.x_train = time_norm(self.x_train, log=True)
        self.x_test = time_norm(self.x_test, log=True)
        self.x_val = time_norm(self.x_val, log=True)
        
        self.average_precision = 0
        if cv_oc:
            for key in self.family:
                flab = self.family[key]
                if flab == cv_oc[0]:
                    try:
                        print(key, flab, self.lab2idx[key], (self.y_test == self.lab2idx[key]).sum())
                        self.average_precision += (self.y_test == self.lab2idx[key]).sum()
                    except:
                        print(key, flab, "not found")
        print(self.average_precision)
        print(len(self.y_test))
        self.average_precision /= len(self.y_test)
        if cv_oc:
            print("{}, avg pre {}".format(cv_oc[0], self.average_precision))

        self.seq_len_train = calculate_seq_len(self.x_train)
        self.seq_len_val = calculate_seq_len(self.x_val)
        self.seq_len_test = calculate_seq_len(self.x_test)

        # temporal class shift        
        idx = 0
        self.temp_labels_dict = {}
        for lab in np.unique(self.y_train):
            self.temp_labels_dict[lab] = idx
            idx += 1
        self.y_train = np.array([self.temp_labels_dict[lab] for lab in self.y_train])
        self.y_val = np.array([self.temp_labels_dict[lab] for lab in self.y_val])
        self.n_inlier_classes = len(np.unique(self.y_train))
        self.ndim = self.x_train.shape[2]
        
        self.train_dataset = MyDataset(self.x_train, self.y_train, self.mean_train, self.std_train, self.seq_len_train, device=args["d"])
        self.val_dataset = MyDataset(self.x_val, self.y_val, self.mean_val, self.std_val, self.seq_len_val, device=args["d"])
        self.test_dataset = MyDataset(self.x_test, self.y_test, self.mean_test, self.std_test, self.seq_len_test, device=args["d"])        
        
        # balancing
        labs, counts = np.unique(self.y_train, return_counts=True)
        # mask = labs != -99
        # weights = 1 / counts[mask]
        # weights /= 2 * weights.sum()
        # weights = np.insert(weights, 0, 0.5)

        weights = 1 / counts
        weights /= weights.sum()
        
        sample_weight = np.zeros(len(self.y_train))
        for i, lab in enumerate(labs):
            mask = self.y_train == lab
            sample_weight[mask] = weights[i]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args["bs"], shuffle=True)#, sampler=sampler)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args["bs"], shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args["bs"], shuffle=False)

    def read_data(self):
        loaded = np.load("../datasets/ztf/train_data.npz")
        self.x_train = loaded["x_train"]
        self.x_test = loaded["x_test"]
        self.x_val = loaded["x_val"]
        self.y_train =loaded["y_train"]
        self.y_test = loaded["y_test"]
        self.y_val = loaded["y_val"]
        return
