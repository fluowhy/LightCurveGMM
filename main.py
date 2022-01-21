import argparse
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from models import GRUGMM

# from datasets import pad_sequence_with_lengths
# from datasets import read_and_normalize_data

from utils import count_parameters
from utils import compute_energy
from utils import WMSELoss
from utils import make_dir
from utils import load_json
# from utils import MyDataset
from utils import seed_everything
from utils import load_yaml
from utils import get_data_loaders
from utils import od_metrics
from utils import save_yaml

from models import compute_params


def singularity_weight(cov):
    return (1 / torch.diagonal(cov, dim1=1, dim2=2)).sum()


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


class Model(object):
    def __init__(self, args, oc, model_dir):
        self.args = args
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        self.device = args["d"]
        self.oc = oc
        self.model_dir = model_dir
        if args["arch"] == "gru":
            self.model = GRUGMM(args["nin"], args["nh"], args["nl"], args["ne"], args["ngmm"], args["nout"], args["nlayers"], args["do"], args["fold"])
        elif args["arch"] == "lstm":
            self.model = LSTMGMM(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"], args["fold"])        
        self.model.to(args["d"])
        print("model params {}".format(count_parameters(self.model)))
        log_path = "logs/{}/{}".format(args["dataset"], args["config"])
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        # self.writer = SummaryWriter(log_path)
        nc = 3
        self.wmse = WMSELoss(nc=nc)
        print(nc, self.wmse)                
        self.ce = torch.nn.CrossEntropyLoss(reduction="mean")
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        recon_loss = 0
        ce_loss = 0
        gmm_loss = 0
        sw_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            if self.args["fold"]:
                x, y, m, s, p, seq_len = batch
            else:
                x, y, seq_len = batch
                x = x.to(self.device)
                y = y.to(self.device)
                seq_len = seq_len.to(self.device)
                p = None
            x_pred, h, logits, phi, mu, cov = self.model(x, seq_len.long(), p)
            recon = self.wmse(x, x_pred, seq_len).mean()
            ce = self.ce(logits, y)
            energy = compute_energy(h, phi, mu, cov).mean()
            sw = singularity_weight(cov)
            loss = recon + self.alpha * ce + self.beta * energy + self.gamma * sw
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
            recon_loss += recon.item()
            ce_loss += ce.item()
            gmm_loss += energy.item()
            sw_loss += sw.item()
        train_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        gmm_loss /= (idx + 1)
        sw_loss /= (idx + 1)
        return train_loss, recon_loss, ce_loss, gmm_loss, sw_loss

    def eval_model(self, data_loader):
        self.model.eval()
        eval_loss = 0
        recon_loss = 0
        ce_loss = 0
        gmm_loss = 0
        sw_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                if self.args["fold"]:
                    x, y, m, s, p, seq_len = batch
                else:
                    x, y, seq_len = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    seq_len = seq_len.to(self.device)
                    p = None
                x_pred, h, logits, phi, mu, cov = self.model(x, seq_len.long(), p)
                recon = self.wmse(x, x_pred, seq_len).mean()
                ce = self.ce(logits, y)
                energy = compute_energy(h, phi, mu, cov).mean()
                sw = singularity_weight(cov)
                loss = recon + self.alpha * ce + self.beta * energy + self.gamma * sw
                eval_loss += loss.item()
                recon_loss += recon.item()
                ce_loss += ce.item()
                gmm_loss += energy.item()
                sw_loss += sw.item()
        eval_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        gmm_loss /= (idx + 1)
        sw_loss /= (idx + 1)
        return eval_loss, recon_loss, ce_loss, gmm_loss, sw_loss

    def fit(self, train_loader, val_loader, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args["lr"], weight_decay=args["wd"], amsgrad=True)
        counter = 0
        train_losses = list()
        val_losses = list()
        for epoch in range(args["e"]):
            counter += 1
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss, train_recon_loss, train_ce_loss, train_gmm_loss, train_sw = self.train_model(train_loader)
            val_loss, val_recon_loss, val_ce_loss, val_gmm_loss, val_sw = self.eval_model(val_loader)
            # self.writer.add_scalars("total", {"train": train_loss, "val": val_loss}, global_step=epoch)
            # self.writer.add_scalars("recon", {"train": train_recon_loss, "val": val_recon_loss}, global_step=epoch)
            # self.writer.add_scalars("cross_entropy", {"train": train_ce_loss, "val": val_ce_loss}, global_step=epoch)
            # self.writer.add_scalars("gmm", {"train": train_gmm_loss, "val": val_gmm_loss}, global_step=epoch)
            train_losses.append((train_loss, train_recon_loss, train_ce_loss, train_gmm_loss, train_sw))
            val_losses.append((val_loss, val_recon_loss, val_ce_loss, val_gmm_loss, val_sw))
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.model_dir}/best_model.pth")
                counter = 0       
            if counter == self.args["patience"]:
                break
        torch.save(self.model.state_dict(), f"{self.model_dir}/last_model.pth")
        return train_losses, val_losses
    
    def load_model(self):
        self.model.load_state_dict(torch.load(f"{self.model_dir}/best_model.pth", map_location=self.device))
        return
    
    def compute_features(self, x, is_dataloader=False):
        self.model.eval()
        with torch.no_grad():
            if is_dataloader:
                features = list()
                targets = list()
                logits = list()
                for idx, batch in tqdm(enumerate(x)):
                    x, y, seq_len = batch
                    _, h, logit, _, _, _ = self.model(x, seq_len.long(), p=None)
                    targets.append(y)
                    features.append(h)
                    logits.append(logit)
                features = torch.vstack(features)
                targets = torch.hstack(targets)
                logits = torch.vstack(logits)
        return features, targets, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="linear",
        choices=["asas", "linear", "macho", "asas_sn", "ztf_transient", "ztf_stochastic", "ztf_periodic"],
        help="dataset name (default linear)"
    )
    parser.add_argument('--config', type=int, default=0, help="config number (default 0)")
    parser.add_argument('--device', type=str, default="cpu", help="device (default cpu)")
    parser.add_argument('--oc', type=int, default=0, help="outlier class (default 0)")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")
    make_dir("files")

    make_dir("figures/{}".format(args.dataset))
    make_dir("models/{}".format(args.dataset))
    make_dir("files/{}".format(args.dataset))

    models_dir = f"models/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    figures_dir = f"figures/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    files_dir = f"files/od/{args.dataset}/config_{args.config}/oc_{args.oc}"

    make_dir(figures_dir)
    make_dir(files_dir)
    make_dir(models_dir)

    seed_everything()

    config_args = load_yaml(f"config/config_{args.config}.yaml")
    config_args["config"] = args.config
    config_args["e"] = args.e

    if args.dataset == "asas_sn":
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"])
        outlier_class = [8]
    elif "ztf" in args.dataset:
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"])
        outlier_labels = load_json("../datasets/ztf/cl/transient/lab2out.json")
        outlier_labels = [key for key in outlier_labels if outlier_labels[key] == "outlier"]
        lab2idx = load_json("../datasets/ztf/cl/transient/lab2idx.json")
        outlier_class = [int(lab2idx[lab]) for lab in outlier_labels]
    elif args.dataset == "asas":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"], args.oc)
    elif args.dataset == "linear":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"], args.oc)
    
    config_args["nin"] = trainloader.dataset.x[0].shape[1]
    config_args["ngmm"] = len(np.unique(testloader.dataset.y))
    print(config_args)

    aegmm = Model(config_args, outlier_class, models_dir)
    train_loss, val_loss = aegmm.fit(trainloader, valloader, config_args)
    plot_loss(train_loss, val_loss, "figures/{}/loss.png".format(args.dataset))
    aegmm.load_model()

    # evaluation
    softmax = torch.nn.Softmax(dim=1)

    feat_train, y_train, logits_train = aegmm.compute_features(trainloader, is_dataloader=True)
    feat_val, y_val, logits_val = aegmm.compute_features(valloader, is_dataloader=True)
    feat_test, y_test, logits_test = aegmm.compute_features(testloader, is_dataloader=True)

    phi_val, mu_val, cov_val = compute_params(feat_val, softmax(logits_val))
    val_energy = compute_energy(feat_val, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()
    test_energy = compute_energy(feat_test, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()

    y_target = np.ones(len(y_test))

    for oc in outlier_class:
        y_target[y_test == oc] = 0
    
    aucpr, _, _, aucroc, _, _ = od_metrics(test_energy, y_target, split=True, n_splits=100)

    data = dict(
        aucpr=float(np.mean(aucpr)),
        aucpr_std=float(np.std(aucpr)),
        aucroc=float(np.mean(aucroc)),
        aucroc_std=float(np.std(aucroc)),
        val_acc=float(((logits_val.argmax(1).cpu() == y_val).sum() / len(y_val)).item()),
        dataset=args.dataset,
        oc=args.oc,
        config=args.config
    )

    save_yaml(data, f"{files_dir}/metrics.yaml")
