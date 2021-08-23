import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy
from utils import pad_seq

from models import *
from utils import *


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
    def __init__(self, args):
        self.args = args
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        self.device = args["d"]
        if args["arch"] == "gru":
            self.model = GRUGMM(args["nin"], args["nh"], args["nl"], args["ne"], args["ngmm"], args["nout"], args["nlayers"], args["do"], args["fold"])
        elif args["arch"] == "lstm":
            self.model = LSTMGMM(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"], args["fold"])        
        self.model.to(args["d"])
        print("model params {}".format(count_parameters(self.model)))
        log_path = "logs/{}/{}".format(args["dataset"], args["config"])
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_path)
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
            loss = recon + self.alpha * ce + self.beta * energy  # + self.gamma * sw
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
                # x = x.to(self.args.d)
                # seq_len = seq_len.to(self.args.d)
                x_pred, h, logits, phi, mu, cov = self.model(x, seq_len.long(), p)
                recon = self.wmse(x, x_pred, seq_len).mean()
                ce = self.ce(logits, y)
                energy = compute_energy(h, phi, mu, cov).mean()
                sw = singularity_weight(cov)
                loss = recon + self.alpha * ce + self.beta * energy  # + self.gamma * sw
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
            self.writer.add_scalars("total", {"train": train_loss, "val": val_loss}, global_step=epoch)
            self.writer.add_scalars("recon", {"train": train_recon_loss, "val": val_recon_loss}, global_step=epoch)
            self.writer.add_scalars("cross_entropy", {"train": train_ce_loss, "val": val_ce_loss}, global_step=epoch)
            self.writer.add_scalars("gmm", {"train": train_gmm_loss, "val": val_gmm_loss}, global_step=epoch)
            train_losses.append((train_loss, train_recon_loss, train_ce_loss, train_gmm_loss, train_sw))
            val_losses.append((val_loss, val_recon_loss, val_ce_loss, val_gmm_loss, val_sw))
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "models/{}/config_{}/best_model.pth".format(self.args["dataset"], self.args["config"]))
                counter = 0       
            if counter == self.args["patience"]:
                break
        torch.save(self.model.state_dict(), "models/{}/config_{}/last_model.pth".format(self.args["dataset"], self.args["config"]))
        return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--dataset', type=str, choices=["ztf", "asas_sn"], default="ztf", help="dataset (default ztf)")
    parser.add_argument('--config', type=int, default=0, help="config number (default 0)")
    args = parser.parse_args()

    make_dir("figures")
    make_dir("models")
    make_dir("files")

    make_dir("figures/{}".format(args.dataset))
    make_dir("models/{}".format(args.dataset))
    make_dir("files/{}".format(args.dataset))

    make_dir("figures/{}/config_{}".format(args.dataset, args.config))
    make_dir("models/{}/config_{}".format(args.dataset, args.config))
    make_dir("files/{}/config_{}".format(args.dataset, args.config))

    seed_everything()

    config_args = load_yaml("config/{}/config_{}.yaml".format(args.dataset, args.config))
    config_args["config"] = args.config

    if args.dataset == "asas_sn":
        from datasets import ASASSNDataset


        dataset = ASASSNDataset(config_args, self_adv=False, oe=False, geotrans=False)
    elif args.dataset == "ztf":
        from ztf_dataset import ZTFDataset


        self_adv = False       
        dataset = ZTFDataset(config_args, self_adv=self_adv, cv_oc=[config_args["outlier_fam"]])
        family = load_json("../datasets/ztf/family.json")
        lab2idx = load_json("../datasets/ztf/lab2idx.json")        
        outlier_class = [lab2idx[key] for key in lab2idx if family[key] == config_args["outlier_fam"]]
        inlier_class = [lab2idx[key] for key in lab2idx if family[key] != config_args["outlier_fam"]]
        print("inlier", inlier_class)
        print("outlier", outlier_class)
        maha = False
    
    config_args["nin"] = dataset.ndim
    config_args["ngmm"] = dataset.n_inlier_classes
    print(config_args)

    aegmm = Model(config_args)
    train_loss, val_loss = aegmm.fit(dataset.train_dataloader, dataset.val_dataloader, config_args)
    plot_loss(train_loss, val_loss, "figures/{}/loss.png".format(args.dataset))
