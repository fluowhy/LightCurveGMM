import torch
import numpy as np
from tqdm import tqdm

from models import GRUGMM
# from models import LSTMGMM

from utils import count_parameters
from utils import WMSELoss
from utils import compute_energy


def singularity_weight(cov):
    return (1 / torch.diagonal(cov, dim1=1, dim2=2)).sum()


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
        # log_path = "logs/{}/{}".format(args["dataset"], args["config"])
        # if os.path.exists(log_path) and os.path.isdir(log_path):
            # shutil.rmtree(log_path)
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
            # if counter == self.args["patience"]:
                # break
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