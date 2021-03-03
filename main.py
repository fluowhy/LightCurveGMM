import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy

from models import *
from utils import *
from datasets import ASASSNDataset


class Model(object):
    def __init__(self, args):
        self.args = args
        if args.arch == "gru":
            self.model = GRUGMM(args.nin, args.nh, args.nl, args.ne, args.ngmm, args.nout, args.nlayers, args.do, args.fold)
        elif args.arch == "lstm":
            self.model = LSTMGMM(args.nin, args.nh, args.nl, args.nout, args.nlayers, args.do, args.fold)        
        self.model.to(args.d)
        print("model params {}".format(count_parameters(self.model)))
        log_path = "logs/autoencoder"
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_path)
        nc = 2 if args.name == "toy" else 3
        self.wmse = WMSELoss(nc=nc)
        print(nc, self.wmse)                
        self.ce = torch.nn.CrossEntropyLoss()
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        recon_loss = 0
        ce_loss = 0
        gmm_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            if self.args.fold:
                x, y, m, s, p, seq_len = batch
                x, y, m, s, p, seq_len = x.to(self.args.d), y.to(self.args.d), m.to(self.args.d), s.to(self.args.d), p.to(self.args.d), seq_len.to(self.args.d)
            else:
                x, y, m, s, seq_len = batch
                x, y, m, s, seq_len = x.to(self.args.d), y.to(self.args.d), m.to(self.args.d), s.to(self.args.d), seq_len.to(self.args.d)
                p = None
            x_pred, h, logits, phi, mu, cov = self.model(x, m, s, seq_len.long(), p)
            recon = self.wmse(x, x_pred, seq_len).mean()
            ce = self.ce(logits, y)
            energy = compute_energy(h, phi, mu, cov).mean()
            loss = recon + self.args.alpha * ce + self.args.beta * energy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
            recon_loss += recon.item()
            ce_loss += ce.item()
            gmm_loss += energy.item()
        train_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        gmm_loss /= (idx + 1)
        return train_loss, recon_loss, ce_loss, gmm_loss

    def eval_model(self, data_loader):
        self.model.eval()
        eval_loss = 0
        recon_loss = 0
        ce_loss = 0
        gmm_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                if self.args.fold:
                    x, y, m, s, p, seq_len = batch
                    x, y, m, s, p, seq_len = x.to(self.args.d), y.to(self.args.d), m.to(self.args.d), s.to(self.args.d), p.to(self.args.d), seq_len.to(self.args.d)
                else:
                    x, y, m, s, seq_len = batch
                    x, y, m, s, seq_len = x.to(self.args.d), y.to(self.args.d), m.to(self.args.d), s.to(self.args.d), seq_len.to(self.args.d)
                    p = None
                # x = x.to(self.args.d)
                # seq_len = seq_len.to(self.args.d)
                x_pred, h, logits, phi, mu, cov = self.model(x, m, s, seq_len.long(), p)
                recon = self.wmse(x, x_pred, seq_len).mean()
                ce = self.ce(logits, y)
                energy = compute_energy(h, phi, mu, cov).mean()
                loss = recon + self.args.alpha * ce + self.args.beta * energy
                eval_loss += loss.item()
                recon_loss += recon.item()
                ce_loss += ce.item()
                gmm_loss += energy.item()
        eval_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        gmm_loss /= (idx + 1)
        return eval_loss, recon_loss, ce_loss, gmm_loss

    def fit(self, train_loader, val_loader, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        for epoch in range(args.e):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss, train_recon_loss, train_ce_loss, train_gmm_loss = self.train_model(train_loader)
            val_loss, val_recon_loss, val_ce_loss, val_gmm_loss = self.eval_model(val_loader)
            self.writer.add_scalars("total", {"train": train_loss, "val": val_loss}, global_step=epoch)
            self.writer.add_scalars("recon", {"train": train_recon_loss, "val": val_recon_loss}, global_step=epoch)
            self.writer.add_scalars("cross_entropy", {"train": train_ce_loss, "val": val_ce_loss}, global_step=epoch)
            self.writer.add_scalars("gmm", {"train": train_gmm_loss, "val": val_gmm_loss}, global_step=epoch)
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_loss = val_loss
        model_state_dict = copy.deepcopy(self.model.state_dict())
        torch.save(self.best_model, "models/{}/fold_{}/{}_best.pth".format(self.args.name, self.args.fold, self.args.arch))
        torch.save(model_state_dict, "models/{}/fold_{}/{}_last.pth".format(self.args.name, self.args.fold, self.args.arch))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--bs', type=int, default=2048, help="batch size (default 2048)")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
    parser.add_argument('--nout', type=int, default=1, help="output size (default 1)")
    parser.add_argument('--nh', type=int, default=2, help="hidden size (default 2)")
    parser.add_argument('--nl', type=int, default=2, help="hidden size (default 2)")
    parser.add_argument('--ne', type=int, default=2, help="estimation network size (default 2)")
    parser.add_argument('--nlayers', type=int, default=1, help="number of hidden layers (default 1)")
    parser.add_argument("--do", type=float, default=0., help="dropout value (default 0)")
    parser.add_argument("--wd", type=float, default=0., help="L2 reg value (default 0)")
    parser.add_argument("--alpha", type=float, default=1., help="cross entropy weight (default 1)")
    parser.add_argument("--beta", type=float, default=1e-3, help="gmm energy weight (default 1e-3)")
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"], help="rnn architecture (default gru)")
    parser.add_argument("--name", type=str, default="linear", choices=["linear", "macho", "asas", "asas_sn", "toy"], help="dataset name (default linear)")
    parser.add_argument("--fold", action="store_true", help="folded light curves")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")
    make_dir("files")

    make_dir("figures/{}".format(args.name))
    make_dir("models/{}".format(args.name))
    make_dir("files/{}".format(args.name))

    make_dir("figures/{}/fold_{}".format(args.name, args.fold))
    make_dir("models/{}/fold_{}".format(args.name, args.fold))
    make_dir("files/{}/fold_{}".format(args.name, args.fold))

    seed_everything()

    if args.name == "asas_sn":
        dataset = ASASSNDataset(args, self_adv=False, oe=False, geotrans=False)
    elif args.name == "toy":
        from toy_dataset import ToyDataset
        dataset = ToyDataset(args, val_size=0.1, sl=64)
    else:
        dataset = LightCurveDataset(args.name, fold=args.fold, bs=args.bs, device=args.d, eval=True)
    args.nin = dataset.x_train.shape[2]
    args.ngmm = len(np.unique(dataset.y_train))

    aegmm = Model(args)
    aegmm.fit(dataset.train_dataloader, dataset.val_dataloader, args)
