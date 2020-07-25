import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *
from datasets import LightCurveDataset


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
        self.wmse = WMSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        recon_loss = 0
        ce_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            if self.args.fold:
                x, y, m, s, p, seq_len = batch
            else:
                x, y, m, s, seq_len = batch
                p = None
            # x = x.to(self.args.d)
            # seq_len = seq_len.to(self.args.d)            
            x_pred, h, logits = self.model(x, m, s, seq_len.long(), p)
            recon_loss = self.wmse(x, x_pred, seq_len).mean()
            ce_loss = self.ce(logits, y)
            loss = recon_loss + self.args.alpha * ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
            recon_loss += recon_loss.item()
            ce_loss += ce_loss.item()
        train_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        return train_loss, recon_loss, ce_loss

    def eval_model(self, data_loader):
        self.model.eval()
        eval_loss = 0
        recon_loss = 0
        ce_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                if self.args.fold:
                    x, y, m, s, p, seq_len = batch
                else:
                    x, y, m, s, seq_len = batch
                    p = None
                # x = x.to(self.args.d)
                # seq_len = seq_len.to(self.args.d)
                x_pred, h, logits = self.model(x, m, s, seq_len.long(), p)
                recon_loss = self.wmse(x, x_pred, seq_len).mean()
                ce_loss = self.ce(logits, y)
                loss = recon_loss + self.args.alpha * ce_loss
                eval_loss += loss.item()
                recon_loss += recon_loss.item()
            ce_loss += ce_loss.item()
        eval_loss /= (idx + 1)
        recon_loss /= (idx + 1)
        ce_loss /= (idx + 1)
        return eval_loss, recon_loss, ce_loss

    def fit(self, train_loader, val_loader, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        loss = np.zeros((args.e, 2, 3))
        for epoch in range(args.e):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss, train_recon_loss, train_ce_loss = self.train_model(train_loader)
            val_loss, val_recon_loss, val_ce_loss = self.eval_model(val_loader)
            loss[epoch, 0] = (train_loss, train_recon_loss, train_ce_loss)
            loss[epoch, 1] = (val_loss, val_recon_loss, val_ce_loss)
            self.writer.add_scalars("total", {"train": train_loss, "val": val_loss}, global_step=epoch)
            self.writer.add_scalars("recon", {"train": train_recon_loss, "val": val_recon_loss}, global_step=epoch)
            self.writer.add_scalars("cross_entropy", {"train": train_ce_loss, "val": val_ce_loss}, global_step=epoch)
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_model = self.model.state_dict()
                self.best_loss = val_loss
        torch.save(self.best_model, "models/{}/{}.pth".format(self.args.name, self.args.arch))
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--bs', type=int, default=128, help="batch size (default 128)")
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
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"], help="rnn architecture (default gru)")
    parser.add_argument("--name", type=str, default="linear", choices=["linear", "macho", "asas"], help="dataset name (default linear)")
    parser.add_argument("--fold", action="store_true", help="folded light curves")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")

    make_dir("figures/{}".format(args.name))
    make_dir("models/{}".format(args.name))

    seed_everything()

    dataset = LightCurveDataset(args.name, fold=args.fold, bs=args.bs, device=args.d, eval=True)  
    args.nin = dataset.x_train.shape[2]
    args.ngmm = len(np.unique(dataset.y_train))

    aegmm = Model(args)
    loss = aegmm.fit(dataset.train_dataloader, dataset.val_dataloader, args)
    plot_loss(loss, "figures/{}/{}_loss.png".format(args.name, args.arch))
