import argparse
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
import pdb

from models import *
from utils import *
from datasets import ASASSNDataset


def plot_loss(loss, savename, dpi=200):
    epochs = np.arange(len(loss))
    plt.figure()
    plt.plot(epochs, loss[:, 0], color="navy", label="train")
    plt.plot(epochs, loss[:, 1], color="red", label="val")
    plt.grid()
    plt.ylabel("WMSE")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi)
    return


class Model(object):
    def __init__(self, args):
        self.args = args
        if args.arch == "gru":
            self.model = GRUAE(args.nin, args.nh, args.nl, args.ne, args.ngmm, args.nout, args.nlayers, args.do)
        elif args.arch == "lstm":
            self.model = LSTMAE(args.nin, args.nh, args.nl, args.nout, args.nlayers, args.do)        
        self.model.to(args.d)
        print("model params {}".format(count_parameters(self.model)))
        #log_path = "logs/autoencoder"
        #if os.path.exists(log_path) and os.path.isdir(log_path):
        #    shutil.rmtree(log_path)
        #self.writer = SummaryWriter(log_path)
        nc = 2 if args.dataset == "toy" else 3
        self.wmse = WMSELoss(nc=nc)
        print(nc, self.wmse)                
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            x, y, seq_len, p = batch
            x = x.to(self.args.d)
            y = y.to(self.args.d)
            seq_len = seq_len.to(self.args.d)
            p = p.to(self.args.d)
            x_pred, h = self.model(x, seq_len.long())
            loss = self.wmse(x, x_pred, seq_len).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= (idx + 1)
        return train_loss

    def eval_model(self, data_loader):
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                x, y, seq_len, p = batch
                x = x.to(self.args.d)
                y = y.to(self.args.d)
                seq_len = seq_len.to(self.args.d)
                p = p.to(self.args.d)
                x_pred, h = self.model(x, seq_len.long())
                loss = self.wmse(x, x_pred, seq_len).mean()
                eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def fit(self, train_loader, val_loader):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, amsgrad=True)
        loss = np.zeros((self.args.e, 2))
        counter = 0
        for epoch in range(self.args.e):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss = self.train_model(train_loader)
            val_loss = self.eval_model(val_loader)
            loss[epoch] = (train_loss, val_loss)
            #self.writer.add_scalars("total", {"train": train_loss, "val": val_loss}, global_step=epoch)
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_model = self.model.state_dict()
                self.best_loss = val_loss
                counter = 0
            else:
                counter += 1
            if counter == self.args.patience:
                break
        loss = loss[:epoch + 1]
        self.last_model = self.model.state_dict()
        torch.save(self.best_model, "models/{}/fold_{}/{}_best.pth".format(self.args.dataset, self.args.fold, self.args.arch))
        torch.save(self.last_model, "models/{}/fold_{}/{}_last.pth".format(self.args.dataset, self.args.fold, self.args.arch))
        return loss

    def cosine_distance(self, x, x_pred):
        x_norm = x[:, :, 1].pow(2).sum(1).sqrt()
        x_pred_norm = x_pred.pow(2).sum(1).sqrt()
        dot_prod = (x[:, :, 1] * x_pred).sum(1)
        return dot_prod / x_norm / x_pred_norm

    def evaluate(self, data_loader):
        feature_vector = torch.empty((0, self.args.nl + 3), dtype=torch.float, device=self.args.d)
        targets = torch.empty((0), dtype=torch.long, device=self.args.d)
        self.model.load_state_dict(self.last_model)
        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                x, y, seq_len, p = batch
                x = x.to(self.args.d)
                y = y.to(self.args.d)
                seq_len = seq_len.to(self.args.d)
                p = p.to(self.args.d)
                x_pred, h = self.model(x, seq_len.long())
                loss = self.wmse(x, x_pred, seq_len)
                cosine_distance = self.cosine_distance(x, x_pred)
                log_p = torch.log10(p)
                fv = torch.cat((h, loss.unsqueeze(-1), cosine_distance.unsqueeze(-1), log_p.unsqueeze(-1)), dim=1)
                feature_vector = torch.cat((feature_vector, fv))
                targets = torch.cat((targets, y))
        return feature_vector, targets

    def save_features(self, x, y, savename):
        np.save("{}_x.npy".format(savename), x.cpu().numpy())
        np.save("{}_y.npy".format(savename), y.cpu().numpy())
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
    parser.add_argument('--nlayers', type=int, default=1, help="number of hidden layers (default 1)")
    parser.add_argument("--do", type=float, default=0., help="dropout value (default 0)")
    parser.add_argument("--wd", type=float, default=0., help="L2 reg value (default 0)")
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"], help="rnn architecture (default gru)")
    parser.add_argument("--dataset", type=str, default="linear", choices=["linear", "macho", "asas", "asas_sn", "toy"], help="dataset name (default linear)")
    parser.add_argument("--fold", action="store_true", help="folded light curves")
    parser.add_argument('--patience', type=int, default=2, help="early stopping patience (default 2)")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")
    make_dir("files")
    make_dir("features")

    make_dir("figures/{}".format(args.dataset))
    make_dir("models/{}".format(args.dataset))
    make_dir("files/{}".format(args.dataset))

    make_dir("figures/{}/fold_{}".format(args.dataset, args.fold))
    make_dir("models/{}/fold_{}".format(args.dataset, args.fold))
    make_dir("files/{}/fold_{}".format(args.dataset, args.fold))

    seed_everything()

    if args.dataset == "asas_sn":
        dataset = ASASSNDataset(args, self_adv=False, oe=False, geotrans=False)
    elif args.dataset == "toy":
        from toy_dataset import ToyDataset
        dataset = ToyDataset(args, val_size=0.1, sl=64)
    args.nin = dataset.x_train.shape[2]
    args.ngmm = len(np.unique(dataset.y_train))

    autoencoder = Model(args)
    loss = autoencoder.fit(dataset.train_dataloader, dataset.val_dataloader)
    plot_loss(loss, "figures/{}/fold_{}/{}_loss.png".format(args.dataset, args.fold, args.arch), dpi=400)
    features, targets = autoencoder.evaluate(dataset.train_dataloader)
    autoencoder.save_features(features, targets, "features/train")
    features, targets = autoencoder.evaluate(dataset.test_dataloader)
    autoencoder.save_features(features, targets, "features/test")
    features, targets = autoencoder.evaluate(dataset.val_dataloader)
    autoencoder.save_features(features, targets, "features/val")
