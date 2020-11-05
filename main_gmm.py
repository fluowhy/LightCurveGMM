import argparse
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
import pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix

from models import MLP
from utils import *


def plot_loss(loss, savename, dpi=200):
    epochs = np.arange(len(loss))
    plt.figure()
    plt.plot(epochs, loss[:, 0, 0], color="navy", label="total train")
    plt.plot(epochs, loss[:, 0, 1], color="navy", linestyle="-.", label="ce train")
    plt.plot(epochs, loss[:, 0, 2], color="navy", linestyle="--", label="gmm train")
    plt.plot(epochs, loss[:, 1, 0], color="red", label="total val")
    plt.plot(epochs, loss[:, 1, 1], color="red", linestyle="-.", label="ce val")
    plt.plot(epochs, loss[:, 1, 2], color="red", linestyle="--", label="gmm val")
    plt.grid()
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi)
    return


class Model(object):
    def __init__(self, args):
        self.args = args
        self.model = MLP(args.nin, args.nh, args.nout, args.do)
        self.model.to(args.d)
        print("model params {}".format(count_parameters(self.model)))
        #log_path = "logs/autoencoder"
        #if os.path.exists(log_path) and os.path.isdir(log_path):
        #    shutil.rmtree(log_path)
        #self.writer = SummaryWriter(log_path)
        nc = 2 if args.dataset == "toy" else 3
        self.best_loss = np.inf
        self.softmax = torch.nn.Softmax(dim=1)
        self.ce = torch.nn.CrossEntropyLoss()

    def compute_gmm_loss(self, z, logits):
        gamma = self.softmax(logits)
        phi, mu, cov = compute_params(z, gamma)
        sample_energy = compute_energy(z, phi=phi, mu=mu, cov=cov, size_average=True)
        return sample_energy

    def train_model(self, data_loader):
        self.model.train()
        train_loss = 0
        train_ce_loss = 0
        train_gmm_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.args.d)
            y = y.to(self.args.d)
            logits = self.model(x)
            ce_loss = self.ce(logits, y)
            gmm_loss = self.compute_gmm_loss(x, logits)
            loss = ce_loss + self.args.alpha * gmm_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_gmm_loss += gmm_loss.item()
        train_loss /= (idx + 1)
        train_ce_loss /= (idx + 1)
        train_gmm_loss /= (idx + 1)
        return train_loss, train_ce_loss, train_gmm_loss

    def eval_model(self, data_loader):
        val_loss = 0
        val_ce_loss = 0
        val_gmm_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                x, y = batch
                x = x.to(self.args.d)
                y = y.to(self.args.d)
                logits = self.model(x)
                ce_loss = self.ce(logits, y)
                gmm_loss = self.compute_gmm_loss(x, logits)
                loss = ce_loss + self.args.alpha * gmm_loss
                val_loss += loss.item()
                val_ce_loss += ce_loss.item()
                val_gmm_loss += gmm_loss.item()
        val_loss /= (idx + 1)
        val_ce_loss /= (idx + 1)
        val_gmm_loss /= (idx + 1)
        return val_loss, val_ce_loss, val_gmm_loss

    def fit(self, train_loader, val_loader):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, amsgrad=True)
        loss = np.zeros((self.args.e, 2, 3))
        counter = 0
        for epoch in range(self.args.e):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss, train_ce_loss, train_gmm_loss = self.train_model(train_loader)
            val_loss, val_ce_loss, val_gmm_loss = self.eval_model(val_loader)
            loss[epoch, 0] = (train_loss, train_ce_loss, train_gmm_loss)
            loss[epoch, 1] = (val_loss, val_ce_loss, val_gmm_loss)
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
        torch.save(self.best_model, "models/{}/gmm_best.pth".format(self.args.dataset))
        torch.save(self.last_model, "models/{}/gmm_last.pth".format(self.args.dataset))
        return loss

    def compute_sample_energy(self, x):
        self.model.load_state_dict(self.last_model)
        self.model.eval()
        with torch.no_grad():            
            x = torch.tensor(x, dtype=torch.float, device=self.args.d)
            logits = self.model(x)            
            gamma = self.softmax(logits)
            phi, mu, cov = compute_params(x, gamma)
            sample_energy = compute_energy(x, phi=phi, mu=mu, cov=cov, size_average=False)
        return sample_energy.cpu().numpy(), logits.cpu().numpy()

    def evaluate(self, x, y, outlier_class):
        sample_energy, logits = self.compute_sample_energy(x)        
        y_pred = logits.argmax(1)

        nclasses = len(np.unique(y))
        target = np.ones(len(logits))
        for i in outlier_class:
            target[y == i] = 0

        # precision recall
        precision, recall, _ = precision_recall_curve(target, sample_energy, pos_label=0)
        aucpr = auc(recall, precision)

        fpr, tpr, _ = roc_curve(target, sample_energy, pos_label=0)
        aucroc = auc(fpr, tpr)
        cm_multi = confusion_matrix(y, y_pred)

        return aucpr, aucroc, cm_multi

    def save_features(self, x, y, savename):
        np.save("{}_x.npy".format(savename), x.cpu().numpy())
        np.save("{}_y.npy".format(savename), y.cpu().numpy())
        return


class FeatureDataset(object):
    def __init__(self, bs):
        self.x_train, self.y_train = self.load_data(split="train")
        self.x_test, self.y_test = self.load_data(split="test")
        self.x_val, self.y_val = self.load_data(split="val")

        train_dataset = MyFeatureDataset(self.x_train, self.y_train, device="cpu")
        val_dataset = MyFeatureDataset(self.x_val, self.y_val, device="cpu")

        # balancing
        labs, counts = np.unique(self.y_train, return_counts=True)

        weights = 1 / counts
        weights /= weights.sum()

        sample_weight = np.zeros(len(self.y_train))
        for i, lab in enumerate(labs):
            mask = self.y_train == lab
            sample_weight[mask] = weights[i]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
        self.train_dataloader = DataLoader(train_dataset, batch_size=bs, sampler=sampler)
        self.val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    def load_data(self, split):
        x = np.load("features/{}_x.npy".format(split))
        y = np.load("features/{}_y.npy".format(split))
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--bs', type=int, default=2048, help="batch size (default 2048)")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
    parser.add_argument('--nh', type=int, default=2, help="hidden size (default 2)")
    parser.add_argument('--nlayers', type=int, default=1, help="number of hidden layers (default 1)")
    parser.add_argument("--do", type=float, default=0., help="dropout value (default 0)")
    parser.add_argument("--wd", type=float, default=0., help="L2 reg value (default 0)")
    parser.add_argument("--dataset", type=str, default="linear", choices=["linear", "macho", "asas", "asas_sn", "toy"], help="dataset name (default linear)")
    parser.add_argument('--patience', type=int, default=2, help="early stopping patience (default 2)")
    parser.add_argument("--alpha", type=float, default=1e-3, help="gmm weight (default 1e-3)")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")
    make_dir("files")
    make_dir("features")

    make_dir("figures/{}".format(args.dataset))
    make_dir("models/{}".format(args.dataset))
    make_dir("files/{}".format(args.dataset))

    make_dir("figures/{}/gmm".format(args.dataset))
    make_dir("models/{}/gmm".format(args.dataset))
    make_dir("files/{}/gmm".format(args.dataset))

    seed_everything()

    dataset = FeatureDataset(args.bs)
    args.nin = dataset.x_train.shape[1]
    args.nout = len(np.unique(dataset.y_train))
    outlier_class = [8]

    gmm = Model(args)
    loss = gmm.fit(dataset.train_dataloader, dataset.val_dataloader)
    plot_loss(loss, "figures/{}/gmm/gmm_loss.png".format(args.dataset), dpi=400)
    aucpr, aucroc, cm_multi = gmm.evaluate(dataset.x_test, dataset.y_test, outlier_class)
    print(aucpr)
    print(aucroc)
    print(cm_multi)
    