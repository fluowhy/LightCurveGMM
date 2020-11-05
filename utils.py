import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import pdb


def plot_confusion_matrix(cms, labels, title, savename, normalize, ni=None):
    nc = len(labels)
    if not ni:
        ni = nc    
    cms_mean = cms.mean(axis=0)
    cms_std = cms.std(axis=0)
    if normalize:        
        total = cms[0].sum(axis=1, keepdims=True)
        cms_mean = cms_mean / total
        cms_std = cms_std / total
        vmax = 1
        template = "{:.2f} \n +- {:.2f}"
    else:
        vmax = cms_mean.max()
        template = "{:.0f} \n +- {:.0f}"
    thr = vmax * 0.5
    fig, ax = plt.subplots()
    im = ax.imshow(cms_mean, cmap="YlGn", aspect="equal", vmin=0, vmax=vmax)
    for i in range(nc):
        for j in range(ni):
            if cms_mean[i, j] > thr:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, template.format(cms_mean[i, j], cms_std[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xticks(np.arange(ni), labels[:ni], rotation=45)
    plt.yticks(np.arange(nc), labels)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
    return


def calculate_seq_len(x):
    sl = (x[:, :, 2] != 0).sum(1)
    return sl


def compute_params(z, gamma):
    # source https://github.com/danieltan07/dagmm/blob/master/model.py
    N = gamma.size(0)
    # K
    sum_gamma = torch.sum(gamma, dim=0)

    # K
    phi = sum_gamma / N

    # K x D
    mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
    # z = N x D
    # mu = K x D
    # gamma N x K

    # z_mu = N x K x D
    z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

    # z_mu_outer = N x K x D x D
    z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

    # K x D x D
    cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

    return phi, mu, cov


def compute_energy(z, phi=None, mu=None, cov=None, size_average=True):
    k, D, _ = cov.size()

    z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

    eps = 1e-12
    cte = D * np.log(2 * np.pi)
    eye = (torch.eye(D, device=cov.device) * eps).unsqueeze(0)
    eye = eye.repeat(k, 1, 1)
    cov = cov + eye
    cov_inverse = torch.inverse(cov)
    # det_cov = (0.5 * (cte + torch.logdet(cov)))
    det_cov = torch.det(cov)

    # cov_inverse = []
    # det_cov = []
    # cov_diag = 0
    # eps = 1e-12
    # for i in range(k):
    #     # K x D x D
    #     cov_k = cov[i] + torch.eye(D) * eps
    #     cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

    #     #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
    #     det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
    #     cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

    # # K x D x D
    # cov_inverse = torch.cat(cov_inverse, dim=0)
    # # K
    # det_cov = torch.cat(det_cov).cuda()
    #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

    # N x K
    exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
    # for stability (logsumexp)
    max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

    exp_term = torch.exp(exp_term_tmp - max_val)

    # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
    sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
    # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


    if size_average:
        sample_energy = torch.mean(sample_energy)

    return sample_energy


def compute_energy_mine(z, phi=None, mu=None, cov=None, logits=None):
    # source https://github.com/danieltan07/dagmm/blob/master/model.py
    if phi is None or mu is None or cov is None:
        phi, mu, cov = compute_params(z, logits)

    k, D, _ = cov.size()

    z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

    eps = 1e-12
    cte = D * np.log(2 * np.pi)
    eye = (torch.eye(D, device=cov.device) * eps).unsqueeze(0)
    eye = eye.repeat(k, 1, 1)
    cov = cov + eye
    cov_inverse = torch.inverse(cov)
    det_cov = (0.5 * (cte + torch.logdet(cov)))
    # if torch.isnan(det_cov.sum()):
        # det_cov = torch.ones_like(det_cov) * eps
    # det_cov = 0.5 * torch.logdet(cov)
    if torch.isnan(det_cov.sum()):
        mask = det_cov != det_cov
        det_cov[mask] = det_cov[~ mask].mean()

    # N x K
    exp_term_tmp = - 0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
    log_norm_prob_dens = exp_term_tmp - det_cov.unsqueeze(0)
    # for stability (logsumexp)
    # max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
    # exp_term = torch.exp(exp_term_tmp - max_val)
    # max_val, _ = exp_term_tmp.clamp(min=0).max(dim=1, keepdim=True)
    # exp_term = torch.exp(exp_term_tmp - max_val) / torch.sqrt(det_cov).unsqueeze(0)
    # pdb.set_trace()
    exp_term = log_norm_prob_dens.exp()

    arg = torch.sum(phi.unsqueeze(0) * exp_term, dim=1)
    # sample_energy = - max_val.squeeze() - torch.log(arg + eps)
    sample_energy = 1 / (arg + eps)
    sample_energy = sample_energy.clamp(max=1e4)
    # sample_energy = - torch.log(arg + eps)
    
    if torch.isnan(sample_energy.mean()):
        pdb.set_trace()

    return sample_energy


def distances(x, x_pred):
    mask = (x[:, :, 0] != 0) * 1.
    x = x[:, :, 1]
    x_norm = (x.pow(2) * mask).sum(1, keepdim=True)
    x_pred_norm = (x_pred.pow(2) * mask).sum(1, keepdim=True)
    euclidean_distance = ((x_pred - x).pow(2) * mask).sum(dim=1, keepdim=True) / x_norm
    cosine_distance = (x * x_pred * mask).sum(1, keepdim=True) / x_norm / x_pred_norm
    return euclidean_distance, cosine_distance


def save_json(data, savename):
    with open(savename, "w") as fp:
        json.dump(data, fp)


def load_json(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


class WMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-10, nc=3):
        super(WMSELoss, self).__init__()
        self.eps = eps
        self.loss_fun = self.wmse_3c if nc == 3 else self.wmse_2c

    def wmse_3c(self, x, x_pred, seq_len):
        mask = (x[:, :, 2] != 0) * 1.
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def wmse_2c(self, x, x_pred, seq_len):
        mask = (x[:, :, 0] != 0) * 1.
        wmse = ((x_pred - x[:, :, 1]).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def forward(self, x, x_pred, seq_len):
        wmse = self.loss_fun(x, x_pred, seq_len)
        return wmse


class MyDataset(Dataset):
    def __init__(self, x, y, z, p, device="cpu"):
        self.n, _, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)
        self.p = torch.tensor(p, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index], self.p[index]

    def __len__(self):
        return self.n


class MyFeatureDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.n, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


def seed_everything(seed=1234):
    # https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # True before
        torch.backends.cudnn.benchmark = True  # False before
    os.environ["PYTHONHASHSEED"] = str(seed)
    return


def count_parameters(model):
    # TODO: add docstring
    """
    Parameters
    ----------
    model
    Returns
    -------
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    