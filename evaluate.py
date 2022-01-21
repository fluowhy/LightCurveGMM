import torch
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve
from sklearn import metrics
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import argparse
from scipy.special import softmax

from utils import seed_everything
from utils import plot_confusion_matrix
from utils import plot_single_confusion_matrix
from utils import load_json
from utils import compute_energy
from models import compute_params
from utils import load_yaml
from utils import MyDataset

from datasets import read_and_normalize_data

from datasets import ASASSNDataset
from models import *


def feature_extraction(dataset, model, device, fold):
    softmax = torch.nn.Softmax(dim=1)
    x_train = torch.tensor(dataset.x_train, dtype=torch.float, device=device)
    x_val = torch.tensor(dataset.x_val, dtype=torch.float, device=device)
    m_train = torch.tensor(dataset.m_train, dtype=torch.float, device=device)
    m_val = torch.tensor(dataset.m_val, dtype=torch.float, device=device)
    s_train = torch.tensor(dataset.s_train, dtype=torch.float, device=device)
    s_val = torch.tensor(dataset.s_val, dtype=torch.float, device=device)
    if fold:
        p_train = torch.tensor(dataset.p_train, dtype=torch.float, device=device)
        p_val = torch.tensor(dataset.p_val, dtype=torch.float, device=device)
    else:
        p_train = None
        p_val = None
    seq_len_train = (x_train[:, :, 0] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 0] != 0).sum(dim=1)
    model.eval()
    with torch.no_grad():
        _, train_features, train_logits, _, _, _ = model(x_train, m_train, s_train, seq_len_train, p_train)
        _, val_features, val_logits, _, _, _ = model(x_val, m_val, s_val, seq_len_val, p_val)
    train_features = train_features.cpu().numpy()
    val_features = val_features.cpu().numpy()
    y_prob_train = softmax(train_logits).cpu().numpy()
    y_prob_val = softmax(val_logits).cpu().numpy()
    return train_features, val_features, y_prob_train, y_prob_val


def evaluate_by_batch(x, seq_len, period, bs, model):
    n = int(np.floor(len(x) / bs))
    res = int(len(x) % bs)
    model.train()
    features = list()
    logits = list()
    with torch.no_grad():
        for i in tqdm(range(n)):
            _, tf, tl, _, _, _ = model(x[i * bs:(i + 1) * bs], seq_len[i * bs:(i + 1) * bs], period)
            features.append(tf.detach().cpu().numpy())
            logits.append(tl.detach().cpu().numpy())
        _, tf, tl, _, _, _ = model(x[-res:], seq_len[-res:], period)
        features.append(tf.detach().cpu().numpy())
        logits.append(tl.detach().cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def feature_extraction_asas_sn(dataset, model, device, fold, bs):
    x_test = torch.tensor(dataset.x_test, dtype=torch.float, device=device)
    x_val = torch.tensor(dataset.x_val, dtype=torch.float, device=device)
    #m_test = torch.tensor(dataset.m_test, dtype=torch.float, device=device)
    #m_val = torch.tensor(dataset.m_val, dtype=torch.float, device=device)
    #s_test = torch.tensor(dataset.s_test, dtype=torch.float, device=device)
    #s_val = torch.tensor(dataset.s_val, dtype=torch.float, device=device)
    if fold:
        p_test = torch.tensor(dataset.p_test, dtype=torch.float, device=device)
        p_val = torch.tensor(dataset.p_val, dtype=torch.float, device=device)
    else:
        p_test = None
        p_val = None
    seq_len_test = (x_test[:, :, 2] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 2] != 0).sum(dim=1)

    test_features, test_logits = evaluate_by_batch(x_test, seq_len_test, p_test, bs, model)
    val_features, val_logits = evaluate_by_batch(x_val, seq_len_val, p_val, bs, model)

    y_prob_test = softmax(test_logits, axis=1)
    y_prob_val = softmax(val_logits, axis=1)
    return test_features, val_features, y_prob_test, y_prob_val, test_logits, val_logits


def feature_extraction_ztf(val_dataset, test_dataset, model, device, fold, bs):
    seq_len_val = torch.tensor(val_dataset.sl, dtype=torch.float, device=device)
    x_val = [torch.tensor(xi, dtype=torch.float, device=device) for xi in val_dataset.x]
    x_val = torch.nn.utils.rnn.pad_sequence(x_val, padding_value=0., batch_first=True)
    seq_len_test = torch.tensor(test_dataset.sl, dtype=torch.float, device=device)
    x_test = [torch.tensor(xi, dtype=torch.float, device=device) for xi in test_dataset.x]
    x_test = torch.nn.utils.rnn.pad_sequence(x_test, padding_value=0., batch_first=True)
    
    if fold:
        p_test = torch.tensor(dataset.p_test, dtype=torch.float, device=device)
        p_val = torch.tensor(dataset.p_val, dtype=torch.float, device=device)
    else:
        p_test = None
        p_val = None

    test_features, test_logits = evaluate_by_batch(x_test, seq_len_test, p_test, bs, model)
    val_features, val_logits = evaluate_by_batch(x_val, seq_len_val, p_val, bs, model)
    y_prob_test = softmax(test_logits, axis=1)
    y_prob_val = softmax(val_logits, axis=1)
    return test_features, val_features, y_prob_test, y_prob_val, test_logits, val_logits


def plot_embeddings(emb, y, savename):
    plt.clf()
    for i in range(len(np.unique(y))):
        mask = y == i
        plt.scatter(emb[mask, 0], emb[mask, 1], marker=".", alpha=0.5, label=idx2lab[i])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
    return


def metrics_summary(scores_test, scores_train, y_true, percentile):
    thr = np.percentile(scores_train, percentile)
    y_pred = np.ones(len(scores_test))
    y_pred[scores_test < thr] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return precision, recall, f1


parser = argparse.ArgumentParser(description="autoencoder")
parser.add_argument('--bs', type=int, default=2048, help="batch size (default 2048)")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument('--ngmm', type=int, default=8, help="number of gaussian components (default 8)")
parser.add_argument('--nin', type=int, default=3, help="input size (default 3)")
parser.add_argument('--nh', type=int, default=96, help="hidden size (default 96)")
parser.add_argument('--nl', type=int, default=16, help="hidden size (default 16)")
parser.add_argument('--ne', type=int, default=16, help="estimation network size (default 16)")
parser.add_argument('--nout', type=int, default=1, help="output size (default 1)")
parser.add_argument('--nlayers', type=int, default=2, help="number of hidden layers (default 2)")
parser.add_argument("--do", type=float, default=0.5, help="dropout value (default 0.5)")
parser.add_argument("--fold", action="store_true", help="folded light curves")
parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"], help="rnn architecture (default gru)")
parser.add_argument("--name", type=str, default="linear", choices=["linear", "macho", "asas", "asas_sn", "toy", "ztf"], help="dataset name (default linear)")
parser.add_argument("--config", type=str, default="none", help="config (default none)")
parser.add_argument('--run', type=int, default=0, help="run (default 0)")

args = parser.parse_args()
print(args)

if args.config != "none":
    myargs = load_yaml("config/{}/config_{}.yaml".format(args.name, args.config))
    myargs["name"] = args.name
    myargs["config"] = args.config
    myargs["run"] = args.run
    args = myargs
    args["nin"] = 3
    args["bs"] = 256
    args["d"] = "cuda:0"
else:
    myargs = dict()
    pass


seed_everything(args["run"] * 1111)

if args["name"] == "asas_sn":
    args["ngmm"] = 8
    dataset = ASASSNDataset(args, self_adv=False, oe=False, geotrans=False)
    lab2idx = load_json("../datasets/asas_sn/lab2idx.json")
    idx2lab = list(lab2idx.keys())
elif args["name"] == "toy":
    from toy_dataset import ToyDataset

    dataset = ToyDataset(args, val_size=0.1, sl=64)
    lab2idx = dataset.lab2idx
    idx2lab = list(lab2idx.keys())
elif args["name"] == "ztf":
    print(args["name"])
    x_train, x_val, x_test, y_train, y_val, y_test, _, lab2idx, idx2lab, lab2newlab, newlab2lab = read_and_normalize_data("../datasets/ztf/train_data_band_1.npz")
    outlier_label = load_json("../datasets/ztf/outlier_class.json")
    oc = [int(lab2idx[lab]) for lab in lab2idx if outlier_label[lab] == "outlier"]
    labs, counts = np.unique(y_train, return_counts=True)

    # train_dataset = MyDataset(x_train, y_train, device=args["d"])
    val_dataset = MyDataset(x_val, y_val, device=args["d"])
    test_dataset = MyDataset(x_test, y_test, device=args["d"])

    args["ngmm"] = val_dataset.n_inlier_classes
else:
    dataset = LightCurveDataset(args["name"], fold=args["fold"], bs=args["bs"], device=args["d"], eval=True)
    lab2idx = load_json("processed_data/{}/lab2idx.json".format(args["name"]))
    idx2lab = list(lab2idx.keys())

if args["arch"] == "gru":
    model = GRUGMM(args["nin"], args["nh"], args["nl"], args["ne"], args["ngmm"], args["nout"], args["nlayers"], args["do"], args["fold"])
elif args["arch"] == "lstm":
    model = LSTMGMM(args["nin"], args["nh"], args["nl"], args["ne"], args["ngmm"], args["nout"], args["nlayers"], args["do"], args["fold"])

fig_path = "figures/{}/config_{}".format(args["name"], args["config"])
model.to(args["d"])
state_dict = torch.load("models/{}/config_{}/best_model.pth".format(args["name"], args["config"]), map_location=args["d"])
model.load_state_dict(state_dict)

if args["name"] == "asas_sn" or args["name"] == "toy":
    test_features, val_features, y_prob_test, y_prob_val, test_logits, val_logits = feature_extraction_asas_sn(dataset, model, args["d"], fold=args["fold"], bs=args["bs"])
elif args["name"] == "ztf":
    test_features, val_features, y_prob_test, y_prob_val, test_logits, val_logits = feature_extraction_ztf(val_dataset, test_dataset, model, args["d"], fold=args["fold"], bs=args["bs"])
else:
    train_features, val_features, y_prob_train, y_prob_val = feature_extraction(dataset, model, args["d"], fold=args["fold"])



if args["name"] == "asas_sn" or args["name"] == "toy":
    y_val = dataset.y_val
    reducer = umap.UMAP()

    y_test = dataset.y_test
    reducer.fit(val_features)
    test_embedding = reducer.transform(test_features)
    plot_embeddings(test_embedding, y_test, "{}/umap_{}_run_{}.png".format(fig_path, args["arch"], args["run"]))

    if args["name"] == "asas_sn":
        mask = y_test != 8
    elif args["name"] == "toy":
        mask = (y_test != 3) & (y_test != 4)
    target = y_test[mask]
    y_prob = y_prob_test[mask]
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = (target == y_pred).sum() / len(target)
    cm = confusion_matrix(target, y_pred)

    if args["name"] == "asas_sn":
        plot_single_confusion_matrix(cm, idx2lab[:-1], args["name"], "{}/{}_cm_norm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=True)
        plot_single_confusion_matrix(cm, idx2lab[:-1], "{}, accuracy: {:.4f}".format(args["name"], accuracy), "{}/{}_cm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=False)
    elif args["name"] == "toy":
        plot_confusion_matrix(cm, idx2lab[:-2], args["name"], "{}/{}_cm_norm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=True)
        plot_confusion_matrix(cm, idx2lab[:-2], "{}, accuracy: {:.4f}".format(args["name"], accuracy), "{}/{}_cm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=False)

elif False:
    y_val = dataset.y_val
    reducer = umap.UMAP()
    y_train = dataset.y_train
    reducer.fit(train_features)    
    val_embedding = reducer.transform(val_features)
    plot_embeddings(val_embedding, y_val, "{}/umap_{}_run_{}.png".format(fig_path, args["arch"], args["run"]))

    y_pred = np.argmax(y_prob_val, axis=1)
    accuracy = (y_val == y_pred).sum() / len(y_val)
    cm = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(cm, idx2lab, args["name"], "{}/{}_cm_norm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=True)
    plot_confusion_matrix(cm, idx2lab, "{}, accuracy: {:.4f}".format(args["name"], accuracy), "{}/{}_cm_run_{}.png".format(fig_path, args["arch"], args["run"]), normalize=False)

softmax = torch.nn.Softmax(dim=1)

if args["name"] == "asas_sn" or args["name"] == "ztf":
    z_val = torch.tensor(val_features, dtype=torch.float, device=args["d"])
    z_test = torch.tensor(test_features, dtype=torch.float, device=args["d"])
    logits_val = torch.tensor(val_logits, dtype=torch.float, device=args["d"])
    logits_test = torch.tensor(test_logits, dtype=torch.float, device=args["d"])
    
    phi_val, mu_val, cov_val = compute_params(z_val, softmax(logits_val))
    # phi_test, mu_test, cov_test = compute_params(z_test, softmax(logits_test))
    
    val_energy = compute_energy(z_val, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()
    test_energy = compute_energy(z_test, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()

    labels = np.ones(len(y_test))
    if args["name"] == "asas_sn":
        labels[y_test == 8] = 0  # class 8 is outlier
    elif args["name"] == "toy":
        labels[y_test == 3] = 0
        labels[y_test == 4] = 0
    elif args["name"] == "ztf":
        y_test_real = np.array([newlab2lab[lab] for lab in y_test])
        for oci in oc:
            labels[y_test_real == oci] = 0
    scores_in = test_energy[labels == 1]
    scores_out = test_energy[labels == 0]
    average_precision = (labels == 0).sum() / len(labels)

    score_min = min(test_energy.min(), val_energy.min())
    score_max = max(test_energy.max(), val_energy.max())
    n_bins = 100
    bins = np.linspace(score_min, score_max, n_bins)

    pr95, re95, f195 = metrics_summary(test_energy, val_energy, labels, 95)
    pr80, re80, f180 = metrics_summary(test_energy, val_energy, labels, 80)

    precision, recall, _ = precision_recall_curve(labels, test_energy, pos_label=0)
    fpr, tpr, _ = metrics.roc_curve(labels, test_energy, pos_label=0)

    aucroc = metrics.auc(fpr, tpr)
    aucpr = metrics.auc(recall, precision)

    df = pd.DataFrame()
    df["percentile"] = [95, 80]
    df["precision"] = [pr95, pr80]
    df["recall"] = [re95, re80]
    df["f1"] = [f195, f180]
    df["aucroc"] = [aucroc, aucroc]
    df["aucpr"] = [aucpr, aucpr]
    print(df)
    df.to_csv("files/{}/config_{}/summary_run_{}.csv".format(args["name"], args["config"], args["run"]), index=False)

    plt.clf()
    plt.hist(val_energy, bins=bins, color="black", label="val", histtype="step")
    plt.hist(scores_in, bins=bins, color="navy", label="inlier", histtype="step")
    plt.hist(scores_out, bins=bins, color="red", label="outlier", histtype="step")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("energy")
    plt.ylabel("counts")
    plt.tight_layout()
    plt.savefig("{}/scores_run_{}.png".format(fig_path, args["run"]), dpi=200)

    plt.clf()
    plt.title("AUCPR: {:.4f}".format(aucpr))
    plt.plot(recall, precision, color="red")
    plt.axhline(average_precision, color="black", linestyle="--")
    plt.grid()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("{}/precision_recall_curve_run_{}.png".format(fig_path, args["run"]), dpi=200)

    plt.clf()
    plt.title("AUCROC: {:.4f}".format(aucroc))
    plt.plot(fpr, tpr, color="red")
    aux = np.linspace(0, 1, 10)
    plt.plot(aux, aux, color="black", linestyle="--")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("{}/roc_curve_run_{}.png".format(fig_path, args["run"]), dpi=200)
