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

from utils import seed_everything, plot_confusion_matrix, load_json, compute_energy, compute_params
from datasets import LightCurveDataset, ASASSNDataset
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
    seq_len_train = (x_train[:, :, 2] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 2] != 0).sum(dim=1)
    model.eval()
    with torch.no_grad():
        _, train_features, train_logits, _, _, _ = model(x_train, m_train, s_train, seq_len_train, p_train)
        _, val_features, val_logits, _, _, _ = model(x_val, m_val, s_val, seq_len_val, p_val)
    train_features = train_features.cpu().numpy()
    val_features = val_features.cpu().numpy()
    y_prob_train = softmax(train_logits).cpu().numpy()
    y_prob_val = softmax(val_logits).cpu().numpy()
    return train_features, val_features, y_prob_train, y_prob_val


def feature_extraction_asas_sn(dataset, model, device, fold):
    softmax = torch.nn.Softmax(dim=1)
    x_test = torch.tensor(dataset.x_test, dtype=torch.float, device=device)
    x_val = torch.tensor(dataset.x_val, dtype=torch.float, device=device)
    m_test = torch.tensor(dataset.m_test, dtype=torch.float, device=device)
    m_val = torch.tensor(dataset.m_val, dtype=torch.float, device=device)
    s_test = torch.tensor(dataset.s_test, dtype=torch.float, device=device)
    s_val = torch.tensor(dataset.s_val, dtype=torch.float, device=device)
    if fold:
        p_test = torch.tensor(dataset.p_test, dtype=torch.float, device=device)
        p_val = torch.tensor(dataset.p_val, dtype=torch.float, device=device)
    else:
        p_test = None
        p_val = None
    seq_len_test = (x_test[:, :, 2] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 2] != 0).sum(dim=1)
    model.eval()
    with torch.no_grad():
        _, test_features, test_logits, _, _, _ = model(x_test, m_test, s_test, seq_len_test, p_test)
        _, val_features, val_logits, _, _, _ = model(x_val, m_val, s_val, seq_len_val, p_val)
    test_features = test_features.cpu().numpy()
    val_features = val_features.cpu().numpy()
    y_prob_test = softmax(test_logits).cpu().numpy()
    y_prob_val = softmax(val_logits).cpu().numpy()
    test_logits = test_logits.cpu().numpy()
    val_logits = val_logits.cpu().numpy()
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


seed_everything()

name = "asas_sn"
bs = 256
device = "cpu"
arch = "gru"
nin = 3
nh = 96
nl = 16
ne = 16
ngmm = 8
nout = 1
nlayers = 2
do = 0.25
fold = True

if name == "asas_sn":    
    dataset = ASASSNDataset(fold=fold, bs=bs, device=device, eval=True)
    lab2idx = load_json("../datasets/asas_sn/lab2idx.json")
    idx2lab = list(lab2idx.keys())
else:
    dataset = LightCurveDataset(name, fold=fold, bs=bs, device=device, eval=True)
    lab2idx = load_json("processed_data/{}/lab2idx.json".format(name))
    idx2lab = list(lab2idx.keys())

if arch == "gru":
    model = GRUGMM(nin, nh, nl, ne, ngmm, nout, nlayers, do, fold)
elif arch == "lstm":
    model = LSTMGMM(nin, nh, nl, ne, ngmm, nout, nlayers, do, fold)

model.to(device)
state_dict = torch.load("models/{}/{}.pth".format(name, arch), map_location=device)
model.load_state_dict(state_dict)

if name == "asas_sn":
    test_features, val_features, y_prob_test, y_prob_val, test_logits, val_logits = feature_extraction_asas_sn(dataset, model, device, fold=fold)
else:
    train_features, val_features, y_prob_train, y_prob_val = feature_extraction(dataset, model, device, fold=fold)

y_val = dataset.y_val
reducer = umap.UMAP()

if name == "asas_sn":
    y_test = dataset.y_test
    reducer.fit(val_features)
    test_embedding = reducer.transform(test_features)
    plot_embeddings(test_embedding, y_test, "figures/{}/umap_{}.png".format(name, arch))

    mask = y_test != 8
    target = y_test[mask]
    y_prob = y_prob_test[mask]
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = (target == y_pred).sum() / len(target)
    cm = confusion_matrix(target, y_pred)

    plot_confusion_matrix(cm, idx2lab[:-1], name, "figures/{}/linear_{}_cm_norm.png".format(name, arch), normalize=True)
    plot_confusion_matrix(cm, idx2lab[:-1], "{}, accuracy: {:.4f}".format(name, accuracy), "figures/{}/linear_{}_cm.png".format(name, arch), normalize=False)

else:
    y_train = dataset.y_train
    reducer.fit(train_features)    
    val_embedding = reducer.transform(val_features)
    plot_embeddings(val_embedding, y_val, "figures/{}/umap_{}.png".format(name, arch))

    y_pred = np.argmax(y_prob_val, axis=1)
    accuracy = (y_val == y_pred).sum() / len(y_val)
    cm = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(cm, idx2lab, name, "figures/{}/linear_{}_cm_norm.png".format(name, arch), normalize=True)
    plot_confusion_matrix(cm, idx2lab, "{}, accuracy: {:.4f}".format(name, accuracy), "figures/{}/linear_{}_cm.png".format(name, arch), normalize=False)

if name == "asas_sn":
    z_val = torch.tensor(val_features, dtype=torch.float, device=device)
    z_test = torch.tensor(test_features, dtype=torch.float, device=device)
    logits_val = torch.tensor(val_logits, dtype=torch.float, device=device)
    logits_test = torch.tensor(test_logits, dtype=torch.float, device=device)

    phi_val, mu_val, cov_val = compute_params(z_val, logits_val)
    phi_test, mu_test, cov_test = compute_params(z_test, logits_test)

    val_energy = compute_energy(z_val, phi_val, mu_val, cov_val).cpu().numpy()
    test_energy = compute_energy(z_test, phi_test, mu_test, cov_test).cpu().numpy()

# new code
    labels = np.ones(len(y_test))
    labels[y_test == 8] = 0  # class 8 is outlier

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
    df.to_csv("summary.csv", index=False)

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
    plt.savefig("figures/asas_sn/scores.png", dpi=200)

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
    plt.savefig("figures/asas_sn/precision_recall_curve.png", dpi=200)

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
    plt.savefig("figures/asas_sn/roc_curve.png", dpi=200)
