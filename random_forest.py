import torch
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pdb

from utils import seed_everything, plot_confusion_matrix, load_json
from datasets import LightCurveDataset
from models import *


def feature_extraction(dataset, model, device):
    x_train = torch.tensor(dataset.x_train, dtype=torch.float, device=device)
    x_val = torch.tensor(dataset.x_val, dtype=torch.float, device=device)
    seq_len_train = (x_train[:, :, 2] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 2] != 0).sum(dim=1)
    model.eval()
    with torch.no_grad():
        _, train_features = model(x_train, seq_len_train)
        _, val_features = model(x_val, seq_len_val)
    train_features = train_features.cpu().numpy()
    val_features = val_features.cpu().numpy() 
    return train_features, val_features


seed_everything()

name = "linear"
bs = 256
device = "cpu"
arch = "gru"
nin = 3
nh = 96
nl = 64
nout = 1
nlayers = 2
do = 0.25

dataset = LightCurveDataset(name, fold=True, bs=bs, device=device, eval=True)
if arch == "gru":
    model = GRUAE(nin, nh, nl, nout, nlayers, do)
elif arch == "lstm":
    model = LSTMAE(nin, nh, nl, nout, nlayers, do)

model.to(device)
state_dict = torch.load("models/{}.pth".format(arch), map_location=device)
model.load_state_dict(state_dict)

train_features, val_features = feature_extraction(dataset, model, device)

train_features = np.concatenate((train_features, dataset.m_train[:, np.newaxis], dataset.s_train[:, np.newaxis], dataset.p_train[:, np.newaxis]), axis=1)
val_features = np.concatenate((val_features, dataset.m_val[:, np.newaxis], dataset.s_val[:, np.newaxis], dataset.p_val[:, np.newaxis]), axis=1)

y_train = dataset.y_train
y_val = dataset.y_val

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True)
skf.get_n_splits(train_features, y_train)
print(skf)

# distributions = dict(C=uniform(loc=0, scale=4),
#                      penalty=['l2', 'l1'])

parameters = {
    "n_estimators": [50, 100, 250],
    "criterion": ["gini", "entropy"],
    "max_features": [3, 6, 12, 18],
    "min_samples_leaf": [1, 2, 3]
    }

rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv=skf, n_jobs=2)
clf.fit(train_features, y_train)

lab2idx = load_json("processed_data/{}/lab2idx.json".format(name))
labels = list(lab2idx.keys())
y_pred = clf.predict(val_features)
accuracy = (y_val == y_pred).sum() / len(y_val)
cm = confusion_matrix(y_val, y_pred)
plot_confusion_matrix(cm, labels, name, "figures/linear_{}_cm_norm.png".format(arch), normalize=True)
plot_confusion_matrix(cm, labels, "{}, accuracy: {:.4f}".format(name, accuracy), "figures/linear_{}_cm.png".format(arch), normalize=False)

idx = clf.best_index_
mean_acc = clf.cv_results_["mean_test_score"][idx]
std_acc = clf.cv_results_["std_test_score"][idx]
print("{} {} k-FCV accuracy: {:.2f} +- {:.2f}".format(name, arch, mean_acc * 100, std_acc * 100))
