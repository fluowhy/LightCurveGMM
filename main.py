import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import compute_energy
from utils import make_dir
from utils import load_json
from utils import seed_everything
from utils import load_yaml
from utils import get_data_loaders
from utils import od_metrics
from utils import save_yaml
from utils import plot_loss

from models import compute_params

from gmm import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="asas",
        choices=["asas", "linear", "macho", "asas_sn", "ztf_transient", "ztf_stochastic", "ztf_periodic"],
        help="dataset name (default asas)"
    )
    parser.add_argument('--config', type=int, default=0, help="config number (default 0)")
    parser.add_argument('--device', type=str, default="cpu", help="device (default cpu)")
    parser.add_argument('--oc', type=int, default=0, help="outlier class (default 0)")
    args = parser.parse_args()
    print(args)

    make_dir("figures")
    make_dir("models")
    make_dir("files")

    make_dir("figures/{}".format(args.dataset))
    make_dir("models/{}".format(args.dataset))
    make_dir("files/{}".format(args.dataset))

    models_dir = f"models/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    figures_dir = f"figures/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    files_dir = f"files/od/{args.dataset}/config_{args.config}/oc_{args.oc}"

    make_dir(figures_dir)
    make_dir(files_dir)
    make_dir(models_dir)

    seed_everything()

    config_args = load_yaml(f"config/config_{args.config}.yaml")
    config_args["config"] = args.config
    config_args["e"] = args.e
    config_args["d"] = args.device

    if args.dataset == "asas_sn":
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"])
        outlier_class = [8]
    elif "ztf" in args.dataset:
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"])
        outlier_labels = load_json("../datasets/ztf/cl/transient/lab2out.json")
        outlier_labels = [key for key in outlier_labels if outlier_labels[key] == "outlier"]
        lab2idx = load_json("../datasets/ztf/cl/transient/lab2idx.json")
        outlier_class = [int(lab2idx[lab]) for lab in outlier_labels]
    elif args.dataset == "asas":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"], args.oc)
    elif args.dataset == "linear":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config_args["bs"], config_args["d"], args.oc)
    
    config_args["nin"] = trainloader.dataset.x[0].shape[1]
    config_args["ngmm"] = len(np.unique(testloader.dataset.y))
    print(config_args)

    aegmm = Model(config_args, outlier_class, models_dir)
    train_loss, val_loss = aegmm.fit(trainloader, valloader, config_args)
    plot_loss(train_loss, val_loss, "figures/{}/loss.png".format(args.dataset))
    aegmm.load_model()

    # evaluation
    softmax = torch.nn.Softmax(dim=1)

    feat_train, y_train, logits_train = aegmm.compute_features(trainloader, is_dataloader=True)
    feat_val, y_val, logits_val = aegmm.compute_features(valloader, is_dataloader=True)
    feat_test, y_test, logits_test = aegmm.compute_features(testloader, is_dataloader=True)

    phi_val, mu_val, cov_val = compute_params(feat_val, softmax(logits_val))
    val_energy = compute_energy(feat_val, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()
    test_energy = compute_energy(feat_test, phi=phi_val, mu=mu_val, cov=cov_val, size_average=False).cpu().numpy()

    y_target = np.ones(len(y_test))

    for oc in outlier_class:
        y_target[y_test == oc] = 0
    
    aucpr, _, _, aucroc, _, _ = od_metrics(test_energy, y_target, split=True, n_splits=100)

    data = dict(
        aucpr=float(np.mean(aucpr)),
        aucpr_std=float(np.std(aucpr)),
        aucroc=float(np.mean(aucroc)),
        aucroc_std=float(np.std(aucroc)),
        val_acc=float(((logits_val.argmax(1).cpu() == y_val).sum() / len(y_val)).item()),
        dataset=args.dataset,
        oc=args.oc,
        config=args.config
    )

    save_yaml(data, f"{files_dir}/metrics.yaml")
    np.savez_compressed(f"{files_dir}/scores.npz", scores=test_energy)
