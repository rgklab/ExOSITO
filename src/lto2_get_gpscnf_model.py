import numpy as np
import random
import torch
import os, sys

# from data_loader.datasets.labtest_pred import LabValueDatasetMIMIC
from data_loader.datasets.labtest_offp import LabGPSDatasetMIMIC
from data_loader.data_loaders import PredictionLoader
import argparse
import pickle, copy

# from exp.labpoli_trainer import LabOrderTrainer
from datetime import datetime
from models.conditionnormflow import ConditionalNormalizingFlow
import pyro, tqdm


def main():
    data_dir = "/datasets/physionet.org/"
    mort_dir = os.path.join(data_dir, "laborder")
    imputation = "clip"
    model = "PatchTST"
    p = "train"
    dp = os.path.join(mort_dir, f"offpdata_{imputation}{model}_{p}.pkl")
    tr_dataset = LabGPSDatasetMIMIC(path=dp, split=p)
    p = "val"
    dp = os.path.join(mort_dir, f"offpdata_{imputation}{model}_{p}.pkl")
    val_dataset = LabGPSDatasetMIMIC(path=dp, split=p)

    b = tr_dataset[0]
    x, ty, py, t = b
    print(x.size(), ty.size(), py.size(), t.size())
    print(len(tr_dataset), len(val_dataset))  # , len(te_dataset))
    batch_size = 4096
    tr_loader = PredictionLoader(tr_dataset, batch_size, True, sampler=None)
    val_loader = PredictionLoader(val_dataset, batch_size, False, sampler=None)

    n_epochs = 200
    init_lr = 1e-4
    random_seed = 2024
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    dim_treat = 10
    dim_cov = int(72 * 71)  # int(1 * 21)  # x.size()[0] * x.size()[1]
    print(dim_cov)
    hidden_dim = 50
    flow_length = 1
    count_bins = 5
    bound = 0.5

    gpu_num = 3
    device = torch.device("cuda:{}".format(gpu_num))
    ckptdir = "/runs/gpscnf"
    time_str = datetime.now().strftime("%b%d_%H-%M-%S_lab")
    save_path = ckptdir  # os.path.join(ckptdir, )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = ConditionalNormalizingFlow(
        input_dim=dim_treat,
        # split_dim=dim_treat - 1,
        context_dim=dim_cov,
        hidden_dim=hidden_dim,
        flow_length=flow_length,
        count_bins=count_bins,
        bound=bound,
        device=device
        # use_cuda=False,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    # train
    pyro.clear_param_store()
    epochs = tqdm.trange(n_epochs)
    best_val_loss = np.inf
    patience = 50
    p = 0
    best_model = model
    for epoch in epochs:
        model.train()
        running_loss = 0
        val_loss = 0
        for idx, batch in enumerate(tr_loader):
            x, ty, py, t = batch
            inp = torch.cat((x[:, :, :21], ty[:, :, :21]), dim=1)
            inp = torch.cat((x, ty), dim=1)
            x = inp
            # print(x.size())
            x = x.view(x.size()[0], -1)
            # print(x.size())

            t -= 0.5
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)
            optimizer.zero_grad()

            t, x = t.to(device), x.to(device)
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            model.flow_dist.clear_cache()
            running_loss += float(loss)
        running_loss /= len(tr_loader)

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                x, ty, py, t = batch
                inp = torch.cat((x[:, :, :21], ty[:, :, :21]), dim=1)
                inp = torch.cat((x, ty), dim=1)
                x = inp
                x = x.view(x.size()[0], -1)

                # center t
                t -= 0.5
                t, x = t.to(device), x.to(device)
                t += torch.randn_like(t) * 0.1
                t = torch.clip(t, -0.5, 0.5)

                loss = -model.log_prob(t, x).mean()
                model.flow_dist.clear_cache()
                val_loss += float(loss)
            val_loss /= len(val_loader)

        epochs.set_description(
            "Train Loss: {:.3f} --- Validation Loss: {:.3f}".format(
                running_loss, val_loss
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            p = 0
        else:
            p += 1
        if p >= patience:
            print(f"Early stopping at epoch: {epoch}, patience: {patience}")
            break
    torch.save(
        best_model.state_dict(), os.path.join(save_path, f"full_model_density.pt")
    )


if __name__ == "__main__":
    print("train and save cnf")
    main()
