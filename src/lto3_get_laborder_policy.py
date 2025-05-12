import numpy as np
import random
import torch
import os, sys

# from data_loader.datasets.labtest_pred import LabValueDatasetMIMIC
from data_loader.datasets.labtest_offp import LabPiDatasetMIMIC

from data_loader.data_loaders import PredictionLoader
import argparse
import pickle, copy
from exp.labpoli_trainer import LabOrderTrainer
from datetime import datetime


def main():
    print("Learn order policy")
    parser = argparse.ArgumentParser(description="Lab test order policy learning")

    parser.add_argument("--random_seed", type=int, default=2024, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="init lr")
    # model & forecasting
    parser.add_argument(
        "--model", type=str, default="PatchTSMixer", choices=["Linear", "PatchTSMixer"]
    )
    parser.add_argument("--pred_model", type=str, default="PatchTST")
    parser.add_argument("--lradj", type=str, default="type3", help="adjust lr")
    parser.add_argument("--output_dim", type=int, default=71, help="outdim for lstm")
    parser.add_argument("--layer_dim", type=int, default=3, help="num layers for lstm")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden for lstm")
    parser.add_argument("--seq_len", type=int, default=48, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=24, help="pred sequence length")
    parser.add_argument("--x_len", type=int, default=72, help="seq len for policy")
    parser.add_argument("--y_len", type=int, default=10, help="num test for policy")
    parser.add_argument("--adjust", action="store_true", help="use gps", default=False)

    # gpu
    parser.add_argument("--test_only", action="store_true", default=False, help="")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=2, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="2,3", help="device ids")
    parser.add_argument("--test_flop", action="store_true", default=False, help="")

    # exp variables
    parser.add_argument("--use_gps", action="store_true", default=False)
    parser.add_argument("--use_ty", action="store_true", default=False)
    parser.add_argument("--use_xyr", action="store_true", default=False)
    parser.add_argument("--use_scost", action="store_true", default=False)
    parser.add_argument("--wDX", type=float, default=1.0)
    parser.add_argument("--wLB", type=float, default=1.0)
    parser.add_argument("--wC", type=float, default=1.0)

    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if True:
        data_dir = "/datasets/physionet.org/"
        mort_dir = os.path.join(data_dir, "laborder")
        imputation = "clip"
        model = "PatchTST"
        p = "train"
        dp = os.path.join(mort_dir, f"offpdata_{imputation}{model}_{p}.pkl")
        tr_dataset = LabPiDatasetMIMIC(path=dp, split=p)
        p = "val"
        dp = os.path.join(mort_dir, f"offpdata_{imputation}{model}_{p}.pkl")
        val_dataset = LabPiDatasetMIMIC(path=dp, split=p)
        p = "test"
        dp = os.path.join(mort_dir, f"offpdata_{imputation}{model}_{p}.pkl")
        te_dataset = LabPiDatasetMIMIC(path=dp, split=p)
        print(len(val_dataset[4]))

        labidx_map = val_dataset.labidx_map
        batch_size = 4096
        tr_loader = PredictionLoader(tr_dataset, batch_size, True, sampler=None)
        val_loader = PredictionLoader(val_dataset, batch_size, False, sampler=None)
        te_loader = PredictionLoader(te_dataset, batch_size, False, sampler=None)
        for batch in tr_loader:
            x, true_y, pred_y, lower, lowerxy, t, upper = batch
            print(
                x.size(),
                true_y.size(),
                pred_y.size(),
                lower.size(),
                lowerxy.size(),
                t.size(),
                upper.size(),
            )
            break

        config = dict()
        config["ckptdir"] = "/runs/laborderpolicy"
        config["dataset"] = "mimic"
        config["lr"] = args.lr
        config["lradj"] = args.lradj
        config["pct_start"] = 0.3
        config["patience"] = 7
        config["epochs"] = 40
        config["batch_size"] = batch_size
        config["labidx_map"] = labidx_map


    time_str = datetime.now().strftime("%b%d_%H-%M-%S_lab")
    identifier = f"{config['dataset']}_{args.model}_{config['lr']:.0e}_{args.lradj}"
    if args.use_gps:
        identifier += f"_wgps"
    else:
        identifier += f"_wogps"
    if args.use_scost:
        identifier += f"_sc"
    else:
        identifier += f"_rc"
    if args.use_xyr:
        identifier += f"_xyr"
    else:
        identifier += "_xr"
    if args.use_ty:
        identifier += "_ty"
    else:
        identifier += "_py"
    identifier += f"_wdx{args.wDX}_wlb{args.wLB}_wc{args.wC}"
    identifier += f"_{args.random_seed}"
    print(identifier)

    trainer = LabOrderTrainer(config, tr_loader, val_loader, te_loader, args=args)
    if not args.test_only:
        trainer.train(identifier)
    else:
        trainer.test(identifier)


if __name__ == "__main__":
    """
    Given patient stays, learn a policy that can order the blood test for the next 24 hours
    """

    # data will z-score/minmax normalized first
    main()
