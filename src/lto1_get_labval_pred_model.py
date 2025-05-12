import numpy as np
import random
import torch
import os, sys
from data_loader.datasets.labtest_pred import LabValueDatasetMIMIC
from data_loader.data_loaders import PredictionLoader
import argparse
import pickle, copy
from exp.labval_trainer import LabValTrainer
from datetime import datetime
import pandas as pd
import time, h5py, json


def load_dict_from_hdf5(filename):
    data_dict = {}
    with h5py.File(filename, "r") as hf:
        # Load datasets
        for key in hf.keys():
            data = hf[key][:]
            # Try converting numeric datasets back to arrays directly
            try:
                data_dict[key] = data
            except Exception as e:
                print(f"Error loading dataset {key}: {e}")

        # Load attributes and attempt to preserve original data types
        for key in hf.attrs.keys():
            # Deserialize JSON data
            attr_data = json.loads(hf.attrs[key])

            # Direct assignment of the deserialized data
            # This ensures that lists and dictionaries are preserved as such
            data_dict[key] = attr_data

            if isinstance(attr_data, list) and all(
                isinstance(x, (int, float, list)) for x in attr_data
            ):
                try:
                    data_dict[key] = np.array(attr_data, dtype=object)
                except ValueError:
                    # Handle the case of ragged sequences; keep as list if conversion fails
                    pass

    return data_dict


def main(args):
    print("get a val pred model")
    # prepare a dataset
    data_dir = "/datasets/physionet.org/"
    mort_dir = os.path.join(data_dir, "laborder")
    dp = os.path.join(mort_dir, "proced_mimic.h5")

    s = time.time()
    # data_dict = pd.read_pickle(dp)
    data_dict = load_dict_from_hdf5(dp)
    e = time.time()
    print(f"Load data took: {e-s} s")

    config = dict()
    config["min_num_vals_past"] = args.min_tp
    config["min_num_vals_future"] = args.min_tf
    config["window_size"] = args.win_size

    path = dp
    tr_dataset = LabValueDatasetMIMIC(
        path=path, split="train", data_dict=data_dict, config=config
    )
    print(len(tr_dataset[4]))
    print("train size:", len(tr_dataset))
    val_dataset = LabValueDatasetMIMIC(
        path=path, split="val", data_dict=data_dict, config=config
    )
    print(len(val_dataset))
    te_dataset = LabValueDatasetMIMIC(
        path=path, split="test", data_dict=data_dict, config=config
    )
    print(len(te_dataset))
    # loader
    batch_size = 512
    tr_loader = PredictionLoader(tr_dataset, batch_size, True, sampler=None)
    val_loader = PredictionLoader(val_dataset, batch_size, False, sampler=None)
    te_loader = PredictionLoader(te_dataset, batch_size, False, sampler=None)

    for batch in tr_loader:
        x, y, xm, ym = batch
        print(x.size(), y.size(), xm.size(), ym.size())
        break

    # config
    # {LSTM, PatchTST, ...}

    config = dict()
    config["ckptdir"] = "/runs/labvalpred"
    config["dataset"] = "mimic"
    config["lr"] = 1e-4
    config["lradj"] = "type3"
    config["pct_start"] = 0.3
    config["loss"] = "mse"
    config["patience"] = 100
    config["epochs"] = 50
    config["batch_size"] = batch_size


    time_str = datetime.now().strftime("%b%d_%H-%M-%S_lab")
    identifier = f"{config['dataset']}_{args.model}_{config['lr']}_{args.min_tp}_{args.min_tf}_{args.win_size}"
    # trainer
    trainer = LabValTrainer(config, tr_loader, val_loader, te_loader, args=args)

    assert len(identifier) > 1
    if not args.test_only:
        print(f">>>>>>>start training : {identifier}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # training
        trainer.train(identifier)
    else:
        # report this model's MAE and MSE on val & test
        print(f">> >>>>>start testing : {identifier}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        id_notime = f"{config['dataset']}_{args.model}_{config['lr']}"
        id_notime = identifier
        trainer.test(id_notime)

    return


if __name__ == "__main__":
    """
    given a patient first 48 hours time-seires, predict all values for the next 24 hrs
    """
    parser = argparse.ArgumentParser(
        description="Lab test value Time Series Forecasting"
    )

    parser.add_argument("--random_seed", type=int, default=2023, help="random seed")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="PatchTST",
        help="model name, options: [Linear, LSTM, PatchTST, PatchTSMixer]",
    )

    # model & forecasting
    parser.add_argument("--output_dim", type=int, default=71, help="outdim for lstm")
    parser.add_argument("--layer_dim", type=int, default=3, help="num layers for lstm")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden for lstm")
    parser.add_argument("--seq_len", type=int, default=48, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=24, help="pred sequence length")
    parser.add_argument("--min_tp", type=int, default=10, help="min test past")
    parser.add_argument("--min_tf", type=int, default=5, help="min test future")
    parser.add_argument("--win_size", type=int, default=6, help="window_size")

    # gpu
    parser.add_argument(
        "--test_only", action="store_true", default=False, help="use gpu"
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=2, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="2,3", help="device ids of multile gpus"
    )
    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

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
    # data will z-score/minmax normalized first
    main(args)
