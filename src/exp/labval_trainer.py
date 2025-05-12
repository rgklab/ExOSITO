# from data_provider.data_factory import data_provider
from exp.trainer_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric


from torch.utils.tensorboard import SummaryWriter

import os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from models import Linear, PatchTST, lstm, PatchTSMixer



class LabValTrainer(Exp_Basic):
    def __init__(self, config, train_loader, val_loader, test_loader, args=None):
        super(LabValTrainer, self).__init__(config, args=args)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        ckptdir = config["ckptdir"]
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        self.ckptdir = ckptdir

    def _build_model(self):
        model_dict = {
            "Linear": Linear,
            "LSTM": lstm,
            "PatchTST": PatchTST,
            "PatchTSMixer": PatchTSMixer,
        }
        if self.args.model == "Linear":
            model = model_dict[self.args.model].Model(self.args).float()
        elif self.args.model == "LSTM":
            model = model_dict[self.args.model].Model(self.args).float()
        elif self.args.model == "PatchTST":
            model = model_dict[self.args.model].Model(self.args).float()
        elif self.args.model == "PatchTSMixer":
            model = model_dict[self.args.model].Model(self.args).float()
        else:
            raise NotImplementedError

        return model

    def _get_data(self, flag):
        pass

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                
                outputs = self.model(batch_x)
                f_dim = 0  # -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader = self.train_loader
        vali_loader = self.val_loader
        # test_loader = self.test_loader

        path = os.path.join(self.ckptdir, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # self.writer = SummaryWriter(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.config["pct_start"],
            epochs=self.config["epochs"],
            max_lr=self.config["lr"],
        )

        for epoch in range(self.config["epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                f_dim = 0  # -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.config["epochs"] - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.2f}s/iter; left time: {:.2f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.config["lradj"] == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.config, printout=False
                    )
                    scheduler.step()

            print(
                "Epoch: {0} cost time: {1:.2f}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = -1  # self.vali(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.5f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config["lradj"] != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.config)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, "bestckpt.pth")
        # path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # test_data, test_loader = self._get_data(flag="test")
        test_loader = self.test_loader
        # if test:
        print("loading model")
        # tmstr = self.config["time_str"]
        path = os.path.join(self.ckptdir, setting)  # f"{tmstr}_{setting}")
        self.model.load_state_dict(torch.load(os.path.join(path, "bestckpt.pth")))

        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(path, "test_results", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                f_dim = 0  # -1 if self.args.features == "MS" else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 5000 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        print(preds.shape, preds[0].shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("mse:{}, mae:{}, rse:{}".format(mse, mae, rse))
        f = open(os.path.join(folder_path, "result.txt"), "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, rse:{}".format(mse, mae, rse))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(
            os.path.join(folder_path, "metrics.npy"),
            np.array([mae, mse, rmse, mape, mspe, rse, corr], dtype=object),
        )
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = (
                    torch.zeros(
                        [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                    )
                    .float()
                    .to(batch_y.device)
                )
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "Linear" in self.args.model or "TST" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if "Linear" in self.args.model or "TST" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
