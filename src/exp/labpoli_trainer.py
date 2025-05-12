from exp.trainer_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric


from torch.utils.tensorboard import SummaryWriter

import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from models import Linear, PatchTST, lstm, PatchTSMixer, LabOrderOutcome
from models.conditionnormflow import ConditionalNormalizingFlow


def full_learning_critierion(out, gps, penalty, dataset_index, epsilon):
    sidx, eidx = dataset_index
    loss = (
        -out
        - (
            (penalty[sidx:eidx, :] * torch.sign(penalty[sidx:eidx, :]))
            * (gps - epsilon)
        )
    ).mean()
    return loss


def learning_critierion(out, gps, penalty, dataset_index, epsilon):
    loss = -out.mean()
    return loss


class LabOrderTrainer(Exp_Basic):
    def __init__(self, config, train_loader, val_loader, test_loader, args=None):
        super(LabOrderTrainer, self).__init__(config, args=args)
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
            model = model_dict[self.args.model].PModel(self.args).float()
        elif self.args.model == "LSTM":
            raise NotImplementedError
            # model = model_dict[self.args.model].PModel(self.args).float()
        elif self.args.model == "PatchTST":
            raise NotImplementedError
            # model = model_dict[self.args.model].PModel(self.args).float()
        elif self.args.model == "PatchTSMixer":
            model = model_dict[self.args.model].PModel(self.args).float()
        else:
            raise NotImplementedError
        self.model_dict = model_dict
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        return model_optim

    def _select_criterion(self):
        if self.args.use_gps:
            criterion = full_learning_critierion
        else:
            criterion = learning_critierion
        return criterion

    def _labusefulness(self, Xprev, Xpost, t, tup, tlow):
        # selected variability
        mp = self.config["labidx_map"]
        assert Xprev.dim() == Xpost.dim() == 3
        labvar = []
        bound_part = []

        for sidx in range(Xprev.size()[0]):
            orders = torch.nonzero(t[sidx] > 0.5, as_tuple=False).squeeze()
            if orders.dim() == 0:
                orders = orders.unsqueeze(0)
            if len(orders) > 0:
                featidx = torch.cat([torch.tensor(mp[lab.item()]) for lab in orders])

                # print(featidx)
                eXprev = Xprev[sidx, :, featidx]
                eXpost = Xpost[sidx, :, featidx]
                # print(eXprev.size())

                epmax = torch.abs(
                    torch.max(eXprev, dim=0)[0] - torch.max(eXprev, dim=0)[0]
                )
                epmin = torch.abs(
                    torch.min(eXprev, dim=0)[0] - torch.min(eXprev, dim=0)[0]
                )
                varr = torch.max(epmax.mean(), epmin.mean())
                varm = torch.abs(eXprev.mean(dim=0) - eXpost.mean(dim=0)).mean()
                # print(varr, varm)
                eachval = torch.add(varr, varm)
                # labvar.append(eachval)
            else:
                eachval = torch.tensor(0)
                # labvar.append(torch.tensor([0]))

            mask = tup[sidx] == tlow[sidx]
            labt = t[sidx, mask]
            labup = tup[sidx, mask]
            # print(mask)
            # print(labt)
            # print(labup)
            eachb = torch.sum(torch.abs(labt - labup)) * 0.1
            # bound_part += eachb

            mask = t[sidx] > 0.5
            cost = torch.sum(t[sidx, mask]) * 0.1
            # print(eachval, eachb, cost)
            a = torch.add(eachval, -eachb)
            b = torch.add(a, -cost)
            c = torch.add(5, b)
            labvar.append(b.unsqueeze(0))
        labvar = torch.cat(labvar).to(self.device)

        return labvar

    def train(self, setting):
        train_loader = self.train_loader
        vali_loader = self.val_loader

        # lambda_start

        # setup model g(t,x)
        outcome_model = LabOrderOutcome.LabOutcome(
            self.args, self.config["labidx_map"], self.device
        )
        outcome_model.eval()
        outcome_model.to(self.device)
        self.outcome_model = outcome_model

        # gps model f(t,x)
        gps_model = None
        if self.args.use_gps:
            gps_model = ConditionalNormalizingFlow(
                input_dim=self.args.y_len,
                context_dim=int(self.args.x_len * self.args.output_dim),
                hidden_dim=50,
                flow_length=1,
                count_bins=5,
                bound=0.5,
                device=self.device,
            )
            gpswp = "runs/gpscnf"
            gps_model.load_state_dict(
                torch.load(os.path.join(gpswp, "full_model_density.pt"))
            )
            gps_model.eval()
            gps_model.to(self.device)
        self.gps_model = gps_model

        if not self.args.use_gps:
            epsilon = 0
            lambda_start = 0
        else:
            lambda_start = 3
            all_eps = []
            for b in train_loader:
                x, true_y, _, _, _, t, _ = b
                inp = torch.cat((x, true_y), dim=1).view(x.size(0), -1).to(self.device)
                t = t.to(self.device)
                btarep = gps_model.log_prob(t - 0.5, inp).exp().detach()
                all_eps.append(btarep)
            # print(btarep[:10])
            tar_eps = torch.cat(all_eps, dim=0)
            print(tar_eps.size())
            tar_eps = torch.quantile(tar_eps.detach(), 0.05)
            a = torch.cat(all_eps, dim=0)
            tc = torch.sum(a > tar_eps).item()
            print(tar_eps, tc)
            epsilon = tar_eps.cpu().item()

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # penalty param for GPS (f)
        penalty_param = None
        optimizer_penalty = None
        if self.args.use_gps:
            print(a.size()[0])
            penalty_param = torch.nn.Parameter(
                torch.ones(a.size()[0], 1) * lambda_start, requires_grad=True
            )  # .to(self.device)
            init_lr_penalty = 0.01
            optimizer_penalty = torch.optim.Adam([penalty_param], lr=init_lr_penalty)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.config["pct_start"],
            epochs=self.config["epochs"],
            max_lr=self.config["lr"],
        )

        penalty_epoch = 0
        penalty_last_epoch = 0
        for epoch in range(self.config["epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            out_epoch = 0
            gps_epoch = 0
            val_out_epoch = 0
            # val_out_true_epoch = 0

            for i, (x, true_y, pred_y, lower, lowerxy, t, upper) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                if optimizer_penalty:
                    optimizer_penalty.zero_grad()

                batch_x = x.float().to(self.device)
                if self.args.use_ty:
                    batch_y = true_y.float().to(self.device)
                else:
                    batch_y = pred_y.float().to(self.device)

                if self.args.use_xyr:
                    rule_label = lowerxy.float().to(self.device)
                else:
                    rule_label = lower.float().to(self.device)
                observe_label = upper.float().to(self.device)
                inp = torch.cat((batch_x, batch_y), dim=1)
                outputs = self.model(inp)
                gps = None
                if self.args.use_gps:
                    order = (outputs > 0.5).float()
                    order = order - 0.5
                    order = torch.clip(order, -0.5, 0.5)
                    gps = gps_model.log_prob(order, inp.view(inp.size(0), -1)).exp()
                    # outputs = self.model(inp)  # outputs + 0.5

                outcome, term_tup = self.outcome_model(
                    batch_x, batch_y, outputs, observe_label, rule_label
                )
                if self.args.use_gps:
                    batchidx = (
                        i * self.config["batch_size"],
                        i * self.config["batch_size"] + batch_x.size()[0],
                    )
                    loss = criterion(
                        outcome,
                        gps,
                        penalty_param.detach().to(self.device),
                        batchidx,
                        epsilon,
                    )
                    loss.backward(retain_graph=True)
                    train_loss.append(loss.item())
                    model_optim.step()
                    loss = -1 * criterion(
                        outcome.detach(),
                        gps.detach(),
                        penalty_param.to(self.device),
                        batchidx,
                        epsilon,
                    )
                    loss.backward()
                    optimizer_penalty.step()
                    gps_epoch += gps.detach().mean()
                    penalty_epoch += penalty_param.detach().mean()
                else:
                    batchidx = 0
                    loss = criterion(outcome, gps, penalty_param, batchidx, epsilon)
                    loss.backward()
                    model_optim.step()
                    train_loss.append(loss.item())

                out_epoch += outcome.detach().mean()

            out_epoch /= len(train_loader)
            train_loss = np.average(train_loss)
            if self.args.use_gps:
                gps_epoch /= len(train_loader)
                penalty_epoch /= len(train_loader)

            # end epoch
            if self.config["lradj"] == "TST":
                adjust_learning_rate(
                    model_optim, scheduler, epoch + 1, self.config, printout=False
                )
                scheduler.step()

            print(
                "Epoch: {0} cost time: {1:.2f}, epoch outcome {2:2f}".format(
                    epoch + 1, time.time() - epoch_time, out_epoch.item()
                )
            )

            vali_loss, val_out = self.vali(vali_loader, criterion, epsilon=epsilon)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.5f} Vali Outcome: {4:.5f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, val_out
                )
            )

            # save path
            path = os.path.join(self.ckptdir, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config["lradj"] != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.config)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, "bestckpt.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        setup = dict()
        setup["args"] = self.args
        setup["config"] = self.config
        setup["setting"] = setting
        with open(os.path.join(path, "configs.pkl"), "wb") as f:
            pickle.dump(setup, f)
        return self.model

    def vali(self, vali_loader, criterion, epsilon=0):
        total_loss = []
        self.model.eval()
        val_out_epoch = 0
        with torch.no_grad():
            for i, (x, true_y, pred_y, lower, lowerxy, t, upper) in enumerate(
                vali_loader
            ):
                batch_x = x.float().to(self.device)
                if self.args.use_ty:
                    batch_y = true_y.float().to(self.device)
                else:
                    batch_y = pred_y.float().to(self.device)
                if self.args.use_xyr:
                    rule_label = lowerxy.float().to(self.device)
                else:
                    rule_label = lower.float().to(self.device)
                observe_label = upper.float().to(self.device)
                inp = torch.cat((batch_x, batch_y), dim=1)
                outputs = self.model(inp)

                gps = None
                if self.args.use_gps:
                    order = (outputs > 0.5).float()
                    order = order - 0.5
                    order = torch.clip(order, -0.5, 0.5)
                    gps = self.gps_model.log_prob(
                        order, inp.view(inp.size(0), -1)
                    ).exp()

                outcome, term_tup = self.outcome_model(
                    batch_x, batch_y, outputs, observe_label, rule_label
                )
                if self.args.use_gps:
                    outcome = (
                        torch.where(
                            gps > epsilon, outcome.reshape(-1), torch.tensor(0.0)
                        ).sum()
                        / x.size()[0]
                    )
                    val_out_epoch += outcome.detach()
                    loss = -outcome.detach()
                else:
                    val_out_epoch += outcome.detach().mean()
                    loss = -outcome.detach().mean()
                total_loss.append(loss.item())
        val_out_epoch /= len(vali_loader)
        total_loss = np.average(total_loss)
        self.model.train()
        return (total_loss, val_out_epoch)

    def test(self, setting):
        test_loader = self.test_loader
        path = os.path.join(self.ckptdir, setting)
        mpath = os.path.join(path, "bestckpt.pth")
        assert os.path.exists(mpath)

        labval_model = (
            self.model_dict[self.config["pred_model"]].Model(self.args).float()
        )
        pred_model_path = os.path.join(
            "runs/labvalpred",
            self.config["pred_model_dir"],
            "bestckpt.pth",
        )
        labval_model.to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            labval_model = nn.DataParallel(
                labval_model, device_ids=self.args.device_ids
            )
        labval_model.load_state_dict(
            torch.load(pred_model_path, map_location=self.device)
        )
        labval_model.eval()
        self.labval_model = labval_model

        outcome_model = LabOrderOutcome.LabOutcome(
            self.args, self.config["labidx_map"], self.device
        )
        outcome_model.eval()
        outcome_model.to(self.device)
        self.outcome_model = outcome_model

        criterion = self._select_criterion()

        # policy pi model
        self.model.load_state_dict(torch.load(mpath, map_location=self.device))

        total_loss = []
        test_out, test_out_realy = 0, 0
        varibs, varibsry = 0, 0
        bounds, boundsry = 0, 0
        costs, costsry = 0, 0

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _, rule_label, observe_label) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                rule_label = rule_label.float().to(self.device)
                observe_label = observe_label.float().to(self.device)
                pred_y = self.labval_model(batch_x)
                inp = torch.cat((batch_x, pred_y), dim=1)
                inp_real = torch.cat((batch_x, batch_y), dim=1)
                outputs = self.model(inp)
                outputs_with_realy = self.model(inp_real)

                outcome, term_tup = self.outcome_model(
                    batch_x, pred_y, outputs, observe_label, rule_label
                )
                outcomery, term_tupry = self.outcome_model(
                    batch_x, batch_y, outputs_with_realy, observe_label, rule_label
                )

                test_out += outcome.detach().mean()
                test_out_realy += outcomery.detach().mean()

                varib, bound, cost = term_tup
                varibs += varib.float().detach().mean()
                bounds += bound.float().detach().mean()
                costs += cost.float().detach().mean()

                varibry, boundry, costry = term_tupry
                varibsry += varibry.float().detach().mean()
                boundsry += boundry.float().detach().mean()
                costsry += costry.float().detach().mean()

                gps, penalty, didx, epsilon = 0, 0, 0, 0
                loss = criterion(outcome, gps, penalty, didx, epsilon)
                total_loss.append(loss.item())
        test_out /= len(test_loader)
        test_out_realy /= len(test_loader)
        varibs /= len(test_loader)
        bounds /= len(test_loader)
        costs /= len(test_loader)

        varibsry /= len(test_loader)
        boundsry /= len(test_loader)
        costsry /= len(test_loader)

        total_loss = np.average(total_loss)
        print(
            f"Testset: Outcome  {test_out:.5f}, Outcome with real future {test_out_realy:.5f}, test loss {total_loss:.3f}"
        )
        print(
            f"Test: Policy Variability {varibs:.3f} Policy bound loss: {bounds:.3f} Policy cost: {costs:.3f}"
        )
        print(
            f"Test (with real future): Policy Variability {varibsry:.3f} Policy bound loss: {boundsry:.3f} Policy cost: {costsry:.3f}"
        )

        _, val_out = self.vali(self.val_loader, criterion)
        print(f"Validation outcome {val_out:.5f}")

        with open(os.path.join(path, "result.txt"), "a") as of:
            of.write(setting + "  \n")
            of.write(f"Validation outcome {val_out:.5f}\n")
            of.write(
                f"Outcome  {test_out:.5f}, Outcome with real future {test_out_realy:.5f}, test loss {total_loss:.3f}\n"
            )
            of.write(
                f"Policy Variability {varibs:.3f} Policy bound loss: {bounds:.3f} Policy cost: {costs:.3f}\n"
            )
            of.write(
                f"Policy Variability {varibsry:.3f} Policy bound loss: {boundsry:.3f} Policy cost: {costsry:.3f} w/ real future\n"
            )
            of.close()

    def test_poli(self, setting):
        test_loader = self.test_loader
        model_path = self.args.ckptpath

        # g and f and \varepsilon
        outcome_model = LabOrderOutcome.LabOutcome(
            self.args, self.config["labidx_map"], self.device
        )
        outcome_model.eval()
        outcome_model.to(self.device)
        self.outcome_model = outcome_model
        gps_model = ConditionalNormalizingFlow(
            input_dim=self.args.y_len,
            context_dim=int(self.args.x_len * self.args.output_dim),
            hidden_dim=50,
            flow_length=1,
            count_bins=5,
            bound=0.5,
            device=self.device,
        )
        gpswp = "/runs/gpscnf"
        gps_model.load_state_dict(
            torch.load(os.path.join(gpswp, "full_model_density.pt"))
        )
        gps_model.eval()
        gps_model.to(self.device)
        if True:
            lambda_start = 3
            all_eps = []
            for b in self.train_loader:
                x, true_y, _, _, _, t, _ = b
                inp = torch.cat((x, true_y), dim=1).view(x.size(0), -1).to(self.device)
                t = t.to(self.device)
                btarep = gps_model.log_prob(t - 0.5, inp).exp().detach()
                all_eps.append(btarep)
            # print(btarep[:10])
            tar_eps = torch.cat(all_eps, dim=0)
            # print(tar_eps.size())
            tar_eps = torch.quantile(tar_eps.detach(), 0.05)
            a = torch.cat(all_eps, dim=0)
            tc = torch.sum(a > tar_eps).item()
            # print(tar_eps, tc)
            epsilon = tar_eps.cpu().item()
        self.gps_model = gps_model

        # policy pi
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        val_out_epoch = 0
        gpsval_out_epoch = 0
        phyval = 0
        gpsphy = 0
        eachval = 0
        eachb = 0
        cost = 0
        eachlowb = 0
        eachupb = 0
        geachval = 0
        geachb = 0
        gcost = 0
        geachlowb = 0
        geachupb = 0
        with torch.no_grad():
            for i, (x, true_y, pred_y, lower, lowerxy, t, upper) in enumerate(
                test_loader
            ):
                batch_x = x.float().to(self.device)
                if self.args.use_ty:
                    batch_y = true_y.float().to(self.device)
                else:
                    batch_y = pred_y.float().to(self.device)
                if self.args.use_xyr:
                    rule_label = lowerxy.float().to(self.device)
                else:
                    rule_label = lower.float().to(self.device)
                observe_label = upper.float().to(self.device)
                inp = torch.cat((batch_x, batch_y), dim=1)
                outputs = self.model(inp)


                order = (outputs > 0.5).float()
                order = order - 0.5
                order = torch.clip(order, -0.5, 0.5)
                gps = self.gps_model.log_prob(order, inp.view(inp.size(0), -1)).exp()

                outcome, term_tup = self.outcome_model.test_calc(
                    batch_x, batch_y, outputs, observe_label, rule_label
                )
                val_out_epoch += outcome.detach().mean()
                a, b, c, d, e = term_tup
                eachval += np.average(a)
                eachb += np.average(b)
                cost += np.average(c)
                eachlowb += np.average(d)
                eachupb += np.average(e)

                gps_outcome = (
                    torch.where(
                        gps > epsilon, outcome.reshape(-1), torch.tensor(0.0)
                    ).sum()
                    / x.size()[0]
                )
                gpsterms = []
                for i in list(term_tup):
                    gpst = (
                        torch.where(
                            gps > epsilon,
                            torch.tensor(i).reshape(-1).to(gps.device),
                            torch.tensor(0.0),
                        ).sum()
                        / x.size()[0]
                    )
                    gpsterms.append(gpst)
                geachval += gpsterms[0].item()
                geachb += gpsterms[1].item()
                gcost += gpsterms[2].item()
                geachlowb += gpsterms[3].item()
                geachupb += gpsterms[4].item()
                gpsval_out_epoch += gps_outcome.detach()

        val_out_epoch /= len(test_loader)
        gpsval_out_epoch /= len(test_loader)

        eachval /= len(test_loader)
        eachb /= len(test_loader)
        cost /= len(test_loader)
        eachlowb /= len(test_loader)
        eachupb /= len(test_loader)
        geachval /= len(test_loader)
        geachb /= len(test_loader)
        gcost /= len(test_loader)
        geachlowb /= len(test_loader)
        geachupb /= len(test_loader)


        print(eachval, eachb, cost, eachlowb, eachupb)
        print(geachval, geachb, gcost, geachlowb, geachupb)
        testout = val_out_epoch.item()
        gpstestout = gpsval_out_epoch.item()
        return (
            testout,
            gpstestout,
            eachval,
            eachb,
            cost,
            eachlowb,
            eachupb,
            geachval,
            geachb,
            gcost,
            geachlowb,
            geachupb,
        )

    def test_count(self, setting):
        test_loader = self.test_loader
        model_path = self.args.ckptpath

        # g and f and \varepsilon
        outcome_model = LabOrderOutcome.LabOutcome(
            self.args, self.config["labidx_map"], self.device
        )
        outcome_model.eval()
        outcome_model.to(self.device)
        self.outcome_model = outcome_model
        gps_model = ConditionalNormalizingFlow(
            input_dim=self.args.y_len,
            context_dim=int(self.args.x_len * self.args.output_dim),
            hidden_dim=50,
            flow_length=1,
            count_bins=5,
            bound=0.5,
            device=self.device,
        )
        gpswp = "runs/gpscnf"
        gps_model.load_state_dict(
            torch.load(os.path.join(gpswp, "full_model_density.pt"))
        )
        gps_model.eval()
        gps_model.to(self.device)
        if True:
            lambda_start = 3
            all_eps = []
            for b in self.train_loader:
                x, true_y, _, _, _, t, _ = b
                inp = torch.cat((x, true_y), dim=1).view(x.size(0), -1).to(self.device)
                t = t.to(self.device)
                btarep = gps_model.log_prob(t - 0.5, inp).exp().detach()
                all_eps.append(btarep)
            # print(btarep[:10])
            tar_eps = torch.cat(all_eps, dim=0)
            # print(tar_eps.size())
            tar_eps = torch.quantile(tar_eps.detach(), 0.05)
            a = torch.cat(all_eps, dim=0)
            tc = torch.sum(a > tar_eps).item()
            # print(tar_eps, tc)
            epsilon = tar_eps.cpu().item()
        self.gps_model = gps_model

        # policy pi
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        val_out_epoch = 0
        gpsval_out_epoch = 0
        phyval = 0
        gpsphy = 0
        eachval = 0
        eachb = 0
        cost = 0
        eachlowb = 0
        eachupb = 0
        geachval = 0
        geachb = 0
        gcost = 0
        geachlowb = 0
        geachupb = 0

        counters = [0] * 10
        mcounters = [0] * 10
        realt = [0] * 10
        with torch.no_grad():
            for i, (x, true_y, pred_y, lower, lowerxy, t, upper) in enumerate(
                test_loader
            ):
                batch_x = x.float().to(self.device)
                if self.args.use_ty:
                    batch_y = true_y.float().to(self.device)
                else:
                    batch_y = pred_y.float().to(self.device)
                if self.args.use_xyr:
                    rule_label = lowerxy.float().to(self.device)
                else:
                    rule_label = lower.float().to(self.device)
                observe_label = upper.float().to(self.device)
                inp = torch.cat((batch_x, batch_y), dim=1)
                outputs = self.model(inp)
                order = torch.sum((outputs > 0.5).float(), dim=0).detach().cpu().numpy()
                print(order, t.sum(dim=0).numpy())
                for i, o in enumerate(order):
                    counters[i] += o
                # assert 1 == 2
                inp[:, :, self.args.modify_idx] = 1.0
                noutputs = self.model(inp)
                norder = (
                    torch.sum((noutputs > 0.5).float(), dim=0).detach().cpu().numpy()
                )
                print(norder)
                for i, o in enumerate(norder):
                    mcounters[i] += o

                for i, o in enumerate(t.sum(dim=0).numpy()):
                    realt[i] += o


        print(counters)
        print(mcounters)
        print(realt)
