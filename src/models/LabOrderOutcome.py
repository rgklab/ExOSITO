import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabOutcome(nn.Module):
    """
    nn module that calculates the outcome for ordered lab test
    """

    def __init__(self, configs, lab2idx, device):
        super(LabOutcome, self).__init__()

        self.wdx = configs.wDX
        self.wlb = configs.wLB
        self.wc = configs.wC
        self.same_cost = configs.use_scost
        self.mp = lab2idx
        self.device = device
        # self.Linear = nn.Linear(self.seq_len, self.pred_len)
        if self.same_cost:
            costs = [0.1] * configs.y_len
            self.costs = torch.tensor(costs)
        else:
            # print(lab2idx.keys())
            rc = [12, 5, 12.36, 18, 9.1, 10, 18.62, 1.5, 18, 1.5]
            costs = [i / sum(rc) for i in rc]
            # print(sum(costs))
            self.costs = torch.tensor(costs)
        assert len(lab2idx) == len(self.costs)

    def forward(self, Xprev, Xpost, t, tup, tlow):
        # x: [Batch, Input length, Channel]
        # t, tup, tlow: [Batch, num_tests]
        # x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        assert Xprev.dim() == Xpost.dim() == 3
        mp = self.mp
        labvar = []
        deltaxs = []
        lbounds = []
        cts = []
        # loop over batch size
        for sidx in range(Xprev.size()[0]):
            # Find order test index
            orders = torch.nonzero(t[sidx] > 0.5, as_tuple=False).squeeze()
            thisorder = torch.sigmoid((t[sidx] - 0.5) * 1000)
            if orders.dim() == 0:
                orders = orders.unsqueeze(0)

            deltats = torch.zeros_like(thisorder)
            if len(orders) > 0 and self.wdx > 0:
                featidx = torch.cat([torch.tensor(mp[lab]) for lab in list(mp.keys())])
                eXprev = Xprev[sidx, :, featidx]
                eXpost = Xpost[sidx, :, featidx]
                epmax = torch.abs(
                    torch.max(eXprev, dim=0)[0] - torch.max(eXprev, dim=0)[0]
                )
                epmin = torch.abs(
                    torch.min(eXprev, dim=0)[0] - torch.min(eXprev, dim=0)[0]
                )
                prevm = eXprev.mean(dim=0)
                postm = eXpost.mean(dim=0)

                for lab in orders:
                    lidx = torch.tensor(mp[lab.item()])
                    varr = torch.max(epmax[lidx].sum(), epmin[lidx].sum())
                    varm = torch.abs(prevm[lidx] - postm[lidx]).sum()
                    eachvalo = torch.add(varr, varm)
                    deltats[lab] += eachvalo
            eachval = torch.sum(deltats * thisorder)
            deltaxs.append(eachval.item())

            # Loss Bound
            mask = tup[sidx] == tlow[sidx]
            labt = t[sidx, mask]
            labup = tlow[sidx, mask]
            eachb = torch.sum(torch.abs(labt - labup))
            
            
            lbounds.append(eachb.item())

            # Order Cost
            wts = self.costs.to(t.device)
            ordert = torch.sigmoid((t[sidx] - 0.5) * 1000)
            cost = torch.sum(ordert.float() * wts)
            cts.append(cost.item())

            # print(eachval, eachb, cost)
            a = torch.add(eachval * self.wdx, -eachb * self.wlb)
            b = torch.add(a, -cost * self.wc)
            labvar.append(b.unsqueeze(0))
            
        labvar = torch.cat(labvar).to(self.device)
        eachval, eachb, cost = (
            np.average(deltaxs),
            np.average(lbounds),
            np.average(cts),
        )
        return labvar, (eachval, eachb, cost)  # [Batch, Output length, Channel]

    def test_calc(self, Xprev, Xpost, t, tup, tlow):
        # x: [Batch, Input length, Channel]
        # t, tup, tlow: [Batch, num_tests]
        # x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        assert Xprev.dim() == Xpost.dim() == 3
        mp = self.mp
        labvar = []
        deltaxs = []
        lbounds = []
        cts = []
        lblows = []
        lbups = []
        # loop over batch size
        for sidx in range(Xprev.size()[0]):
            # Find order test index
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
                varr = torch.max(epmax.sum(), epmin.sum())
                varm = torch.abs(eXprev.mean(dim=0) - eXpost.mean(dim=0)).sum()
                # print(varr, varm)
                eachval = torch.add(varr, varm)
                # labvar.append(eachval)
            else:
                eachval = torch.tensor(0)
                # labvar.append(torch.tensor([0]))
            deltaxs.append(eachval.item())

            # Loss Bound
            mask = tup[sidx] == tlow[sidx]
            labt = t[sidx, mask]
            labup = tlow[sidx, mask]
            eachb = torch.sum(torch.abs(labt - labup))
            lbounds.append(eachb.item())

            # new for test
            mask = tup[sidx] == 0
            labt = t[sidx, mask]
            upb = tup[sidx, mask]
            elbup = torch.sum(labt > 0.5)

            lbups.append(elbup.item())

            mask = tlow[sidx] == 1
            labt = t[sidx, mask]
            lowb = tlow[sidx, mask]
            # print(labt, lowb, labt <= 0.5)
            elblow = torch.sum(labt <= 0.5)  # torch.abs(labt - lowb))
            # print(elblow)
            lblows.append(elblow.item())

            # assert 1 == 2

            # Order Cost
            mask = t[sidx] > 0.5
            wts = self.costs.to(t.device)
            cost = torch.sum(mask.float() * wts)
            cts.append(cost.item())

            # print(eachval, eachb, cost)
            a = torch.add(eachval * self.wdx, -eachb * self.wlb)
            b = torch.add(a, -cost * self.wc)
            labvar.append(b.unsqueeze(0))
            # assert 1 == 2
            # c = torch.add(5, b)
        labvar = torch.cat(labvar).to(self.device)
        eachval, eachb, cost = deltaxs, lbounds, cts
        eachlowb, eachupb = lblows, lbups  # np.average(lblows), np.average(lbups)
        return labvar, (
            eachval,
            eachb,
            cost,
            eachlowb,
            eachupb,
        )  # [Batch, Output length, Channel]


if __name__ == "__main__":
    print("testing laborder model")
