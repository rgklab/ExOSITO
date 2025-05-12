import os, sys, tarfile, collections, json, pickle, time, math
import torch
from os import listdir
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def normalize_array_clip01(arr, lower_bound, upper_bound):
    def normalize_value(value):
        # Clamping the value to the range [lower_bound, upper_bound]
        clamped_value = max(min(value, upper_bound), lower_bound)

        # Normalizing the clamped value to the range [-1, 1]
        res = (clamped_value - lower_bound) / (upper_bound - lower_bound)
        return round(res, 5)

    res = []
    for x in arr:
        if x >= 0:
            res.append(normalize_value(x))
        else:
            res.append(0)
    return res


def normalize_w_bound(arr, mean, std, lower_bound, upper_bound):
    res = []
    for v in arr:
        if v >= 0:
            clamped_value = max(min(v, upper_bound), lower_bound)
            val = (clamped_value - mean) / std
            res.append(val)
        else:
            res.append(0)
    return res


def normalize_array_clip(arr, lower_bound, upper_bound):
    def normalize_value(value):
        # Clamping the value to the range [lower_bound, upper_bound]
        clamped_value = max(min(value, upper_bound), lower_bound)

        # Normalizing the clamped value to the range [-1, 1]
        res = 2 * (clamped_value - lower_bound) / (upper_bound - lower_bound) - 1
        return round(res, 5)

    res = []
    for x in arr:
        if x >= 0:
            res.append(normalize_value(x))
        else:
            res.append(0)
    return res


def match_tuples(*args):
    # Find the intersection of all 't' values across the lists
    common_ts = set.intersection(*[set([t for t, _ in lst]) for lst in args])
    # Filter each list to only include tuples where 't' is in the common set
    filtered_lists = [[(t, x) for (t, x) in lst if t in common_ts] for lst in args]
    return filtered_lists


def bin_tuples(T, vals, num_bins):
    bin_size = T / num_bins
    # bins = [[] for _ in range(num_bins)]
    res = []
    for t, y in vals:
        bin_index = int(t // bin_size)
        # Ensure that the index is within the range [0, num_bins - 1]
        bin_index = min(bin_index, num_bins - 1)
        # bins[bin_index].append((t, y))
        res.append((t, y, bin_index))
    return res


def remove_duplicate_ts(tuple_list):
    """
    This function receives a list of tuples and returns a new list with unique t values.
    """
    seen_ts = set()
    new_list = []
    for t, x in tuple_list:
        if t not in seen_ts:
            new_list.append((t, x))
            seen_ts.add(t)
    return new_list


class LabValueDatasetMIMIC(Dataset):
    """Dataset for mortality for MIMIC-IV"""

    def __init__(
        self,
        path=None,
        split="train",
        config=dict(),
        data_dict=None,
    ):
        self.path = path
        self.split = split
        assert self.split in ["train", "val", "test", "all"]

        # s = time.time()
        if data_dict is None:
            raise NotImplementedError()
            # with open(self.path, "rb") as f:
            #     data_dict = pickle.load(f)
        else:
            data_dict = data_dict
        # e = time.time()
        # print(data_dict.keys())
        # print(f"Load data took: {e-s} s")
        assert f"{split}_mask" in data_dict
        mask = data_dict[f"{self.split}_mask"].astype(bool)

        if "imputation" in config:
            self.imputation = config["imputation"]
        else:
            self.imputation = "mean"
        if "normalization" in config:
            self.normalization = config["normalization"]
        else:
            self.normalization = "clip"
        if "min_val_past" in config:
            self.min_val_past = config["min_val_past"]
        else:
            self.min_val_past = 10
        if "min_val_future" in config:
            self.min_val_future = config["min_val_future"]
        else:
            self.min_val_future = 5
        assert "min_num_vals_past" in config
        assert "min_num_vals_future" in config
        assert "window_size" in config
        self.window_size = config["window_size"]
        self.future_hours = 24
        self.past_hours = 48
        self.min_num_vals_past = config["min_num_vals_past"]
        self.min_num_vals_future = config["min_num_vals_future"]

        s = time.time()
        self._preproc_dataset(data_dict, mask)
        assert len(self.sparse_ms) > 0
        assert self.total_idx
        e = time.time()
        print(f"Preproc data took: {e-s} s")

    def __len__(self):
        return len(self.total_idx)

    def __getitem__(self, idx):
        pair = self.total_idx[idx]
        smidx, pt = pair
        sm = self.sparse_ms[smidx]
        et = self.etimes[smidx]
        x, y, xm, ym = self._get_one_stay(
            sm, pt, et, imputation=self.imputation, normalization=self.normalization
        )
        return x, y, xm, ym

    def _preproc_dataset(self, data_dict, mask):
        # for k in data_dict:
        #     print(k, len(data_dict[k]))
        #     print(f'self.{k} = data_dict["{k}"]')
        self.sids = data_dict["sids"][mask]
        self.Ts = data_dict["Ts"][mask]
        self.Ys = data_dict["Ys"][mask]
        self.ind_kts = data_dict["ind_ts"][mask]
        self.ind_kfs = data_dict["ind_fs"][mask]
        self.sparse_ms = np.array(data_dict["sparse_matrices"], dtype=object)[mask]
        self.labels = data_dict["labels"][mask]
        self.etimes = data_dict["etimes"][mask]

        self.fns_dict = data_dict["fns_dict"]
        self.f2i = data_dict["f2i"]
        self.i2f = data_dict["i2f"]
        self.counts = data_dict["counts"]
        self.var_df = data_dict["var_df"]
        self.feature_norms = data_dict["feature_norms"]
        self.train_mask = data_dict["train_mask"]
        self.val_mask = data_dict["val_mask"]
        self.test_mask = data_dict["test_mask"]
        self.all_mask = data_dict["all_mask"]
        self.t2f = data_dict["t2f"]
        self.f2t = data_dict["f2t"]

        self.include_before_death = -1
        self.before_end = 0


        print(len(self.sids))

        labfns = self.fns_dict["labfns"]
        labidfs = []
        for f in self.f2i:
            if f.split("_")[0] in labfns:
                labidfs.append(self.f2i[f])
        # self._get_sparse_matrices()
        self.total_idx = []
        for idx, (sm, et, T, ind_f, ind_t) in enumerate(
            zip(self.sparse_ms, self.etimes, self.Ts, self.ind_kfs, self.ind_kts)
        ):
            num_pts = int(et // self.window_size - 1)
            # assert num_pts >= 1
            if num_pts < 1:
                self.total_idx.append((idx, et // 2))
                # print(et, idx, "not included")
                # continue
            if num_pts == 1:
                self.total_idx.append((idx, self.window_size))
            else:
                labT = []
                for idf, t in zip(ind_f, T):
                    if idf in labidfs:
                        labT.append(t)

                for i in range(num_pts):
                    pt = (i + 1) * self.window_size
                    lower = max(0, pt - self.past_hours)
                    upper = min(et, pt + self.future_hours)
                    arrT = np.array(labT, dtype=float)
                    T = np.array(T, dtype=float)
                    # arrT = np.array(T, dtype=float)

                    numvalp = len(T[(T >= lower) & (T <= pt)]) >= self.min_val_past
                    numvalf = len(T[(T <= upper) & (T >= pt)]) >= self.min_val_future
                    numtestp = (
                        len(arrT[(arrT >= lower) & (arrT <= pt)])
                        >= self.min_num_vals_past
                    )
                    numtestf = (
                        len(arrT[(arrT <= upper) & (arrT >= pt)])
                        >= self.min_num_vals_future
                    )
                    if numvalp and numvalf and numtestp and numtestf:
                        self.total_idx.append((idx, pt))
                # print(self.total_idx)

        print(f"total idx {len(self.total_idx)}")
        # list -> sparse matrix
        # sparse matrix -> prev value matrix  + future value matrix + observation masks

        vaso_names = [
            "Vasopressors_221289",
            "Vasopressors_221653",
            "Vasopressors_221662",
            "Vasopressors_221749",
            "Vasopressors_221906",
            "Vasopressors_221986",
            "Vasopressors_222315",
            "Vasopressors_229617",
        ]

        labfns = self.fns_dict["labfns"]
        binaryfns = self.fns_dict["binaryfns"]
        valfns = self.fns_dict["valfns"]

        # if "Vasopressors" in
        assert "Vasopressors" in valfns
        import copy

        valfns = copy.deepcopy(valfns)
        valfns.remove("Vasopressors")
        valfns.extend(vaso_names)
        self.valfns = valfns
        self.labfns = labfns
        self.binaryfns = binaryfns

        allfns = labfns + binaryfns + valfns
        feat2idx = {k: i for i, k in enumerate(allfns)}
        self.feat2idx = feat2idx

        # with in a range of values
        self.clipfns = [
            "Hemoglobin",
            "WBC",
            "Platelets",
            "Sodium",
            "Potassium",
            "Calcium",
            "Phosphate",
            "Magnesium",
            "INR",
            "ALP",
            "Bilirubin",
            "ALT",
            "Lactate",
            "PaCO2",
            "PaO2",
            "ph",
            "Bicarbonate",
            "Creatinine",
            "BloodUreaNitrogen",
            "CreatinineKinase",
            "MeanBloodPressure",
            "Temperature",
            "HeartRate",
            "SystolicBloodPressure",
            "MinuteVentilation",  # ?
            "TidalVolume",  # ?
            "DiastolicBloodPressure",
            "PEEP",
            "RespiratoryRate",
            "SAS",
        ]
        # lower bound or 0 as default
        self.normfns = vaso_names + [
            "UrineOutput",
            "FiO2",
            "AirwayPressure",
        ]
        assert len(self.feature_norms) == len(self.normfns) + len(self.clipfns) + 3
        return

    def _get_one_stay(
        self, sparse_m, point, etime, imputation="mean", normalization="clip"
    ):
        pt = point
        lower = max(0, pt - self.past_hours)
        upper = min(etime, pt + self.future_hours)

        xdic, ydic = {}, {}
        # get x and y
        for k, v in sparse_m.items():
            xvals, yvals = [], []
            for tup in v:
                t, y = tup
                if t >= lower and t <= pt:
                    xvals.append(tup)
                if t >= pt and t <= upper:
                    yvals.append(tup)
            if len(xvals) > 0:
                xdic[k] = xvals
            if len(yvals) > 0:
                ydic[k] = yvals

        # normalize x and y
        normx = self._normalize_stay(xdic, point, isx=True)
        normy = self._normalize_stay(ydic, point, isx=False)
        # print(normx.shape, normy.shape)
        # print(normx)
        # print("y")
        # print(normy)
        # impute x and y

        impx = self._impute_stay(normx)
        impy = self._impute_stay(normy)

        xm = torch.from_numpy(np.zeros_like(normx)).float()
        ym = torch.from_numpy(np.zeros_like(normy)).float()

        return impx, impy, xm, ym

    def _normalize_stay(self, stay, pt, isx=True, labonly=False):
        for k in stay:
            if (k.split("_")[0] not in self.feat2idx) and (k not in self.feat2idx):
                raise ValueError(f"{k} not in feat2idx")
        # construct array
        num_feats = len(self.feat2idx)
        if isx:
            res = np.zeros((self.past_hours, num_feats))
        else:
            if labonly:
                num_feats = len(self.labfns)
            else:
                num_feats = num_feats
            res = np.zeros((self.future_hours, num_feats))

        # norm values
        for k in stay:
            fn = k.split("_")[0]
            vals = stay[k]
            if k in self.feat2idx:
                midx = self.feat2idx[k]
            else:
                midx = self.feat2idx[fn]

            vs = [tp[1] for tp in vals]
            ts = [tp[0] for tp in vals]
            if fn not in self.binaryfns and fn != "ICDSC":
                if self.normalization == "clip":
                    nvs = self._norm_vals_clip(vs, k)
                else:
                    raise NotImplementedError
            else:
                if fn == "Ventilation" or fn == "ICDSC":
                    nvs = vs
                else:
                    nvs = [1 for _ in vals]
            # if fn not in self.binaryfns and fn != "ICDSC":
            #     print(nvs)

            if isx:
                valres = np.zeros(self.past_hours)
                valcts = np.zeros(self.past_hours)
            else:
                valres = np.zeros(self.future_hours)
                valcts = np.zeros(self.future_hours)
            for t, v in zip(ts, nvs):
                # obtain interval
                if isx:
                    assert t <= pt
                    if pt - t == self.past_hours:
                        dtidx = 0
                    else:
                        dtidx = self.past_hours - math.floor(pt - t) - 1
                    if dtidx < 0 or dtidx >= self.past_hours:
                        print(t, pt, dtidx)
                        assert dtidx >= 0 and dtidx < self.past_hours
                else:
                    assert t >= pt
                    if t - pt == 0:
                        dtidx = 0
                    else:
                        dtidx = math.ceil(t - pt) - 1
                    if dtidx < 0 or dtidx >= self.future_hours:
                        print(t, pt, dtidx)
                        assert dtidx >= 0 and dtidx < self.future_hours

                valcts[dtidx] += 1
                valres[dtidx] += v

                # if fn not in self.binaryfns and fn != "ICDSC":
                #     print(pt, t, dtidx)

            if "Urine" not in fn:
                # if fn not in self.binaryfns and fn != "ICDSC":
                #     print(valcts, valres)
                normres = np.divide(
                    valres,
                    valcts,
                    out=np.zeros_like(valres),
                    where=(valcts != 0),
                )
            else:
                normres = valres
            # print(fn, normres, nvs, ts, pt)
            res[:, midx] = normres
            # res[res < 0] = 0
        return res

    def _norm_vals_clip(self, vals, ffn):
        # print(ffn)
        fn = ffn.split("_")[0]
        if ffn not in self.feature_norms and fn not in self.feature_norms:
            print(f"{ffn} not exists")
            assert 1 == 2
        if "Vaso" in fn:
            statsdic = self.feature_norms[ffn]
        else:
            statsdic = self.feature_norms[fn]

        if "Troponin" in fn:
            # Troponin care
            return [1 for _ in vals]
        elif "GCS" in fn:
            # max should be norm
            # print(vals)
            vals = [15 - v for v in vals]
            res = normalize_array_clip01(vals, 0, 12)
            # print(res)
            return res
        elif fn in self.clipfns:
            # -1 to 1
            mean, std, low, up = (
                statsdic["mean"],
                statsdic["std"],
                statsdic["1"],
                statsdic["99"],
            )
            # print(vals)
            res = normalize_array_clip(vals, low, up)
            # res = normalize_w_bound(vals, mean, std, low, up) # z-score
            # print(res)
            return res
        elif fn in self.normfns or ffn in self.normfns:
            # 0 to 1
            # print(vals)
            mean, std, low, up = (
                statsdic["mean"],
                statsdic["std"],
                statsdic["1"],
                statsdic["99"],
            )
            # res = normalize_w_bound(vals, mean, std, low, up) # z-score
            res = normalize_array_clip01(vals, low, up)
            # print(res)
            return res
        else:
            raise ValueError(f"{ffn} not taken cared")

    def _impute_stay(self, stay):
        # stay is already a numpy array

        if self.imputation == "mean":
            res = torch.from_numpy(stay).float()
            return res
        elif self.imputation == "interpolate":
            raise NotImplementedError
        elif self.imputation == "right":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _get_sparse_matrices(self):
        # print(self.f2i)
        sparse_ms = []
        for idx, (T, Y, ind_kt, ind_kf, etime, label) in enumerate(
            zip(
                self.Ts,
                self.Ys,
                self.ind_kts,
                self.ind_kfs,
                self.etimes,
                self.labels,
            )
        ):
            stay_data = (T, Y, ind_kt, ind_kf, etime, label)
            # icdscid = [96, 100, 129, 139]
            # icdscid = [63, 69, 117]
            # icdscid = [9, 10, 11]
            # icdscid = [30, 31, 81, 164, 176, 178]
            ct = 0
            if True:
                sparse_m = self._proc_one_stay(stay_data)
                sparse_ms.append(sparse_m)
        self.sparse_ms = sparse_ms
        # return

    def _proc_one_stay(self, stay_data):
        T, Y, ind_t, ind_f, end_time, label = stay_data
        pfeat = list(set(ind_f))

        # generate sparse matrix ds
        tfeatdic = {self.i2f[k]: [] for k in pfeat}
        if self.include_before_death > 0:
            # self.num_X_pred = self.include_before_death
            last = self.include_before_death
            skip = self.before_end
            assert end_time - last - skip > 0
            for y, idt, idf in zip(Y, ind_t, ind_f):
                if T[idt] >= (end_time - last - skip) and T[idt] <= (end_time - skip):
                    tfeatdic[self.i2f[idf]].append((round(T[idt], 3), y))
        else:
            for y, idt, idf in zip(Y, ind_t, ind_f):
                tfeatdic[self.i2f[idf]].append((round(T[idt], 3), y))

        # sort based on time and remove duplicates
        tfeatdic = {k: sorted(v, key=lambda x: x[0]) for k, v in tfeatdic.items()}
        for k in tfeatdic:
            newl = remove_duplicate_ts(tfeatdic[k])
            tfeatdic[k] = newl

        # make sure no strings anymore
        # ICDSC
        icdsc_res = []
        icdsc = {}
        for k in tfeatdic:
            if "ICDSC" in k:
                icdsc[k] = tfeatdic[k]
        for k in icdsc:
            tfeatdic.pop(k, None)
        if len(icdsc) > 0:
            icdscts = []
            for k in icdsc:
                for v in icdsc[k]:
                    t, y = v
                    icdscts.append(t)
            icdscts = sorted(list(set(icdscts)))
            for t in icdscts:
                temp = {}
                for k in icdsc:
                    tvs = icdsc[k]
                    for tup in tvs:
                        if tup[0] == t:
                            temp[k] = tup[1]
                # print(temp)
                if (
                    "ICDSC_228334" in temp
                    and "ICDSC_228336" in temp
                    and temp["ICDSC_228334"] == "No"
                    and temp["ICDSC_228336"] == "No"
                ):
                    icdsc_res.append((t, 0))
                elif "ICDSC_228335" in temp and temp["ICDSC_228335"] != "Yes":
                    icdsc_res.append((t, 0))
                elif "ICDSC_228337" in temp and temp["ICDSC_228337"] != "Yes":
                    icdsc_res.append((t, 0))
                else:
                    icdsc_res.append((t, 1))
            for k in tfeatdic:
                assert "ICDSC" not in k
            tfeatdic["ICDSC_0"] = icdsc_res
            # print(icdsc_res)

        # Ventilation
        for k in tfeatdic:
            if "223849" in k or "229314" in k:
                vals = tfeatdic[k]
                new_vals = [(t, 2) for (t, _) in vals]
                tfeatdic[k] = new_vals
                # print(tfeatdic[k])
            if "227577" in k:
                vals = tfeatdic[k]
                # print(vals)
                new_vals = [(t, 1) for (t, _) in vals]
                tfeatdic[k] = new_vals
                # print(tfeatdic[k])

        # ICP
        tfeatdic.pop("ICPMonitor_226125", None)

        # Propofol
        tfeatdic.pop("Propofol_222168", None)

        # AirwayPressure
        airway = []
        for k in tfeatdic:
            if "AirwayPressure" in k:
                airway.append(k)
        for k in airway:
            if "224697" not in k:
                tfeatdic.pop(k, None)

        # RR
        tfeatdic.pop("RespiratoryRate_224690", None)

        # GCS
        gcs = {}
        for k, v in tfeatdic.items():
            if "GCS" in k:
                gcs[k] = v
        for k in gcs:
            tfeatdic.pop(k, None)
        if len(gcs) > 0:
            # print(gcs)
            if len(gcs.keys()) == 3:
                gcsvals = [v for _, v in gcs.items()]
                fgcs = match_tuples(gcsvals[0], gcsvals[1], gcsvals[2])
                ngcs = [
                    (x[0], x[1] + y[1] + z[1])
                    for x, y, z in zip(fgcs[0], fgcs[1], fgcs[2])
                ]
                tfeatdic["GCS_0"] = ngcs
                # print(tfeatdic["GCS_0"])

        # UrineOutput
        uout = {}
        for k, v in tfeatdic.items():
            if "UrineOutput" in k:
                uout[k] = v
        for k in uout:
            tfeatdic.pop(k, None)
        if len(uout) > 0:
            ts = []
            for k, v in uout.items():
                for tp in v:
                    ts.append(tp[0])
            ts = sorted(list(set(ts)))
            uout_res = []
            for t in ts:
                temp = []
                for k, v in uout.items():
                    for thist, y in v:
                        if thist == t:
                            temp.append(y)
                uout_res.append((t, sum(temp)))
            tfeatdic["UrineOutput_0"] = uout_res

        # find duplicate features
        fn2full = {}
        dupfns = []
        for n in tfeatdic:
            fn = n.split("_")[0]
            if fn in fn2full:
                fn2full[fn].append(n)
            else:
                fn2full[fn] = [n]

        keepfns = ["Vasopressors"]
        for k, v in fn2full.items():
            if len(v) > 1:
                if k not in keepfns:
                    dupfns.append(k)

        print(fn2full)
        print(dupfns)

        newtfdic = {}
        for fn in dupfns:
            this_fn_vals = []
            thisfns = fn2full[fn]
            for k in thisfns:
                this_fn_vals.extend(tfeatdic[k])
            a = len(this_fn_vals)
            b = this_fn_vals
            this_fn_vals.sort(key=lambda x: x[0])
            this_fn_vals = remove_duplicate_ts(this_fn_vals)
            if a != len(this_fn_vals):
                print(fn)
                print(b, a)
                print(this_fn_vals, len(this_fn_vals))

            newtfdic[str(fn) + "_0"] = this_fn_vals

        for k in tfeatdic:
            if k.split("_")[0] not in dupfns:
                newtfdic[k] = tfeatdic[k]

        tfeatdic = newtfdic

        return tfeatdic  # tf
