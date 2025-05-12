import os, sys, tarfile, collections, json, pickle, time, math
import torch
from os import listdir
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def has_twenty_percent_increase(arr):
    for i in range(1, len(arr)):
        if arr[i] > 1.20 * arr[i - 1]:
            return True
    return False


def has_25_percent_increase(arr):
    for i in range(1, len(arr)):
        if arr[i] > 1.25 * arr[i - 1]:
            return True
    return False


def has_50_increase(arr):
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > 50:
            return True
    return False


def has_2_increase(arr):
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > 2:
            return True
    return False


def has_thirty_percent_decrease(arr):
    for i in range(1, len(arr)):
        if arr[i] > 0.7 * arr[i - 1]:
            return True
    return False


def has_five_unit_change_in_past_24_hours(t, x, curr_time):
    for i in range(len(t) - 1, 0, -1):  # Start from the most recent time
        current_time = curr_time
        past_24_hours_time = current_time - 24

        # Get the index of the earliest time within the past 24 hours
        j = i
        while j >= 0 and t[j] > past_24_hours_time:
            j -= 1

        # Check if any value within the past 24 hours has changed by 5 or more units
        for k in range(j + 1, i):
            if abs(x[i] - x[k]) >= 5:
                return True
    return False


def has_2_unit_decrease_in_past_24_hours(t, x, curr_time):
    for i in range(len(t) - 1, 0, -1):  # Start from the most recent time
        current_time = curr_time  # t[i]
        past_24_hours_time = current_time - 24

        # Get the index of the earliest time within the past 24 hours
        j = i
        while j >= 0 and t[j] > past_24_hours_time:
            j -= 1

        # Check if any value within the past 24 hours has changed by 5 or more units
        for k in range(j + 1, i):
            if x[k] - x[i] >= 2:
                return True
    return False


def has_30p_unit_decrease_in_past_48_hours(t, x, curr_time):
    for i in range(len(t) - 1, 0, -1):  # Start from the most recent time
        current_time = curr_time
        past_24_hours_time = current_time - 48

        # Get the index of the earliest time within the past 24 hours
        j = i
        while j >= 0 and t[j] > past_24_hours_time:
            j -= 1

        # Check if any value within the past 24 hours has changed by 5 or more units
        for k in range(j + 1, i):
            if x[i] < x[k] * 0.7:
                return True
    return False


def has_six_unit_change_in_past_24_hours(t, x, curr_time):
    for i in range(len(t) - 1, 0, -1):  # Start from the most recent time
        current_time = curr_time  # t[i]
        past_24_hours_time = current_time - 24

        # Get the index of the earliest time within the past 24 hours
        j = i
        while j >= 0 and t[j] > past_24_hours_time:
            j -= 1

        # Check if any value within the past 24 hours has increased by 5 or more units
        for k in range(j + 1, i):
            if abs(x[i] - x[k]) >= 6:
                return True

    return False


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


class LabOrderDatasetMIMIC(Dataset):
    """Dataset for lab test order (with bound) for MIMIC-IV"""

    def __init__(
        self,
        path=None,
        split="train",
        preproc_params=None,
        imputation="mean",
        data_dict=None,
    ):
        self.path = path
        self.split = split
        self.imputation = imputation
        assert self.split in ["train", "val", "test", "all"]

        s = time.time()
        if data_dict is None:
            with open(self.path, "rb") as f:
                data_dict = pickle.load(f)
        else:
            data_dict = data_dict
        e = time.time()
        # print(data_dict.keys())
        print(f"Load data took: {e-s:.3f} s")
        assert f"{split}_mask" in data_dict
        mask = data_dict[f"{self.split}_mask"]

        s = time.time()
        self._preproc_dataset(data_dict, mask)
        assert len(self.sparse_ms) > 0
        assert self.total_idx
        e = time.time()
        print(f"Preproc data took: {e-s:.3f} s")

        self.imputation = imputation
        self.normalization = "clip"

    def __len__(self):
        return len(self.total_idx)

    def __getitem__(self, idx):
        pair = self.total_idx[idx]
        smidx, pt = pair
        sm = self.sparse_ms[smidx]
        et = self.etimes[smidx]
        x, y, xm, ym, rule, obs = self._get_one_stay(
            sm, pt, et, imputation=self.imputation, normalization=self.normalization
        )
        return x, y, xm, ym, rule, obs

    def _preproc_dataset(self, data_dict, mask):
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

        self.window_size = 6
        self.future_hours = 24
        self.past_hours = 48
        self.include_before_death = -1
        self.before_end = 0

        self.min_num_vals_past = 10
        self.min_num_vals_future = 5
        self.num_tests = 10
        self.t2f_map = {
            "cbc": ["Hemoglobin", "WBC", "Platelets"],
            "ele": ["Sodium", "Potassium"],
            "cal": ["Calcium", "Phosphate", "Magnesium"],
            "inr": ["INR"],
            "lvr": ["ALP", "Bilirubin", "ALT"],
            "lac": ["Lactate"],
            "abg": ["PaCO2", "PaO2", "ph", "Bicarbonate"],
            "cre": ["Creatinine", "BloodUreaNitrogen"],
            "tro": ["Troponin"],
            "ck": ["CreatinineKinase"],
        }
        tmpd = {}
        for k, v in self.t2f_map.items():
            for kk in v:
                tmpd[kk] = k
        self.f2t_map = tmpd

        self.t2i_map = {
            "cbc": 0,
            "ele": 1,
            "cal": 2,
            "inr": 3,
            "lvr": 4,
            "lac": 5,
            "abg": 6,
            "cre": 7,
            "tro": 8,
            "ck": 9,
        }
        self.i2t_map = {v: k for k, v in self.t2i_map.items()}
        self.tests = [
            "cbc",
            "ele",
            "cal",
            "inr",
            "lvr",
            "lac",
            "abg",
            "cre",
            "tro",
            "ck",
        ]

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
            num_pts = int(et // 6 - 1)
            assert num_pts >= 1
            if num_pts == 1:
                self.total_idx.append((idx, self.window_size))
            else:
                labT = []
                for idf, t in zip(ind_f, T):
                    if idf in labidfs:
                        labT.append(t)

                for i in range(num_pts):
                    pt = (i + 1) * 6
                    lower = max(0, pt - self.past_hours)
                    upper = min(et, pt + self.future_hours)
                    arrT = np.array(labT, dtype=float)
                    # arrT = np.array(T, dtype=float)
                    if (
                        len(arrT[(arrT >= lower) & (arrT <= pt)])
                        >= self.min_num_vals_past
                        and len(arrT[(arrT <= upper) & (arrT >= pt)])
                        >= self.min_num_vals_future
                    ):
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
        # print(labfns)
        # print(binaryfns)
        # print(valfns)
        # print(valfns)
        # print(feat2idx)
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
        # print(len(self.feature_norms), len(self.normfns), len(self.clipfns))
        # missing = set(self.feature_norms.keys()).difference(
        #     set(self.normfns).union(set(self.clipfns))
        # )
        # print(missing)
        assert len(self.feature_norms) == len(self.normfns) + len(self.clipfns) + 3
        self.labidx_map = {}
        for i in self.i2t_map:
            fns = self.t2f_map[self.i2t_map[i]]
            res = []
            for f in fns:
                res.append(self.feat2idx[f])
            self.labidx_map[i] = res
        # print(self.labidx_map)

        self.ro_disagree = 0
        self.total_ls = 0
        self.rcts = [0] * self.num_tests
        self.octs = [0] * self.num_tests
        self.ucts = [0] * self.num_tests
        return

    def _get_one_stay(
        self, sparse_m, point, etime, imputation="mean", normalization="clip"
    ):
        pt = point
        lower = max(0, pt - self.past_hours)
        upper = min(etime, pt + self.future_hours)

        xdic, ydic = {}, {}
        alldic = {}
        # get x and y
        for k, v in sparse_m.items():
            xvals, yvals = [], []
            allvals = []
            for tup in v:
                t, y = tup
                if t >= lower and t <= pt:
                    xvals.append(tup)
                if t >= pt and t <= upper:
                    yvals.append(tup)
                if t >= lower and t <= upper:
                    allvals.append(tup)
            if len(xvals) > 0:
                xdic[k] = xvals
            if len(yvals) > 0:
                ydic[k] = yvals
            if len(allvals) > 0:
                alldic[k] = allvals
        # get rule and observed lab labels
        reasons, rule_label, obs_label = self._get_label(xdic, ydic, point)
        _, wyrule_label, _ = self._get_label(alldic, ydic, point)
        # print(reasons, rule_label)
        # print(obs_label)
        # make upper upper bound, and count diff
        ct = 0
        for idx, (a, b) in enumerate(zip(rule_label, obs_label)):
            if a == 1 and b == 0:
                ct += 1
            if a == 1:
                self.rcts[idx] += 1
            if b == 1:
                self.octs[idx] += 1

        self.ro_disagree += ct
        self.total_ls += len(obs_label)
        true_obs = [i for i in obs_label]
        for idx, l in enumerate(rule_label):
            if l == 1:
                obs_label[idx] = 1
        for idx, l in enumerate(obs_label):
            if l == 1:
                self.ucts[idx] += 1

        # normalize x and y
        normx = self._normalize_stay(xdic, point, isx=True)
        normy = self._normalize_stay(ydic, point, isx=False)
        # print(normx.shape, normy.shape)

        # impute x and y
        impx = self._impute_stay(normx)
        impy = self._impute_stay(normy)

        xm = torch.from_numpy(np.zeros_like(normx)).float()
        ym = torch.from_numpy(np.zeros_like(normy)).float()
        assert len(obs_label) == len(rule_label) == self.num_tests
        rule = torch.tensor(rule_label).float()
        obs = torch.tensor(obs_label).float()
        true_obs = torch.tensor(true_obs).float()
        wyrule_label = torch.tensor(wyrule_label).float()
        return impx, impy, wyrule_label, true_obs, rule, obs

    def _get_label(self, xdic, ydic, pt):
        hist_label = [0] * self.num_tests
        for k in ydic:
            # print(k)
            fn = k.split("_")[0]
            if fn in self.f2t_map:
                # print(fn)
                tidx = self.t2i_map[self.f2t_map[fn]]
                hist_label[tidx] = 1

        rule_label = [0] * self.num_tests
        reasons, testl = self._apply_rules(xdic, ydic, pt)
        testl = list(set(testl))
        for t in testl:
            rule_label[self.t2i_map[t]] = 1
        return (reasons, rule_label, hist_label)

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

    def _apply_rules(self, xdic, ydic, curr_time):
        # print(bfdic)
        res = []
        reasons = []
        # "cbc",
        # "ele",
        # "cal",
        # "inr",
        # "lvr",
        # "lac",
        # "abg",
        # "cre",
        # "tro",
        # "ck",

        # print(xdic)
        bfdic = {}
        for k, v in xdic.items():
            nvs = [(j, i) for (i, j) in v]
            bfdic[k] = nvs
        # print(bfdic)

        names = list(set([x.split("_")[0] for x in list(bfdic.keys())]))
        if "Transfusions" in names:
            s = "bldtrans-cbc"
            res.append("cbc")
            reasons.append(s)
            s = "bldtrans-ele"
            res.append("ele")
            reasons.append(s)
            s = "bldtrans-inr"
            res.append("inr")
            reasons.append(s)

        if "UrineOutput" in names:
            uohigh = False
            uolow = False
            v = 0
            for fn in bfdic:
                if "UrineOutput" in fn:
                    for i in bfdic[fn]:
                        if i[0] > 0:
                            v += i[0]
            uoval = v / 1000
            if uoval > 4:
                uohigh = True
            elif uoval < 1 and uoval > 0:
                uolow = True
            if uohigh or uolow:
                s = "urineoutput-ele"
                res.append("ele")
                reasons.append(s)
                s = "urineoutput-cre"
                res.append("cre")
                reasons.append(s)

        if "Vasopressors" in names:
            vasos = 0
            for fn in bfdic:
                if "Vasopressors" in fn:
                    vasos += 1
                    vals = [i[0] for i in bfdic[fn]]
                    if len(vals) > 1:
                        if has_25_percent_increase(vals):
                            s = "vaso-cbc"
                            reasons.append(s)
                            res.append("cbc")
                            s = "vaso-lvr"
                            reasons.append(s)
                            res.append("lvr")
                            s = "vaso-lac"
                            reasons.append(s)
                            res.append("lac")
                            s = "vaso-tro"
                            reasons.append(s)
                            res.append("tro")
            if vasos > 1:
                # new vaso
                s = "vaso-cbc"
                reasons.append(s)
                res.append("cbc")
                s = "vaso-lvr"
                reasons.append(s)
                res.append("lvr")
                s = "vaso-lac"
                reasons.append(s)
                res.append("lac")
                s = "vaso-tro"
                reasons.append(s)
                res.append("tro")

        if "Dialysis" in names:
            s = "dialysis-cal"
            res.append("cal")
            reasons.append(s)

        if "Temperature" in names:
            fever = False
            for fn in bfdic:
                if "Temperature" in fn:
                    for i in bfdic[fn]:
                        if i[0] > 37.8:
                            fever = True
            if fever:
                s = "fever-cbc"
                res.append("cbc")
                reasons.append(s)
                s = "fever-lvr"
                res.append("lvr")
                reasons.append(s)

        if "MinuteVentilation" in names:
            for fn in bfdic:
                if "MinuteVentilation" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    if len(vals) > 1:
                        if has_25_percent_increase(vals):
                            s = "mvent-abg"
                            reasons.append(s)
                            res.append("abg")

        if "AirwayPressure" in names:
            for fn in bfdic:
                if "AirwayPressure" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    if len(vals) > 1:
                        if has_25_percent_increase(vals):
                            s = "airpre-abg"
                            reasons.append(s)
                            res.append("abg")

        if "Antibiotics" in names:
            s = "antibio-cbc"
            res.append("cbc")
            reasons.append(s)

        if "Antiarrhythmics" in names:
            s = "antiarr-ele"
            reasons.append(s)
            res.append("ele")
            s = "antiarr-cal"
            reasons.append(s)
            res.append("cal")

        if "Anticoagulants" in names:
            s = "ancoag-inr"
            reasons.append(s)
            res.append("inr")

        if "Propofol" in names:
            s = "propo-ck"
            reasons.append(s)
            res.append("ck")

        if "ICPMonitor" in names:
            s = "icp-ele"
            reasons.append(s)
            res.append("ele")

        if "WBC" in names:
            for fn in bfdic:
                if "WBC" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for v in vals:
                        if v < 1 or v > 12:
                            s = "wbcval-wbc"
                            reasons.append(s)
                            res.append("cbc")
                            s = "wbcval-lvr"
                            reasons.append(s)
                            res.append("lvr")
                    if has_five_unit_change_in_past_24_hours(ts, vals, curr_time):
                        s = "wbcchg-wbc"
                        reasons.append(s)
                        res.append("cbc")

        if "Creatinine" in names:
            for fn in bfdic:
                if "Creatinine" in fn:
                    vals = [i[0] * 88.4 for i in bfdic[fn]]
                    order = False
                    for i in vals:
                        if i > 150:
                            order = True
                            break
                    if len(vals) > 1:
                        if has_50_increase(vals):
                            order = True
                    if order:
                        s = "creval-abg"
                        reasons.append(s)
                        res.append("abg")
                        s = "creval-ele"
                        reasons.append(s)
                        res.append("ele")
                        s = "creval-cal"
                        reasons.append(s)
                        res.append("cal")

        if "CreatinineKinase" in names:
            for fn in bfdic:
                if "CreatinineKinase" in fn:
                    orderck = False
                    # vals = [i[0] for i in bfdic[fn]]
                    for i in bfdic[fn]:
                        if i[0] > 5000:
                            orderck = True
                    if orderck:
                        s = "ckval-ck"
                        reasons.append(s)
                        res.append("ck")

        if "PEEP" in names:
            for fn in bfdic:
                if "PEEP" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    if len(vals) > 1:
                        if has_2_increase(vals):
                            s = "peep-abg"
                            reasons.append(s)
                            res.append("abg")

        if "ph" in names:
            for fn in bfdic:
                if "ph" in fn:
                    # vals = [i[0] for i in ]
                    for i in bfdic[fn]:
                        if i[0] < 7.3:
                            s = "ph-lac"
                            reasons.append(s)
                            res.append("lac")
                            s = "ph-cre"
                            reasons.append(s)
                            res.append("cre")

        if "Hemoglobin" in names:
            for fn in bfdic:
                if "Hemoglobin" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i < 7:
                            s = "hbval-cbc"
                            reasons.append(s)
                            res.append("cbc")
                            s = "hbval-inr"
                            reasons.append(s)
                            res.append("inr")

                    if has_2_unit_decrease_in_past_24_hours(ts, vals, curr_time):
                        s = "hbdec-cbc"
                        reasons.append(s)
                        res.append("cbc")

        if "Platelets" in names:
            for fn in bfdic:
                if "Platelets" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i < 30 or i > 600000:
                            s = "platval-cbc"
                            reasons.append(s)
                            res.append("cbc")
                            break
                    if has_30p_unit_decrease_in_past_48_hours(ts, vals, curr_time):
                        s = "pladec-cbc"
                        reasons.append(s)
                        res.append("cbc")

        if "KReplacement" in names:
            for fn in bfdic:
                if "KReplacement" in fn:
                    ts = [i[1] for i in bfdic[fn] if i[1] > curr_time - 12]
                    s = "krep12hr-ele"
                    reasons.append(s)
                    res.append("ele")

        if "CaReplacement" in names:
            for fn in bfdic:
                if "CaReplacement" in fn:
                    ts = [i[1] for i in bfdic[fn] if i[1] > curr_time - 12]
                    s = "carep12hr-cal"
                    reasons.append(s)
                    res.append("cal")

        if "PReplacement" in names:
            for fn in bfdic:
                if "PReplacement" in fn:
                    ts = [i[1] for i in bfdic[fn] if i[1] > curr_time - 12]
                    s = "prep12hr-cal"
                    reasons.append(s)
                    res.append("cal")

        if "MgReplacement" in names:
            for fn in bfdic:
                if "MgReplacement" in fn:
                    ts = [i[1] for i in bfdic[fn] if i[1] > curr_time - 12]
                    s = "mgrep12hr-cal"
                    reasons.append(s)
                    res.append("cal")

        if "Sodium" in names:
            for fn in bfdic:
                if "Sodium" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    if has_six_unit_change_in_past_24_hours(ts, vals, curr_time):
                        s = "na-ele"
                        reasons.append(s)
                        res.append("ele")
                    for i in vals:
                        if i < 135 or i > 150:
                            s = "na-ele"
                            reasons.append(s)
                            res.append("ele")

        if "Potassium" in names:
            for fn in bfdic:
                if "Potassium" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i > 5 or i < 3.5:
                            s = "k-ele"
                            reasons.append(s)
                            res.append("ele")
                            break
                    for i in vals:
                        if i > 4.5:
                            s = "k-cre"
                            reasons.append(s)
                            res.append("cre")
                            break

        if "Calcium" in names:
            for fn in bfdic:
                if "Calcium" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i > 3 * 4 or i < 2 * 4:
                            s = "cal-cal"
                            reasons.append(s)
                            res.append("cal")
                            break

        if "Phosphate" in names:
            for fn in bfdic:
                if "Phosphate" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i < 0.6 * 3.1 or i > 1.8 * 3.1:
                            s = "pho-cal"
                            reasons.append(s)
                            res.append("cal")
                            break

        if "Magnesium" in names:
            for fn in bfdic:
                if "Magnesium" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i < 0.8 * 2:
                            s = "mg-cal"
                            reasons.append(s)
                            res.append("cal")
                            break

        if "INR" in names:
            for fn in bfdic:
                if "INR" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i > 1.6:
                            s = "inr-inr"
                            reasons.append(s)
                            res.append("inr")
                            break

        if "ALT" in names:
            for fn in bfdic:
                if "ALT" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    ts = [i[1] for i in bfdic[fn]]
                    for i in vals:
                        if i > 100:
                            s = "alt-lvr"
                            reasons.append(s)
                            res.append("lvr")
                            break

        if "Bilirubin" in names:
            for fn in bfdic:
                if "Bilirubin" in fn:
                    vals = [i[0] for i in bfdic[fn]]
                    for i in vals:
                        if i * 17 > 50:
                            s = "bili-lvr"
                            reasons.append(s)
                            res.append("lvr")
                            break

        for t in res:
            if t not in self.t2i_map:
                print(t)
                raise ValueError
        return reasons, res
