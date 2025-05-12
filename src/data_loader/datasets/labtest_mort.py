import os, sys, tarfile, collections, json, pickle
import torch
from os import listdir
import numpy as np
from torch.utils.data import Dataset
import time, math


def fill_zeros(arr):
    # Convert list to numpy array for vectorized operations
    arr = np.array(arr)

    # If the entire array is zeros, return the same array
    if np.all(arr == 0):
        return arr

    # Identify the indices of non-zero values
    non_zero_indices = np.nonzero(arr)[0]

    # Fill leading zeros with the first non-zero value
    if non_zero_indices[0] != 0:
        arr[: non_zero_indices[0]] = arr[non_zero_indices[0]]

    # Interpolate zeros between non-zero values
    for start, end in zip(non_zero_indices, non_zero_indices[1:]):
        gap = end - start
        # Create a linear space between the two non-zero values
        interpolated_values = np.linspace(arr[start], arr[end], gap + 1)
        arr[start : end + 1] = interpolated_values

    # Fill trailing zeros with the last non-zero value
    if non_zero_indices[-1] != len(arr) - 1:
        arr[non_zero_indices[-1] :] = arr[non_zero_indices[-1]]

    return arr


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


class InvalidPatientException(Exception):
    pass


class PatientMortalityDataset(Dataset):
    """Dataset for mortality for MIMIC-IV"""

    def __init__(
        self, path=None, split="train", preproc_params=None, imputation="mean"
    ):
        self.path = path
        self.split = split
        self.preproc_params = preproc_params
        assert self.split in ["train", "val", "test"]

        with open(self.path, "rb") as f:
            data_dict = pickle.load(f)
        print(data_dict.keys())
        assert f"{split}_mask" in data_dict
        mask = data_dict[f"{split}_mask"]
        self._setup_params()
        self._preprocess_dataset(data_dict, mask)
        self.imputation = imputation
        # assert 1 == 2
        print(len(self.labels))

        # res = []
        # for i in range(900):
        #     res.append(len(self.Xs[i]))
        # print(res, max(res), min(res))
        # print(self.Xs[5])

    def __len__(self):
        # ct = 0
        # for i in self.num_ints:
        #    ct += int(i)
        # assert len(self.labels) == ct
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            T, Y, X, ind_kt, ind_kf, cov, label = (
                self.Ts[idx],
                self.Ys[idx],
                self.Xs[idx],
                self.ind_kts[idx],
                self.ind_kfs[idx],
                self.covs[idx],
                self.labels[idx],
            )

            patient_data = (X, T, Y, ind_kt, ind_kf, cov, label)
            x, y = self._proc_one_datapoint(patient_data)
            # print(x.shape, y)
        elif isinstance(idx, slice):
            Ts, Ys, Xs, ind_kts, ind_kfs, covs, labels = (
                self.Ts[idx],
                self.Ys[idx],
                self.Xs[idx],
                self.ind_kts[idx],
                self.ind_kfs[idx],
                self.covs[idx],
                self.labels[idx],
            )

            xs, ys = [], []
            for T, Y, X, ind_kt, ind_kf, cov, label in zip(
                Ts, Ys, Xs, ind_kts, ind_kfs, covs, labels
            ):
                patient_data = (X, T, Y, ind_kt, ind_kf, cov, label)
                x, y = self._proc_one_datapoint(patient_data)
                xs.append(x)
                ys.append(y)
            x = xs
            y = ys
        else:
            raise NotImplementedError

        return x, y

    def get_label_weights(self):
        lb, cts = np.unique(self.labels, return_counts=True)

        res = np.ones(len(self.labels))
        for l in lb:
            res[self.labels == l] = res[self.labels == l] / cts[l]
        assert len(res) == len(self.labels)
        return res

    def _proc_one_datapoint(self, patient_data):
        if self.imputation == "mean":
            labtest, label = self._mean_imputation(patient_data)
            labtest = labtest.astype(np.float64)
            labtest = torch.from_numpy(labtest).float()
            label = torch.tensor(label, dtype=torch.int32)
        elif self.imputation == "last":
            labtest, label = self._mean_imputation(patient_data)
            labtest = labtest.astype(np.float64)
            labtest = torch.from_numpy(labtest).float()
            label = torch.tensor(label, dtype=torch.int32)
        else:
            raise NotImplementedError

        return labtest, label

    def _mean_imputation(self, patient_data):
        X, T, Y, ind_kt, ind_kf, cov, label = patient_data

        sum_testres = np.zeros((self.num_X_pred, self.num_feats))
        count_testres = np.zeros((self.num_X_pred, self.num_feats))
        # print(list(self.f2t.keys()), len(self.f2t))

        final_fi2i = dict()
        for idx, f in enumerate(self.f2t):
            final_fi2i[self.f2i[f]] = idx
        # print(final_fi2i)

        assert len(Y) > 0
        if len(Y) < 6:
            the_interval = 6 - len(Y)
        else:
            the_interval = 0
        for y, ind_f, ind_t in zip(Y, ind_kf, ind_kt):
            # the_interval = #max(int(((T[ind_t]) - X[0]) // self.X_interval), 0)
            # if (
            #    the_interval == self.num_X_pred
            #    and (T[ind_t] - (X[0])) == the_interval * self.X_interval
            # ):
            #    the_interval -= 1
            for yy, idf, idt in zip(y, ind_f, ind_t):
                if idf in final_fi2i:
                    sum_testres[the_interval, final_fi2i[idf]] += yy
                    count_testres[the_interval, final_fi2i[idf]] += 1
            the_interval += 1
        testres = np.divide(
            sum_testres,
            count_testres,
            out=np.zeros_like(sum_testres),
            where=(count_testres != 0),
        )
        # fill in the zeross
        # for c in range(testres.shape[1]):
        #     if sum(count_testres[:, c]) == 0:
        #         # fill in the mean
        #         testres[:, c] = 0.5
        #         # print("here")
        #     else:
        #         # print(testres[:, c])
        #         interp_res = fill_zeros(testres[:, c])
        #         testres[:, c] = interp_res
        #         # print(testres[:, c])

        # print(testres)
        # print(count_testres)

        # assert 1 == 2
        concat_arr = [testres]
        if not isinstance(cov, np.ndarray):
            cov = np.array(cov)
        the_covs = np.tile(np.expand_dims(cov, 0), (self.num_X_pred, 1))
        concat_arr.append(the_covs)

        testres = np.concatenate(concat_arr, axis=-1)
        return testres, label

    def _setup_params(self):
        if self.preproc_params is not None:
            preproc_params = preproc_params
        else:
            preproc_params = dict()
            preproc_params["before_end"] = 0.0
            preproc_params["include_before_death"] = 48.01  # 24.01
            preproc_params["data_interval"] = 6
            preproc_params["X_interval"] = 8
            preproc_params["num_X_pred"] = 6
            preproc_params["num_hours_pred"] = 24
            preproc_params["num_hours_warmup"] = 0  # 3
            preproc_params["min_measurements_in_warmup"] = 0  # 5
            preproc_params["neg_sampled"] = True
            # if "iii" in self.path:
            #     preproc_params["num_feats"] = 39
            # else:
            preproc_params["num_feats"] = 21
            preproc_params["num_covs"] = 38
            self.preproc_params = preproc_params

        self.before_end = preproc_params["before_end"]
        self.include_before_death = preproc_params["include_before_death"]
        self.data_interval = preproc_params["data_interval"]
        self.X_interval = preproc_params["X_interval"]
        self.num_X_pred = preproc_params["num_X_pred"]
        self.num_hours_pred = preproc_params["num_hours_pred"]
        self.num_hours_warmup = preproc_params["num_hours_warmup"]
        self.min_measurements_in_warmup = preproc_params["min_measurements_in_warmup"]
        self.neg_sampled = preproc_params["neg_sampled"]
        self.num_feats = preproc_params["num_feats"]
        self.num_covs = preproc_params["num_covs"]

    def _preprocess_dataset(self, data_dict, mask):
        """
        'i2f', 'f2i', 'testsn', 't2f',
        'f2t', 'i2t', 't2i', 'cov_names', 'feature_stats',
        'sids', 'num_intervals', 'Ts', 'values', 'labels',
        'etimes', 'labeltups', 'covs'
        """
        import time

        self.i2f = data_dict["i2f"]
        self.f2i = data_dict["f2i"]
        self.i2t = data_dict["i2t"]
        self.t2i = data_dict["t2i"]
        self.f2t = data_dict["f2t"]
        self.t2f = data_dict["t2f"]
        self.feature_stats = data_dict["feature_stats"]

        self.stay_ids = np.array(data_dict["sids"])[mask]
        self.num_ints = np.array(data_dict["num_intervals"])[mask]
        self.Ts = np.array(data_dict["Ts"], dtype=object)[mask]
        self.values = np.array(data_dict["values"], dtype=object)[mask]
        Ys, ind_kts, ind_kfs = [], [], []
        for tup in self.values:
            _, y, idt, idf = tup
            Ys.append(y)
            ind_kts.append(idt)
            ind_kfs.append(idf)
        self.Ys, self.ind_kts, self.ind_kfs = (
            np.array(Ys, dtype=object),
            np.array(ind_kts, dtype=object),
            np.array(ind_kfs, dtype=object),
        )
        self.covs = np.array(data_dict["covs"], dtype=object)[mask]
        self.labels = np.array(data_dict["labels"], dtype=np.float64)[mask]
        self.etimes = np.array(data_dict["etimes"], dtype=np.float64)[mask]
        self.labeltups = np.array(data_dict["labeltups"], dtype=object)[mask]

        assert len(self.stay_ids) == len(self.Ts)
        assert len(self.Ts) == len(self.Ys)
        assert len(self.Ys) == len(self.ind_kfs)
        assert len(self.ind_kfs) == len(self.ind_kts)
        assert len(self.ind_kts) == len(self.labels)
        assert len(self.labels) == len(self.etimes)
        assert len(self.covs) == len(self.labels)

        self.num_skip_patients = 0
        kept_idx = self._validate()
        num_patients = len(self.stay_ids)
        print(
            "Skip %d patients out of total %d patients"
            % (self.num_skip_patients, num_patients)
        )

        self.stay_ids = self.stay_ids[kept_idx]
        self.num_ints = self.num_ints[kept_idx]
        self.values = self.values[kept_idx]
        self.Ts = self.Ts[kept_idx]
        self.Ys = self.Ys[kept_idx]
        self.ind_kts = self.ind_kts[kept_idx]
        self.ind_kfs = self.ind_kfs[kept_idx]
        self.labels = self.labels[kept_idx]
        self.etimes = self.etimes[kept_idx]
        self.covs = self.covs[kept_idx]
        self.labeltups = self.labeltups[kept_idx]
        assert len(self.valid_time_pairs) == len(self.stay_ids)
        assert len(self.stay_ids) == len(self.Ts)
        assert len(self.Ts) == len(self.Ys)
        assert len(self.Ys) == len(self.ind_kfs)
        assert len(self.ind_kfs) == len(self.ind_kts)
        assert len(self.ind_kts) == len(self.labels)
        assert len(self.labels) == len(self.etimes)
        assert len(self.covs) == len(self.labels)
        assert len(self.valid_time_pairs) == len(self.stay_ids)

        # clip and norm ys
        tempYs = []
        for Ys, ind_kfs in zip(self.Ys, self.ind_kfs):
            tempYs.append(self._norm_Y(Ys, ind_kfs))

        self.Ys = tempYs
        assert len(self.Ys) == len(self.labels)

        self._make_inputs()
        print(f"valid stay_ids {len(self.stay_ids)}")
        print(f"proccessed len {len(self.labels)}")

    def _norm_Y(self, Y, ind_kf):
        upd, lowd, meand, stdd = (
            self.feature_stats["upperbound_dict"],
            self.feature_stats["lowerbound_dict"],
            self.feature_stats["mean_dict"],
            self.feature_stats["std_dict"],
        )
        resY, residt = [], []
        # print(Y)
        for y, idfs in zip(Y, ind_kf):
            tpy = []
            for yy, idf in zip(y, idfs):
                fn = self.i2f[idf]
                if yy > upd[fn]:
                    yy = upd[fn]
                elif yy < lowd[fn]:
                    yy = lowd[fn]

                if fn == "dialysis":
                    yy = 1.0
                else:
                    yy = (yy - lowd[fn]) / (upd[fn] - lowd[fn])
                    if yy < 0:
                        yy = 0
                    elif yy > 1:
                        yy = 0
                # print(yy)
                # print(fn, lowd[fn], upd[fn])
                # assert yy >= 0 and yy <= 1
                tpy.append(yy)
            assert len(tpy) == len(idfs)
            resY.append(tpy)
        assert len(resY) == len(Y)
        # print(resY)
        # print("==========")
        return resY

    def _make_inputs(self):
        Xs, Ts, Ys, ind_kts, ind_kfs, labels = [], [], [], [], [], []
        patient_inds, the_steps, total_steps, time_to_ends, covs = [], [], [], [], []

        for idx, (T, Y, ind_kt, ind_kf, end_time, label, cov, pair, ints) in enumerate(
            zip(
                self.Ts,
                self.Ys,
                self.ind_kts,
                self.ind_kfs,
                self.etimes,
                self.labels,
                self.covs,
                self.valid_time_pairs,
                self.num_ints,
            )
        ):
            stay_data = (T, Y, ind_kt, ind_kf, end_time, label, cov, pair, idx, ints)
            this_stay_datapoints = self._load_each_stay(stay_data)
            for p in this_stay_datapoints:
                (
                    new_Y,
                    new_X,
                    new_T,
                    new_ind_kf,
                    new_ind_kt,
                    new_labels,
                    new_time_to_end,
                    patient_idx,
                    step,
                    total_step,
                    cov,
                ) = p
                Xs.append(new_X)
                Ts.append(new_T)
                Ys.append(new_Y)
                ind_kts.append(new_ind_kt)
                ind_kfs.append(new_ind_kf)
                labels.append(new_labels)
                patient_inds.append(patient_idx)
                the_steps.append(step)
                total_steps.append(total_step)
                time_to_ends.append(time_to_ends)
                covs.append(cov)
        self.Xs = Xs
        self.Ts = Ts
        self.Ys = Ys
        self.ind_kts = ind_kts
        self.ind_kfs = ind_kfs
        self.labels = labels
        self.patient_inds = patient_idx
        self.the_steps = the_steps
        self.total_steps = total_steps
        self.time_to_ends = time_to_ends
        self.covs = covs

    def _load_each_stay(self, stay_data):
        T, Y, ind_kt, ind_kf, end_time, label, cov, pair, patient_idx, ints = stay_data
        start_ind, end_ind = pair
        index_paramid = self._get_array_index(
            start_ind, end_ind, end_time, T, ind_kt, ints
        )

        result = []
        for time_index_tuple in index_paramid:
            # ((0, i + 1), et, i, ints)
            ind_arr, end_pred_time, step, total_step = time_index_tuple
            assert len(ind_arr) > 0
            # Construct X
            # start_time = max(end_pred_time - self.num_X_pred, 0)
            start_time = max(end_pred_time - 48, 0)  # 48 hours
            if end_pred_time % 8 == 0 or end_pred_time // 8 == 0:
                new_X = np.arange(end_pred_time, start_time, -self.X_interval)[::-1]
            else:
                new_X = np.append(
                    np.arange(step * 8, start_time, -self.X_interval)[::-1],
                    end_pred_time,
                )
            # print(new_X)

            new_labels = (
                0
                if label == 0
                else int(end_pred_time + self.num_hours_pred >= end_time)
            )
            new_time_to_end = end_time - end_pred_time

            # offset = ind_kt[ind_arr[0]]
            s, e = ind_arr
            new_Y = Y[s:e]  # [Y[idx] for idx in ind_arr]
            new_ind_kt = ind_kt[s:e]  # [ind_kt[idx] - offset for idx in ind_arr]
            new_ind_kf = ind_kf[s:e]  # [ind_kf[idx] for idx in ind_arr]
            # new_T = T[offset : (new_ind_kt[-1] + 1 + offset)]

            allkts = list(set(list(np.concatenate(new_ind_kt))))
            if len(allkts) > 1:
                minkt = min(allkts)
                maxkt = max(allkts)
                # print(minkt, maxkt, new_ind_kt)
                new_T = T[int(minkt) : int(maxkt)]
            elif len(allkts) == 1:
                new_T = T[int(allkts[0])]
            else:
                new_T = []

            # TODO: no gs hs rea
            new_data_tup = (
                new_Y,
                new_X,
                new_T,
                new_ind_kf,
                new_ind_kt,
                new_labels,
                new_time_to_end,
                patient_idx,
                step,
                total_step,
                cov,
            )
            result.append(new_data_tup)
        return result

    def _get_array_index(self, start_ind, end_ind, end_time, T, ind_kt, ints):
        res_tup = []

        for i in range(int(ints)):
            # 6  = 48 hours / 8 hours split
            et = (i + 1) * 8 if i < int(ints) - 1 else end_time
            if i < 6:
                res_tup.append(((0, i + 1), et, i, ints))
            else:
                res_tup.append(((i + 1 - 6, i + 1), et, i, ints))
        return res_tup

    def _validate(self):
        kept_idx = np.zeros(len(self.labels))
        self.valid_time_pairs = []
        for idx, (T, Y, ind_kt, ind_kf, end_time, label) in enumerate(
            zip(
                self.Ts,
                self.Ys,
                self.ind_kts,
                self.ind_kfs,
                self.etimes,
                self.labels,
            )
        ):
            try:
                start_ind, end_ind = self._is_valid(
                    T, Y, ind_kt, ind_kf, end_time, label
                )
                # kept_idx.append(idx)
                kept_idx[idx] = 1
                self.valid_time_pairs.append((start_ind, end_ind))
            except InvalidPatientException as e:
                self.num_skip_patients += 1
                # print(e)
                continue

        # kept_idx = np.array(
        #     [True if i in kept_idx else False for i in range(len(self.labels))]
        # )
        kept_idx = kept_idx.astype(np.bool)
        return kept_idx

    def _is_valid(self, T, Y, ind_kt, ind_kf, end_time, label):
        # print(T, Y, ind_kt, ind_kf)
        # print([len(v) for v in ind_kt])
        len_kt = int(sum([len(v) for v in ind_kt]))
        # if self.min_measurements_in_warmup >= len_kt:
        if 5 >= len_kt:
            error_msg = (
                "Only have %d measurements but required min %d number. Discard."
                % (len(ind_kt), self.min_measurements_in_warmup)
            )
            raise InvalidPatientException(error_msg)

        # print(len(T), [len(v) for v in ind_kt])
        mxkt = -1
        for kt in ind_kt:
            if len(kt) > 0:
                mxkt = max(kt)
        if T[mxkt] > end_time:
            raise InvalidPatientException(
                "Patient has measurements after the end time..."
            )

        start_ind = self.min_measurements_in_warmup

        end_ind = len_kt - 1

        if end_ind < start_ind:
            raise InvalidPatientException("End index and start index has no overlap!")

        return start_ind, end_ind


class LabRLDatasetMIMIC(Dataset):
    """dataset for prep RL exp"""

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
        print(f"Load data took: {e-s} s")
        assert f"{split}_mask" in data_dict
        mask = data_dict[f"{self.split}_mask"]

        s = time.time()
        self._preproc_dataset(data_dict, mask)
        assert len(self.sparse_ms) > 0
        assert self.total_idx
        e = time.time()
        print(f"Preproc data took: {e-s} s")

        self.imputation = imputation
        self.normalization = "clip"

    def __len__(self):
        return len(self.total_idx)

    def __getitem__(self, idx):
        pair = self.total_idx[idx]
        smidx, pt = pair
        sm = self.sparse_ms[smidx]
        et = self.etimes[smidx]
        label = self.labels[smidx]
        sid = self.sids[smidx]
        x, y, hist_label = self._get_one_stay(
            sm, pt, et, imputation=self.imputation, normalization=self.normalization
        )
        if pt + self.future_hours >= et:
            l = label
        else:
            l = 0
        mort = self.labels[smidx]
        label = int(l)  # torch.tensor(l, dtype=torch.int32)

        # return x, y, obs_act, label, sid here
        action = hist_label
        return x, y, action, sid, mort, label

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

        hist_label = [0] * self.num_tests
        for k in ydic:
            # print(k)
            fn = k.split("_")[0]
            if fn in self.f2t_map:
                # print(fn)
                tidx = self.t2i_map[self.f2t_map[fn]]
                hist_label[tidx] = 1

        # normalize x and y
        normx = self._normalize_stay(xdic, point, isx=True)
        normy = self._normalize_stay(ydic, point, isx=False)

        # impute x and y
        impx = self._impute_stay(normx)
        impy = self._impute_stay(normy)

        return impx, impy, hist_label

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

        self.window_size = 12
        self.future_hours = 24
        self.past_hours = 24
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
        self.post_labels = []
        for idx, (sm, et, T, ind_f, ind_t) in enumerate(
            zip(self.sparse_ms, self.etimes, self.Ts, self.ind_kfs, self.ind_kts)
        ):
            num_pts = int(et // self.window_size - 1)
            assert num_pts >= 0
            if num_pts <= 1:
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
                    # arrT = np.array(T, dtype=float)
                    if (
                        len(arrT[(arrT >= lower) & (arrT <= pt)])
                        >= self.min_num_vals_past
                        and len(arrT[(arrT <= upper) & (arrT >= pt)])
                        >= self.min_num_vals_future
                    ):
                        self.total_idx.append((idx, pt))
                        if pt + self.future_hours >= et:
                            l = self.labels[idx]
                        else:
                            l = 0
                        self.post_labels.append(l)
                    elif pt + self.future_hours >= et and self.labels[idx] > 0:
                        self.total_idx.append((idx, pt))
                        l = self.labels[idx]
                        self.post_labels.append(l)
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
        

        assert len(self.feature_norms) == len(self.normfns) + len(self.clipfns) + 3
        self.labidx_map = {}
        for i in self.i2t_map:
            fns = self.t2f_map[self.i2t_map[i]]
            res = []
            for f in fns:
                res.append(self.feat2idx[f])
            self.labidx_map[i] = res
        return

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
        if self.imputation == "mean":
            res = torch.from_numpy(stay).float()
            return res
        elif self.imputation == "interpolate":
            raise NotImplementedError
        elif self.imputation == "right":
            raise NotImplementedError
        else:
            raise NotImplementedError


class LabMortDatasetMIMIC(Dataset):
    """Dataset for mortality for MIMIC-IV"""

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
        print(f"Load data took: {e-s} s")
        assert f"{split}_mask" in data_dict
        mask = data_dict[f"{self.split}_mask"]

        s = time.time()
        self._preproc_dataset(data_dict, mask)
        assert len(self.sparse_ms) > 0
        assert self.total_idx
        e = time.time()
        print(f"Preproc data took: {e-s} s")

        self.imputation = imputation
        self.normalization = "clip"

    def get_label_weights(self):
        lb, cts = np.unique(self.post_labels, return_counts=True)

        res = np.ones(len(self.post_labels))
        print(lb, cts)
        # l = 1
        # print(self.labels == l, type(self.labels == l))
        # a = self.labels == l
        # print(a.shape, res.shape)

        for l in lb:
            a = list(self.post_labels == l)
            l = int(l)
            res[a] = res[a] / cts[l]
        assert len(res) == len(self.post_labels)
        return res

    def __len__(self):
        return len(self.total_idx)

    def __getitem__(self, idx):
        pair = self.total_idx[idx]
        smidx, pt = pair
        sm = self.sparse_ms[smidx]
        et = self.etimes[smidx]
        label = self.labels[smidx]
        x = self._get_one_stay(
            sm, pt, et, imputation=self.imputation, normalization=self.normalization
        )
        if pt + self.future_hours >= et:
            l = label
        else:
            l = 0
        y = torch.tensor(l, dtype=torch.int32)
        return x, y

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

        self.window_size = 6
        self.future_hours = 24
        self.past_hours = 24
        self.include_before_death = -1
        self.before_end = 0

        self.min_num_vals_past = 10
        self.min_num_vals_future = 5

        print(len(self.sids))

        labfns = self.fns_dict["labfns"]
        labidfs = []
        for f in self.f2i:
            if f.split("_")[0] in labfns:
                labidfs.append(self.f2i[f])
        # self._get_sparse_matrices()
        self.total_idx = []
        self.post_labels = []
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
                        if pt + self.future_hours >= et:
                            l = self.labels[idx]
                        else:
                            l = 0
                        self.post_labels.append(l)
                    elif pt + self.future_hours >= et and self.labels[idx] > 0:
                        self.total_idx.append((idx, pt))
                        l = self.labels[idx]
                        self.post_labels.append(l)
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
        # normy = self._normalize_stay(ydic, point, isx=False)

        # impute x and y
        impx = self._impute_stay(normx)
        # impy = self._impute_stay(normy)

        return impx

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
