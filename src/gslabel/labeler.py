import numpy as np
import pandas as pd
import time, pickle


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


class Gslabeler:
    def __init__(self, path, f2i, i2f, interval=8, num_test=10):
        self.path = path
        self.f2i = f2i
        self.i2f = i2f
        self.interval = interval
        self.num_test = num_test
        # self.day = day
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
        # print(self.f2t_map)
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

    def _split_stay(self, dp):
        sid, T, Y, ind_t, ind_f, label, etime = dp
        days = etime // 24
        rhrs = etime % 24
        # print(days, rhrs)
        # print(Y[:10])
        # print(ind_f[:10])
        T = [round(t, 2) for t in T]
        # print(T)
        rT = np.array(T.copy())
        rY = np.array(Y)
        ridt = np.array(ind_t)
        ridf = np.array(ind_f)
        # print("==================")

        ints = etime // self.interval
        if etime % self.interval > 0:
            ints += 1

        sTidx, sY, sidt, sidf = [], [], [], []
        for i in range(int(ints)):
            idxes = np.where(
                (rT >= i * self.interval) & (rT <= self.interval * i + self.interval)
            )[0]
            # print(idxes, "time")
            ytmp, idttmp, idftmp = [], [], []

            for elem in idxes:
                indices = np.where((ridt == elem))[0]
                idttmp.extend(ridt[indices])
                ytmp.extend(rY[indices])
                idftmp.extend(ridf[indices])

                ridt = np.delete(ridt, indices)
                rY = np.delete(rY, indices)
                ridf = np.delete(ridf, indices)

            sY.append(ytmp)
            sidt.append(idttmp)
            sidf.append(idftmp)
            sTidx.append(idxes)

        return sTidx, sY, sidt, sidf

    def gen_label(self, dp):
        sid, T, Y, ind_t, ind_f, label, etime = dp
        assert len(Y) == len(ind_t) == len(ind_f)
        # print(len(T), round(etime, 2), len(Y))

        sstay = dict()
        sTidx, sY, sidt, sidf = self._split_stay(dp)
        # for x, y, z in zip(sY, sidt, sidf):
        # print(len(x), len(y), len(z))

        ints = etime // self.interval
        if etime % self.interval > 0:
            ints += 1
        T = [round(t, 2) for t in T]
        out_dict = {
            "sid": sid,
            "ints": ints,
            "T": T,
            "splitvals": [sTidx, sY, sidt, sidf],
            "label": label,
            "etime": etime,
            "observation": [],
            "rule": [],
            "reasons": [],
            # "gshistreason": [],
        }
        for i in range(int(ints)):
            # 6 points
            backp = 24 // self.interval  # * 2
            # print(backp)
            beridx = i + 1
            belidx = i + 1 - backp if i + 1 >= backp else 0

            afterp = 24 // self.interval
            aflidx = i + 1
            afridx = i + 1 + afterp

            beY = sY[belidx:beridx]
            beidt = sidt[belidx:beridx]
            beidf = sidf[belidx:beridx]

            afY = sY[aflidx:afridx]
            afidt = sidt[aflidx:afridx]
            afidf = sidf[aflidx:afridx]

            # print(beridx * self.interval)
            # print(belidx, beridx, aflidx, afridx)
            this_time = beridx * self.interval
            sdp = (this_time, beY, beidt, beidf, afY, afidt, afidf, T)
            lbtup = self._get_label(sdp)
            reasons, rule_label, hist_label = lbtup
            out_dict["observation"].append(hist_label)
            out_dict["rule"].append(rule_label)
            out_dict["reasons"].append(reasons)
        return out_dict

    def _get_label(self, tp):
        _, beY, beidt, beidf, afY, afidt, afidf, T = tp
        T = [round(t, 2) for t in T]
        if len(afY) == 0:
            hist_label = None
        else:
            hist_label = self._get_hist_label(afidf)

        reasons, rule_label = self._get_rule_label(tp)
        return (reasons, rule_label, hist_label)

    def _get_hist_label(self, idf):
        hist_label = [0] * self.num_test
        vals = np.concatenate(idf)
        vals = [x for x in vals]
        indexes = np.unique(vals)

        for fi in indexes:
            fnlong = self.i2f[fi]
            fn = fnlong.split("_")[0]
            if fn in self.f2t_map:
                tidx = self.t2i_map[self.f2t_map[fn]]
                hist_label[tidx] = 1
        # print(indexes)
        return hist_label

    def _get_rule_label(self, tp):
        rule_label = [0] * self.num_test
        this_time, beY, beidt, beidf, afY, afidt, afidf, T = tp
        # print(beidt, afidt)
        bfdic = {}
        beY, beidt, beidf = (
            np.concatenate(beY),
            np.concatenate(beidt),
            np.concatenate(beidf),
        )
        # print(beidf)
        T = [round(t, 2) for t in T]
        beY = [round(y, 3) for y in beY if not isinstance(y, str)]
        for y, idt, idf in zip(beY, beidt, beidf):
            if y >= 0:
                idt = int(idt)
                idf = int(idf)
                # print(type(idf))print(idf)
                fn = self.i2f[int(idf)]
                if fn not in bfdic:
                    bfdic[fn] = [(y, T[idt])]
                else:
                    bfdic[fn].append((y, T[idt]))

        afdic = {}
        if len(afidf) > 0:
            afY, afidt, afidf = (
                np.concatenate(afY),
                np.concatenate(afidt),
                np.concatenate(afidf),
            )
            afY = [round(y, 3) for y in afY if not isinstance(y, str)]
            for y, idt, idf in zip(afY, afidt, afidf):
                idt = int(idt)
                idf = int(idf)

                fn = self.i2f[idf]
                if fn not in afdic:
                    afdic[fn] = [(y, T[idt])]
                else:
                    afdic[fn].append((y, T[idt]))

        reasons, testl = self._apply_rules(bfdic, afdic, this_time)
        testl = list(set(testl))
        for t in testl:
            rule_label[self.t2i_map[t]] = 1
        return reasons, rule_label

    def _apply_rules(self, bfdic, afdic, curr_time):
        # print(bfdic)
        res = []
        reasons = []
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

        names = list(set([x.split("_")[0] for x in list(bfdic.keys())]))
        if "Transfusions" in names:
            s = "bldtrans-cbc"
            res.append("cbc")
            reasons.append(s)
            s = "bldtrans-ele"
            res.append("ele")
            reasons.append(s)
            s = "bldtrans-inr"
            res.append("ele")
            reasons.append(s)

        if "UrineOutput" in names:
            uohigh = False
            uolow = False
            v = 0
            for fn in bfdic:
                if "UrineOutput" in fn:
                    for i in bfdic[fn]:
                        v += i[0]
            uoval = v / 1000
            if uoval > 4:
                uohigh = True
            elif uoval < 1:
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


