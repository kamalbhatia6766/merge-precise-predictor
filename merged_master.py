"""Merged master predictor combining SCR1–SCR9 behaviours.

This script centralises data loading, backtesting, and stake sizing while
keeping each legacy logic mathematically close to its original intent. Heavy
ML components are represented with lighter-weight equivalents to keep runtime
manageable inside a single file; section headers highlight which portion of the
legacy pipeline they mirror.
"""
from __future__ import annotations
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_excel_loader import load_results_excel, SLOT_NAMES, SLOT_MAP
from quant_data_core import load_results_dataframe

# ----------------------------- GLOBAL CONSTANTS -----------------------------
S40 = {
    "00","06","07","09","15","16","18","19","24","25","27","28",
    "33","34","36","37","42","43","45","46","51","52","54","55",
    "60","61","63","64","70","72","73","79","81","82","88","89",
    "90","91","97","98",
}

FAMILY_DIGITS = {"0","1","4","5","6","9"}
PACK_164950_FAMILY = {f"{a}{b}" for a in FAMILY_DIGITS for b in FAMILY_DIGITS}

BASE_STAKE = 10.0
RETURN_EXACT = 90
RETURN_DIGIT = 9


# ------------------------------ DATA UTILITIES ------------------------------
def two_digit(n: int) -> str:
    return f"{int(n) % 100:02d}"


def load_long_dataframe(path: str = "number prediction learn.xlsx") -> pd.DataFrame:
    df = load_results_excel(path)
    if not df.empty:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["slot"] = df["slot"].astype(int)
        df["number"] = df["number"].astype(int)
    return df


def detect_intraday(df: pd.DataFrame) -> Tuple[pd.Timestamp, List[int]]:
    latest_date = df["date"].max()
    latest_rows = df[df["date"] == latest_date]
    filled_slots = sorted(latest_rows["slot"].unique().tolist())
    return latest_date, filled_slots


# -------------------------- COMMON SCORING HELPERS -------------------------
def ewma_weights(n: int, halflife: float = 30.0) -> np.ndarray:
    if n <= 0:
        return np.array([])
    idx = np.arange(n)
    w = 0.5 ** ((n - 1 - idx) / halflife)
    s = w.sum()
    return (w / s) if s > 0 else np.ones(n) / n


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    arr = np.array(list(scores.values()), dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if math.isclose(mx, mn):
        return {k: 0.0 for k in scores}
    return {k: (float(v) - mn) / (mx - mn) for k, v in scores.items()}


def andar_bahar_from_predictions(preds: Sequence[int]) -> Tuple[int, int]:
    if not preds:
        return 0, 0
    tens = [p // 10 for p in preds]
    ones = [p % 10 for p in preds]
    return Counter(tens).most_common(1)[0][0], Counter(ones).most_common(1)[0][0]


# --------------------------- LOGIC: SCR1 (LIGHT) ---------------------------
def scr1_scores(df: pd.DataFrame, target_dow: int, slot: int) -> Dict[int, float]:
    slot_df = df[df["slot"] == slot].sort_values("date")
    if slot_df.empty:
        return {n: 0.0 for n in range(100)}

    weights = ewma_weights(len(slot_df), halflife=30.0)
    slot_df = slot_df.assign(w=weights)
    rec = {n: float(slot_df.loc[slot_df["number"] == n, "w"].sum()) for n in range(100)}
    rec = normalize_scores(rec)

    last_seen = int(slot_df.iloc[-1]["number"])
    trans_counts = {i: defaultdict(lambda: 1e-3) for i in range(100)}
    for i in range(1, len(slot_df)):
        prev_n = int(slot_df.iloc[i - 1]["number"])
        curr_n = int(slot_df.iloc[i]["number"])
        trans_counts[prev_n][curr_n] += 1
    trans = {}
    for i in range(100):
        row = trans_counts[i]
        ssum = sum(row.values())
        trans[i] = {j: row[j] / ssum for j in range(100)} if ssum else {j: 0.0 for j in range(100)}
    trans = normalize_scores(trans[last_seen])

    dow_df = df.copy()
    dow_df["dow"] = dow_df["date"].dt.dayofweek
    sub = dow_df[dow_df["dow"] == target_dow]
    vc = sub["number"].value_counts()
    mx = vc.max() if not vc.empty else 1
    dow_bonus = {int(n): (c / mx) for n, c in vc.items()}
    for n in range(100):
        dow_bonus.setdefault(n, 0.0)

    return {n: 0.5 * rec.get(n, 0.0) + 0.35 * trans.get(n, 0.0) + 0.15 * dow_bonus.get(n, 0.0) for n in range(100)}


def scr1_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    dow = pd.Timestamp(target_date).dayofweek
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        scores = scr1_scores(df, dow, slot)
        picks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_k]
        out[slot] = [num for num, _ in picks]
    return out


# --------------------------- LOGIC: SCR2 (ENSEMBLE) ------------------------
def scr2_gap_scores(numbers: List[int]) -> Dict[int, float]:
    positions: Dict[int, List[int]] = defaultdict(list)
    for idx, num in enumerate(numbers):
        positions[num].append(idx)
    current_idx = len(numbers)
    scores = {}
    for num in range(100):
        if num in positions and len(positions[num]) > 1:
            gaps = [positions[num][i] - positions[num][i - 1] for i in range(1, len(positions[num]))]
            avg_gap = np.mean(gaps)
            curr_gap = current_idx - positions[num][-1]
            scores[num] = curr_gap / avg_gap if avg_gap > 0 else 1.0
        else:
            scores[num] = 1.0
    return scores


def scr2_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        slot_nums = df[df["slot"] == slot]["number"].tolist()
        if len(slot_nums) < 5:
            out[slot] = slot_nums[-top_k:]
            continue
        freq = Counter(slot_nums[-90:])
        gap_scores = scr2_gap_scores(slot_nums)
        last = slot_nums[-1]
        trans = Counter()
        for i in range(1, len(slot_nums)):
            if slot_nums[i - 1] == last:
                trans[slot_nums[i]] += 1
        combined = {}
        for num in range(100):
            combined[num] = 0.4 * (freq.get(num, 0) / max(1, len(slot_nums[-90:])))
            combined[num] += 0.25 * min(gap_scores.get(num, 0), 3) / 3
            combined[num] += 0.15 * (trans.get(num, 0) / max(1, sum(trans.values())))
            digit_sum = sum(int(d) for d in two_digit(num))
            combined[num] += 0.1 * (digit_sum / 18)
        picks = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        out[slot] = [n for n, _ in picks[: top_k]]
    return out


# ----------------------- LOGIC: SCR3 (HYBRID ENSEMBLE) ---------------------
def scr3_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        nums = df[df["slot"] == slot]["number"].tolist()
        if len(nums) < 5:
            out[slot] = nums[-top_k:]
            continue
        freq_part = Counter(nums[-30:])
        gap_scores = scr2_gap_scores(nums)
        patterns = defaultdict(list)
        for i in range(len(nums) - 3):
            seq = tuple(nums[i : i + 3])
            patterns[seq].append(nums[i + 3])
        pattern_pred: List[int] = []
        recent_seq = tuple(nums[-3:])
        if recent_seq in patterns:
            pattern_pred = [n for n, _ in Counter(patterns[recent_seq]).most_common(3)]
        scores = Counter()
        for num in range(100):
            scores[num] += 0.35 * (freq_part.get(num, 0) / max(1, sum(freq_part.values())))
            scores[num] += 0.3 * (gap_scores.get(num, 0) / max(gap_scores.values()))
            if num in pattern_pred:
                scores[num] += 0.35
        picks = [n for n, _ in scores.most_common(top_k * 2)]
        # ensure diversity
        ranges = {"low": [], "mid": [], "high": []}
        for n in picks:
            if n <= 33:
                ranges["low"].append(n)
            elif n <= 66:
                ranges["mid"].append(n)
            else:
                ranges["high"].append(n)
        selected: List[int] = []
        for key in ["low", "mid", "high"]:
            if ranges[key]:
                selected.append(ranges[key][0])
        for n in picks:
            if len(selected) >= top_k:
                break
            if n not in selected:
                selected.append(n)
        out[slot] = selected[:top_k]
    return out


# ---------------- LOGIC: SCR4 & SCR5 (6-STRATEGY ENSEMBLE LITE) ------------
def scr45_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        nums = df[df["slot"] == slot]["number"].tolist()
        if len(nums) < 5:
            out[slot] = nums[-top_k:]
            continue
        bayes_freq = Counter(nums[-90:])
        gap_scores = scr2_gap_scores(nums)
        momentum_scores = Counter()
        if len(nums) > 20:
            recent = Counter(nums[-20:])
            hist = Counter(nums[-60:-20]) if len(nums) > 40 else Counter()
            for n in range(100):
                momentum_scores[n] = recent.get(n, 0) - hist.get(n, 0)
        markov = Counter()
        for i in range(1, len(nums)):
            if nums[i - 1] == nums[-1]:
                markov[nums[i]] += 1
        features = Counter()
        for num in range(100):
            features[num] = 0.25 * (bayes_freq.get(num, 0) / max(1, sum(bayes_freq.values())))
            features[num] += 0.2 * (gap_scores.get(num, 0) / max(gap_scores.values()))
            features[num] += 0.15 * (markov.get(num, 0) / max(1, sum(markov.values())))
            features[num] += 0.15 * (momentum_scores.get(num, 0) / max(1, max(momentum_scores.values()) if momentum_scores else 1))
        candidates = [n for n, _ in features.most_common(top_k * 2)]
        # intelligent filter similar to scr4/scr5
        ranges = {"low": [], "mid": [], "high": []}
        for n in candidates:
            if n <= 33:
                ranges["low"].append(n)
            elif n <= 66:
                ranges["mid"].append(n)
            else:
                ranges["high"].append(n)
        selected: List[int] = []
        for key in ["low", "mid", "high"]:
            if ranges[key]:
                selected.append(ranges[key][0])
        for n in candidates:
            if len(selected) >= top_k:
                break
            if n not in selected:
                selected.append(n)
        out[slot] = selected[:top_k]
    return out


# ----------------------- LOGIC: SCR6 (PATTERN BOOST) -----------------------
def scr6_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        nums = df[df["slot"] == slot]["number"].tolist()
        freq = Counter(nums[-50:])
        base_scores = {n: freq.get(n, 0) for n in range(100)}
        boosted = {}
        for num, val in base_scores.items():
            score = val
            num_str = two_digit(num)
            if num_str in S40:
                score += 3
            if num_str in PACK_164950_FAMILY:
                score += 1.5
            if nums and num == nums[-1]:
                score += 0.5
            boosted[num] = score
        picks = [n for n, _ in sorted(boosted.items(), key=lambda x: x[1], reverse=True)[: top_k]]
        out[slot] = picks
    return out


# ----------------------- LOGIC: SCR7 (VOTING ENGINE) -----------------------
def scr7_predict(component_preds: List[Dict[int, List[int]]], top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        vote = Counter()
        for pred in component_preds:
            for rank, num in enumerate(pred.get(slot, [])):
                vote[num] += 1 / (rank + 1)
        picks = [n for n, _ in vote.most_common(top_k)]
        out[slot] = picks
    return out


# ----------------------- LOGIC: SCR8 (PATTERN PACKS) -----------------------
SCR8_HOT = {
    "FRBD": [1, 3, 26, 44, 64, 71, 77, 82, 84],
    "GZBD": [20, 23, 50, 52, 67, 89],
    "GALI": [23, 28, 31, 32, 42, 57, 64, 72, 80, 94, 95],
    "DSWR": [25, 36, 48, 55, 68, 70, 88, 94, 96],
}

SCR8_COLD = {
    "FRBD": [21, 25, 48, 53, 61, 65, 85, 86, 87, 93, 97],
    "GZBD": [0, 1, 5, 12, 18, 21, 32, 46, 47, 48, 55, 66, 68, 84],
    "GALI": [0, 4, 10, 18, 29, 33, 37, 38, 43, 44, 48, 51, 52, 61, 65, 76, 77, 78, 88, 89],
    "DSWR": [7, 13, 19, 23, 27, 28, 32, 39, 52, 65, 69, 72, 81, 86, 95, 97],
}

DIGIT_BIAS = {
    "FRBD": {"tens": [0, 7, 4, 6, 5], "ones": [4, 2, 8, 0, 1]},
    "GZBD": {"tens": [9, 5, 2, 6, 8], "ones": [3, 9, 0, 1, 4]},
    "GALI": {"tens": [6, 9, 4, 3, 2], "ones": [2, 4, 1, 5, 7]},
    "DSWR": {"tens": [9, 5, 7, 8, 0], "ones": [8, 6, 4, 3, 1]},
}


def scr8_predict(target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot_name, slot_idx in SLOT_MAP.items():
        scores = Counter()
        for num in range(100):
            score = 0.0
            if num in SCR8_HOT.get(slot_name, []):
                score += 2.0
            if num in SCR8_COLD.get(slot_name, []):
                score -= 0.5
            tens = num // 10
            ones = num % 10
            if tens in DIGIT_BIAS.get(slot_name, {}).get("tens", []):
                score += 0.3
            if ones in DIGIT_BIAS.get(slot_name, {}).get("ones", []):
                score += 0.3
            scores[num] = score
        picks = [n for n, _ in scores.most_common(top_k)]
        out[slot_idx] = picks
    return out


# ----------------------- LOGIC: SCR9 (META ENSEMBLE) -----------------------
def scr9_predict(component_preds: List[Dict[int, List[int]]], roi_weights: Dict[str, float], top_k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for slot in [1, 2, 3, 4]:
        vote = Counter()
        for name, pred in zip(roi_weights.keys(), component_preds):
            weight = roi_weights.get(name, 1.0)
            for rank, num in enumerate(pred.get(slot, [])):
                vote[num] += weight / (rank + 1)
        picks = [n for n, _ in vote.most_common(top_k)]
        out[slot] = picks
    return out


# -------------------------- BACKTESTING & STAKING --------------------------
@dataclass
class LogicPerformance:
    name: str
    roi: float
    profit: float
    hits: int


def evaluate_day_predictions(actual: Dict[int, int], predicted: List[int], andar: int, bahar: int, stake: float) -> float:
    total_stake = stake * len(predicted) + 2 * stake
    returns = 0.0
    for num in predicted:
        if num == actual.get("number"):
            returns += RETURN_EXACT * stake
    if andar == actual.get("tens"):
        returns += RETURN_DIGIT * stake
    if bahar == actual.get("ones"):
        returns += RETURN_DIGIT * stake
    return returns - total_stake


def backtest_logic(df: pd.DataFrame, predictor, name: str, window: int = 60, top_k: int = 5) -> LogicPerformance:
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return LogicPerformance(name, 0.0, 0.0, 0)
    start_idx = max(1, len(dates) - window)
    profit = 0.0
    hits = 0
    stake = BASE_STAKE
    for i in range(start_idx, len(dates)):
        hist = df[df["date"] < dates[i]]
        if hist.empty:
            continue
        preds = predictor(hist, dates[i], top_k)
        day_rows = df[df["date"] == dates[i]]
        for _, row in day_rows.iterrows():
            slot = int(row["slot"])
            actual_num = int(row["number"])
            tens, ones = actual_num // 10, actual_num % 10
            pick_list = preds.get(slot, [])
            andar, bahar = andar_bahar_from_predictions(pick_list)
            profit += evaluate_day_predictions(
                {"number": actual_num, "tens": tens, "ones": ones},
                pick_list,
                andar,
                bahar,
                stake,
            )
            if actual_num in pick_list:
                hits += 1
    total_stake = (len(dates) - start_idx) * len([1, 2, 3, 4]) * (top_k * stake + 2 * stake)
    roi = (profit / total_stake * 100) if total_stake else 0.0
    return LogicPerformance(name=name, roi=roi, profit=profit, hits=hits)


def adjust_stake(profit_history: float, base: float = BASE_STAKE) -> float:
    if profit_history < -100:
        return base * 1.25
    if profit_history < 0:
        return base * 1.1
    if profit_history > 200:
        return base * 0.9
    return base


# -------------------------- MAIN DECISION PIPELINE -------------------------
def combine_predictions(pred_maps: Dict[str, Dict[int, List[int]]], roi_table: Dict[str, float], top_k: int = 5) -> Dict[int, List[int]]:
    ordered_names = list(pred_maps.keys())
    roi_weights = {name: max(0.1, roi_table.get(name, 0.0) + 1.0) for name in ordered_names}
    preds_list = [pred_maps[name] for name in ordered_names]
    return scr9_predict(preds_list, roi_weights, top_k)


def generate_all_logic_predictions(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[str, Dict[int, List[int]]]:
    p1 = scr1_predict(df, target_date, top_k)
    p2 = scr2_predict(df, target_date, top_k)
    p3 = scr3_predict(df, target_date, top_k)
    p45 = scr45_predict(df, target_date, top_k)
    p6 = scr6_predict(df, target_date, top_k)
    p8 = scr8_predict(target_date, top_k)
    p7 = scr7_predict([p1, p2, p3, p45, p6, p8], top_k)
    return {
        "SCR1": p1,
        "SCR2": p2,
        "SCR3": p3,
        "SCR4": p45,
        "SCR5": p45,
        "SCR6": p6,
        "SCR7": p7,
        "SCR8": p8,
    }


def prediction_summary(pred: Dict[int, List[int]]) -> Dict[str, List[str]]:
    return {SLOT_NAMES[slot - 1]: [two_digit(n) for n in nums] for slot, nums in pred.items()}


def main():
    print("=== MERGED MASTER PREDICTOR (SCR1–SCR9) ===")
    excel_path = "number prediction learn.xlsx"
    df_long = load_long_dataframe(excel_path)
    if df_long.empty:
        print("❌ No data available. Please update the Excel file.")
        return

    print(f"📊 Loaded {len(df_long)} records from {df_long['date'].min().date()} to {df_long['date'].max().date()}")
    latest_date, filled_slots = detect_intraday(df_long)
    all_slots = [1, 2, 3, 4]
    missing_slots = [s for s in all_slots if s not in filled_slots]
    intraday = len(filled_slots) < len(all_slots)
    target_dates = []
    if intraday:
        target_dates.append(latest_date)
    target_dates.append(latest_date + timedelta(days=1))

    # Backtest to score logics
    perf_table: Dict[str, LogicPerformance] = {}
    for name, func in [
        ("SCR1", scr1_predict),
        ("SCR2", scr2_predict),
        ("SCR3", scr3_predict),
        ("SCR4", scr45_predict),
        ("SCR5", scr45_predict),
        ("SCR6", scr6_predict),
        ("SCR8", lambda d, td, tk: scr8_predict(td, tk)),
    ]:
        perf_table[name] = backtest_logic(df_long, func, name)
    roi_table = {name: perf.roi for name, perf in perf_table.items()}

    cumulative_profit = sum(perf.profit for perf in perf_table.values())
    stake = adjust_stake(cumulative_profit, BASE_STAKE)

    for tgt_date in target_dates:
        scope_df = df_long[df_long["date"] < tgt_date]
        if scope_df.empty:
            continue
        logic_preds = generate_all_logic_predictions(scope_df, tgt_date, top_k=5)
        combined = combine_predictions(logic_preds, roi_table, top_k=5)
        summary = prediction_summary(combined)
        andar, bahar = andar_bahar_from_predictions([n for nums in combined.values() for n in nums])
        total_stake = stake * sum(len(v) for v in combined.values()) + 2 * stake * len(combined)
        header = "INTRADAY" if tgt_date == latest_date else "NEXT DAY"
        print(f"\n🗓️ Prediction date: {tgt_date.date()} ({header})")
        for slot_name, nums in summary.items():
            if intraday and tgt_date == latest_date:
                slot_idx = SLOT_MAP[slot_name]
                if slot_idx not in missing_slots:
                    continue
            print(f"  {slot_name}: {', '.join(nums)}")
        print(f"  ANDAR digit: {andar} | BAHAR digit: {bahar}")
        print(f"  Stake per pick: ₹{stake:.2f} | Estimated total stake: ₹{total_stake:.2f}")
        top_logic = max(perf_table.values(), key=lambda p: p.roi)
        print(f"  Leading logic: {top_logic.name} ({top_logic.roi:.2f}% ROI last window)")

    # Optional: show detected pattern strength
    strong_s40 = [n for n in combined.values() if any(two_digit(x) in S40 for x in n)]
    if strong_s40:
        print("\n✨ S40 alignment detected in combined picks – treat as bonus confidence layer.")


if __name__ == "__main__":
    main()
