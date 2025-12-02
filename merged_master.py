"""Merged master predictor combining SCR1–SCR9 behaviours.

Upgrades in this version:
- CLI modes: NEW (default), OLD, and COMPARE for side-by-side trust-building.
- Output refresh with A/B/C stake tiers, snapshots, and pattern/risk panels.
- Dynamic pick counts (3–5) driven by consensus/score shape instead of a
  hard-coded top_k=5, plus an old-style formatter for legacy clarity.

Logic inspiration borrowed from historical modules such as
``deepseek_scr9.py``, ``pattern_packs.py``, ``high_conviction_filter.py``, and
``bet_pnl_tracker.py`` while compressing behaviour into this single file. Some
assumptions are made around money-management caps and risk labels to keep the
implementation lightweight but still faithful to the previous ecosystem.
"""
from __future__ import annotations
import argparse
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

SHOW_COMPARISON = True
BET_PLAN_FOLDER = os.path.join(os.getcwd(), "bet_plans")

BASE_STAKE = 10.0
RETURN_EXACT = 90
RETURN_DIGIT = 9
TIER_MULTIPLIERS = {"A": 2.0, "B": 1.0, "C": 0.5}
ANDAR_BAHAR_MULTIPLIER = 1.0


# ------------------------------ DATA UTILITIES ------------------------------
def two_digit(n: int) -> str:
    return f"{int(n) % 100:02d}"


def load_long_dataframe(path: str = "number prediction learn.xlsx") -> pd.DataFrame:
    df = load_results_excel(path)
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["slot"] = df["slot"].astype(int)
    # Keep XX (closed day) rows out of the modelling dataframe
    df = df[pd.notna(df["number"])]
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


def decide_top_k_for_slot(scores: Counter, base_top_k: int = 5) -> int:
    """Decide between 3–5 picks based on score spread/variance.

    A sharp drop after the top few scores or high variance means we
    confidently trim to 3; a modest edge trims to 4; otherwise stay at 5.
    """

    if not scores:
        return base_top_k
    values = sorted(scores.values(), reverse=True)
    if len(values) <= 3:
        return max(1, len(values))

    pivot = values[min(4, len(values) - 1)]
    top = values[0]
    ratio = top / max(values[min(2, len(values) - 1)], 1e-6)
    std = float(np.std(values[: min(5, len(values))]))

    if (top - pivot) > (0.35 * top) or ratio > 1.8:
        return 3
    if ratio > 1.3 or std > 0.15 * top:
        return 4
    return base_top_k


def assign_tiers(scores: Dict[int, float], picks: List[int], u_live: float) -> List[Dict[str, float]]:
    ranks = sorted(picks, key=lambda n: scores.get(n, 0), reverse=True)
    tiered = []
    for idx, num in enumerate(ranks):
        if idx < 2:
            tier = "A"
        elif idx < 4:
            tier = "B"
        else:
            tier = "C"
        stake_val = TIER_MULTIPLIERS[tier] * u_live
        tiered.append({"num": num, "tier": tier, "stake": stake_val})
    return tiered


def format_slot_line_old_style(
    slot_name: str,
    tiered: List[Dict[str, float]],
    andar: int,
    bahar: int,
    andar_stake: float,
    bahar_stake: float,
) -> Tuple[str, float]:
    parts = [f"{two_digit(item['num'])}({item['tier']} ₹{item['stake']:.0f})" for item in tiered]
    numbers_block = ", ".join(parts)
    slot_stake = sum(item["stake"] for item in tiered) + andar_stake + bahar_stake
    line = (
        f"{slot_name}: {numbers_block} | ANDAR={andar} (₹{andar_stake:.0f}), "
        f"BAHAR={bahar} (₹{bahar_stake:.0f}) → Slot stake: ₹{slot_stake:.0f}"
    )
    return line, slot_stake


def compute_slot_confidence(scores: Dict[int, float]) -> float:
    if not scores:
        return 0.0
    values = sorted(scores.values(), reverse=True)
    top3 = sum(values[:3])
    total = sum(values)
    ratio = top3 / total if total else 0.0
    return round(min(100.0, ratio * 120), 1)


def digit_features(num: int) -> Dict[str, int]:
    tens, ones = num // 10, num % 10
    digit_sum = tens + ones
    is_prime = int(num in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97})
    return {
        "tens": tens,
        "ones": ones,
        "parity": (num % 2),
        "digit_sum": digit_sum,
        "sum_parity": digit_sum % 2,
        "mod9": num % 9,
        "is_prime": is_prime,
    }


def recent_hit_features(past_df: pd.DataFrame, date: pd.Timestamp, slot: int, num: int) -> Dict[str, float]:
    if past_df.empty:
        return {k: 0.0 for k in ["days_since_any", "days_since_slot", "hit7", "hit30", "hit90", "hit7_slot", "hit30_slot", "hit90_slot", "neighbor_recent", "mirror_recent"]}
    look7 = date - timedelta(days=7)
    look30 = date - timedelta(days=30)
    look90 = date - timedelta(days=90)
    subset7 = past_df[past_df["date"] >= look7]
    subset30 = past_df[past_df["date"] >= look30]
    subset90 = past_df[past_df["date"] >= look90]

    hit7 = int(num in subset7["number"].values)
    hit30 = int(num in subset30["number"].values)
    hit90 = int(num in subset90["number"].values)

    slot7 = subset7[subset7["slot"] == slot]
    slot30 = subset30[subset30["slot"] == slot]
    slot90 = subset90[subset90["slot"] == slot]
    hit7_slot = int(num in slot7["number"].values)
    hit30_slot = int(num in slot30["number"].values)
    hit90_slot = int(num in slot90["number"].values)

    last_any = past_df[past_df["number"] == num]["date"].max()
    last_slot = past_df[(past_df["number"] == num) & (past_df["slot"] == slot)]["date"].max()
    days_since_any = (date - last_any).days if pd.notna(last_any) else 120
    days_since_slot = (date - last_slot).days if pd.notna(last_slot) else 120

    neighbor_recent = 0.0
    mirror_recent = 0.0
    neighbors = {(num + 1) % 100, (num - 1) % 100}
    mirror = int(f"{num%10}{num//10}")
    if not subset30.empty:
        neighbor_recent = int(any(n in subset30["number"].values for n in neighbors))
        mirror_recent = int(mirror in subset30["number"].values)

    return {
        "days_since_any": float(days_since_any),
        "days_since_slot": float(days_since_slot),
        "hit7": float(hit7),
        "hit30": float(hit30),
        "hit90": float(hit90),
        "hit7_slot": float(hit7_slot),
        "hit30_slot": float(hit30_slot),
        "hit90_slot": float(hit90_slot),
        "neighbor_recent": float(neighbor_recent),
        "mirror_recent": float(mirror_recent),
    }


def build_ml_features(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    dates = sorted(df_long["date"].unique())
    rows: List[Dict[str, float]] = []
    for date in dates:
        past_df = df_long[df_long["date"] < date]
        day_rows = df_long[df_long["date"] == date]
        for slot in [1, 2, 3, 4]:
            actual_row = day_rows[day_rows["slot"] == slot]
            if actual_row.empty:
                continue
            actual_num = int(actual_row.iloc[0]["number"])
            for num in range(100):
                feat = {"date": date, "slot": slot, "candidate": num}
                feat.update(digit_features(num))
                feat.update(recent_hit_features(past_df, date, slot, num))
                feat["s40"] = float(two_digit(num) in S40)
                feat["f164950"] = float(two_digit(num) in PACK_164950_FAMILY)
                feat["target"] = 1 if num == actual_num else 0
                rows.append(feat)
    return pd.DataFrame(rows)


def generate_feature_row(past_df: pd.DataFrame, date: pd.Timestamp, slot: int, num: int) -> Dict[str, float]:
    feat = {"slot": slot, "candidate": num}
    feat.update(digit_features(num))
    feat.update(recent_hit_features(past_df, date, slot, num))
    feat["s40"] = float(two_digit(num) in S40)
    feat["f164950"] = float(two_digit(num) in PACK_164950_FAMILY)
    return feat


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


# ----------------------- LOGIC: SCR6 (ML PREDICTOR) -----------------------
_SCR6_MODEL_CACHE: Dict[str, GradientBoostingClassifier] = {}


def _train_scr6_model(df: pd.DataFrame) -> Optional[GradientBoostingClassifier]:
    feats = build_ml_features(df)
    if feats.empty:
        return None
    X = feats.drop(columns=["target", "date"])
    y = feats["target"]
    if y.sum() < 5:
        return None
    model = GradientBoostingClassifier(random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    return model


def scr6_predict(df: pd.DataFrame, target_date: pd.Timestamp, top_k: int = 5) -> Dict[int, List[int]]:
    hist = df[df["date"] < target_date]
    cache_key = f"{len(hist)}-{hist['date'].max()}"
    model = _SCR6_MODEL_CACHE.get(cache_key)
    if model is None:
        model = _train_scr6_model(hist)
        if model is not None:
            _SCR6_MODEL_CACHE[cache_key] = model

    out: Dict[int, List[int]] = {}
    if model is None or hist.empty:
        for slot in [1, 2, 3, 4]:
            nums = hist[hist["slot"] == slot]["number"].tolist()
            freq = Counter(nums[-50:])
            picks = [n for n, _ in freq.most_common(top_k)]
            out[slot] = picks
        return out

    past_df = hist
    for slot in [1, 2, 3, 4]:
        scores: Dict[int, float] = {}
        for num in range(100):
            feat = generate_feature_row(past_df, target_date, slot, num)
            X_row = pd.DataFrame([feat])
            proba = float(model.predict_proba(X_row)[0][1])
            bonus = 0.0
            if two_digit(num) in S40:
                bonus += 0.02
            if two_digit(num) in PACK_164950_FAMILY:
                bonus += 0.01
            scores[num] = proba + bonus
        picks = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_k]]
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


@dataclass
class CombinedPrediction:
    picks: Dict[int, List[int]]
    scores: Dict[int, Dict[int, float]]
    top_k_by_slot: Dict[int, int]


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


def compute_logic_performance_table(df_long: pd.DataFrame) -> Tuple[Dict[str, LogicPerformance], Dict[str, float], float]:
    perf_table: Dict[str, LogicPerformance] = {}
    health_table: Dict[str, Dict[str, float]] = {}
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
        health_table[name] = compute_layer_health(df_long, func)
    roi_table = {name: perf.roi for name, perf in perf_table.items()}
    cumulative_profit = sum(perf.profit for perf in perf_table.values())
    return perf_table, roi_table, cumulative_profit, health_table


def compute_layer_health(df_long: pd.DataFrame, predictor, window: int = 120) -> Dict[str, float]:
    dates = sorted(df_long["date"].unique())
    if len(dates) < 5:
        return {"roi7": 0.0, "roi30": 0.0, "roi90": 0.0, "sigma90": 0.0, "drawdown": 0.0}
    start_idx = max(1, len(dates) - window)
    profit_by_date: Dict[pd.Timestamp, float] = defaultdict(float)
    stake_by_date: Dict[pd.Timestamp, float] = defaultdict(float)
    stake = BASE_STAKE
    for i in range(start_idx, len(dates)):
        hist = df_long[df_long["date"] < dates[i]]
        if hist.empty:
            continue
        preds = predictor(hist, dates[i], top_k=5)
        day_rows = df_long[df_long["date"] == dates[i]]
        for _, row in day_rows.iterrows():
            slot = int(row["slot"])
            actual_num = int(row["number"])
            tens, ones = actual_num // 10, actual_num % 10
            picks = preds.get(slot, [])
            profit_delta = evaluate_day_predictions({"number": actual_num, "tens": tens, "ones": ones}, picks, *andar_bahar_from_predictions(picks), stake)
            profit_by_date[dates[i]] += profit_delta
            stake_by_date[dates[i]] += stake * len(picks) + 2 * stake
    def roi_window(days: int) -> float:
        start = dates[-1] - timedelta(days=days - 1)
        p = sum(v for d, v in profit_by_date.items() if d >= start)
        s = sum(v for d, v in stake_by_date.items() if d >= start)
        return (p / s * 100) if s else 0.0
    profits_sorted = [profit_by_date[d] for d in sorted(profit_by_date.keys())]
    sigma90 = float(np.std(profits_sorted[-90:])) if profits_sorted else 0.0
    running = 0.0
    peak = 0.0
    drawdown = 0.0
    for p in profits_sorted:
        running += p
        peak = max(peak, running)
        drawdown = min(drawdown, running - peak)
    return {"roi7": roi_window(7), "roi30": roi_window(30), "roi90": roi_window(90), "sigma90": abs(sigma90), "drawdown": abs(drawdown)}


def adjust_stake(profit_history: float, base: float = BASE_STAKE) -> float:
    if profit_history < -150:
        return base * 1.4
    if profit_history < -50:
        return base * 1.2
    if profit_history > 300:
        return base * 0.85
    return base


def compute_roi_weights(roi_table: Dict[str, float], health_table: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    weights = {}
    for name, roi in roi_table.items():
        gate = 1.0
        if health_table and name in health_table:
            stats = health_table[name]
            if stats["roi30"] < -10 and stats["drawdown"] > 1.5 * stats.get("sigma90", 0):
                gate = 0.0
            elif stats["roi7"] < -5:
                gate = 0.5
        weights[name] = max(0.0, (roi + 1.0) * gate)
    return weights


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


def combine_predictions(
    pred_maps: Dict[str, Dict[int, List[int]]],
    roi_table: Dict[str, float],
    health_table: Optional[Dict[str, Dict[str, float]]] = None,
    risk_mode: str = "BALANCED",
    base_top_k: int = 5,
    dynamic_topk: bool = True,
) -> CombinedPrediction:
    roi_weights = compute_roi_weights(roi_table, health_table)
    slot_scores: Dict[int, Counter] = {slot: Counter() for slot in [1, 2, 3, 4]}
    for name, preds in pred_maps.items():
        weight = roi_weights.get(name, 1.0)
        for slot, nums in preds.items():
            for rank, num in enumerate(nums):
                slot_scores[slot][num] += weight / (rank + 1)
    picks: Dict[int, List[int]] = {}
    top_k_by_slot: Dict[int, int] = {}
    for slot, counter in slot_scores.items():
        k = decide_top_k_for_slot(counter, base_top_k) if dynamic_topk else base_top_k
        top_k_by_slot[slot] = max(3, min(base_top_k, k))
        ranked = counter.most_common(20)
        if ranked:
            top_score = ranked[0][1]
            ev_gate = 0.97 * top_score
            filtered = [(n, s) for n, s in ranked if s >= ev_gate]
            cap = 12 if risk_mode != "DEFENSIVE" else 8
            if risk_mode == "AGGRESSIVE":
                cap = 15
            picks_list = [n for n, _ in filtered][:cap]
        else:
            picks_list = []
        picks[slot] = picks_list[: top_k_by_slot[slot]]
    return CombinedPrediction(
        picks=picks,
        scores={slot: dict(counter) for slot, counter in slot_scores.items()},
        top_k_by_slot=top_k_by_slot,
    )


def prediction_summary(pred: Dict[int, List[int]]) -> Dict[str, List[str]]:
    return {SLOT_NAMES[slot - 1]: [two_digit(n) for n in nums] for slot, nums in pred.items()}


def run_combined_backtest(
    df_long: pd.DataFrame, roi_table: Dict[str, float], dynamic_topk: bool = True, window: int = 60
) -> Dict[str, object]:
    dates = sorted(df_long["date"].unique())
    if len(dates) < 2:
        return {
            "profit": 0.0,
            "roi": 0.0,
            "profit_by_slot": Counter(),
            "profit_by_date": defaultdict(float),
            "stake_by_date": defaultdict(float),
            "stake_total": 0.0,
        }
    start_idx = max(1, len(dates) - window)
    profit = 0.0
    stake_total = 0.0
    profit_by_slot: Counter = Counter()
    profit_by_date: defaultdict = defaultdict(float)
    stake_by_date: defaultdict = defaultdict(float)

    for i in range(start_idx, len(dates)):
        target_date = dates[i]
        hist = df_long[df_long["date"] < target_date]
        if hist.empty:
            continue
        logic_preds = generate_all_logic_predictions(hist, target_date, top_k=5)
        combined = combine_predictions(logic_preds, roi_table, risk_mode="BALANCED", dynamic_topk=dynamic_topk)
        flat_preds = [n for nums in combined.picks.values() for n in nums]
        andar, bahar = andar_bahar_from_predictions(flat_preds)
        day_stake = 0.0
        for _, row in df_long[df_long["date"] == target_date].iterrows():
            slot = int(row["slot"])
            actual_num = int(row["number"])
            tens, ones = actual_num // 10, actual_num % 10
            picks = combined.picks.get(slot, [])
            stake = BASE_STAKE
            stake_for_row = stake * len(picks) + 2 * stake
            day_stake += stake_for_row
            profit_delta = evaluate_day_predictions(
                {"number": actual_num, "tens": tens, "ones": ones},
                picks,
                andar,
                bahar,
                stake,
            )
            profit += profit_delta
            profit_by_slot[slot] += profit_delta
            profit_by_date[target_date] += profit_delta
            stake_by_date[target_date] += stake_for_row
        stake_total += day_stake
    roi = (profit / stake_total * 100) if stake_total else 0.0
    return {
        "profit": profit,
        "roi": roi,
        "profit_by_slot": profit_by_slot,
        "profit_by_date": profit_by_date,
        "stake_by_date": stake_by_date,
        "stake_total": stake_total,
    }


def compute_window(profit_by_date: Dict[pd.Timestamp, float], stake_by_date: Dict[pd.Timestamp, float], days: int) -> Tuple[float, float]:
    if not profit_by_date:
        return 0.0, 0.0
    max_date = max(profit_by_date.keys())
    start = max_date - timedelta(days=days - 1)
    profit = sum(v for d, v in profit_by_date.items() if d >= start)
    stake = sum(v for d, v in stake_by_date.items() if d >= start)
    roi = (profit / stake * 100) if stake else 0.0
    return profit, roi


def compute_pnl_snapshot(df_long: pd.DataFrame, roi_table: Dict[str, float], dynamic_topk: bool) -> Dict[str, object]:
    sim = run_combined_backtest(df_long, roi_table, dynamic_topk=dynamic_topk)
    best_slot_idx = max(sim["profit_by_slot"], key=lambda k: sim["profit_by_slot"][k], default=1)
    weak_slot_idx = min(sim["profit_by_slot"], key=lambda k: sim["profit_by_slot"][k], default=1)
    last7_profit, last7_roi = compute_window(sim["profit_by_date"], sim["stake_by_date"], 7)
    last30_profit, last30_roi = compute_window(sim["profit_by_date"], sim["stake_by_date"], 30)
    return {
        "overall_profit": sim["profit"],
        "overall_roi": sim["roi"],
        "last7": {"profit": last7_profit, "roi": last7_roi},
        "last30": {"profit": last30_profit, "roi": last30_roi},
        "best_slot": SLOT_NAMES[best_slot_idx - 1],
        "weak_slot": SLOT_NAMES[weak_slot_idx - 1],
        "profit_by_date": sim["profit_by_date"],
        "stake_by_date": sim["stake_by_date"],
    }


def compute_pattern_stats(df_long: pd.DataFrame) -> Dict[str, float]:
    if df_long.empty:
        return {
            "hits": 0,
            "s40_hits": 0,
            "pack_hits": 0,
            "s40_hit_rate": 0.0,
            "pack_hit_rate": 0.0,
            "near_miss": 0,
            "hot_neighbors": [],
            "golden_pack": None,
        }
    hits = len(df_long)
    s40_hits = sum(1 for n in df_long["number"] if two_digit(n) in S40)
    pack_hits = sum(1 for n in df_long["number"] if two_digit(n) in PACK_164950_FAMILY)
    near_miss = 0
    hot_neighbors: Counter = Counter()
    recent_df = df_long[df_long["date"] >= df_long["date"].max() - timedelta(days=90)]
    prev_numbers = {}
    for _, row in recent_df.iterrows():
        num = int(row["number"])
        slot = int(row["slot"])
        prev_numbers.setdefault(row["date"], {})[slot] = num
    sorted_dates = sorted(prev_numbers.keys())
    for i in range(1, len(sorted_dates)):
        today = prev_numbers[sorted_dates[i]]
        yesterday = prev_numbers[sorted_dates[i - 1]]
        for slot, num in today.items():
            for cand in yesterday.values():
                if num in {(cand + 1) % 100, (cand - 1) % 100, int(f"{cand%10}{cand//10}")}:
                    near_miss += 1
                    hot_neighbors[num] += 1
    hot_list = [two_digit(n) for n, _ in hot_neighbors.most_common(5)]

    last30 = df_long[df_long["date"] >= df_long["date"].max() - timedelta(days=30)]
    golden_pack = None
    if not last30.empty:
        counts = Counter(last30["number"])
        pack = [n for n, c in counts.most_common(6) if c > 1][:3]
        if pack:
            hit_rate = sum(counts[n] for n in pack) / max(1, len(last30)) * 100
            if hit_rate > 20:
                golden_pack = {two_digit(n) for n in pack}
    return {
        "hits": hits,
        "s40_hits": s40_hits,
        "pack_hits": pack_hits,
        "s40_hit_rate": (s40_hits / hits * 100) if hits else 0.0,
        "pack_hit_rate": (pack_hits / hits * 100) if hits else 0.0,
        "near_miss": near_miss,
        "hot_neighbors": hot_list,
        "golden_pack": golden_pack,
    }


def compute_money_manager(pnl_snapshot: Dict[str, object], confidence: Dict[str, float], base_unit: float = BASE_STAKE) -> Dict[str, object]:
    roi30 = pnl_snapshot.get("last30", {}).get("roi", 0.0)
    roi7 = pnl_snapshot.get("last7", {}).get("roi", 0.0)
    volatility = float(np.std(list(pnl_snapshot.get("profit_by_date", {}).values())[-30:])) if pnl_snapshot.get("profit_by_date") else 0.0
    streak_factor = 1.0
    profits = list(pnl_snapshot.get("profit_by_date", {}).values())
    if profits:
        streak = deque(profits[-5:], maxlen=5)
        if all(p <= 0 for p in streak):
            streak_factor = 1.3
        elif all(p >= 0 for p in streak):
            streak_factor = 0.9

    risk_mode = "BALANCED"
    if roi30 > 15 and roi7 > 5:
        risk_mode = "AGGRESSIVE"
    elif roi30 < -10:
        risk_mode = "DEFENSIVE"

    u_live = base_unit * streak_factor
    if risk_mode == "DEFENSIVE":
        u_live *= 0.9
    elif risk_mode == "AGGRESSIVE":
        u_live *= 1.1

    execution = "GO_LIVE_FULL" if risk_mode == "AGGRESSIVE" and volatility < abs(pnl_snapshot.get("overall_profit", 0.0)) / 20 else "GO_LIVE_LIGHT"
    daily_cap = u_live * (60 if risk_mode == "DEFENSIVE" else 90)
    slot_cap = daily_cap / 4
    return {
        "risk_mode": risk_mode,
        "execution": execution,
        "u_live": u_live,
        "daily_cap": daily_cap,
        "slot_cap": slot_cap,
        "confidence": confidence,
    }


def compute_risk_execution_summary(
    pnl_snapshot: Dict[str, object],
    pattern_stats: Dict[str, float],
    confidence: Dict[str, float],
    u_live: float,
) -> Dict[str, object]:
    money_state = compute_money_manager(pnl_snapshot, confidence, base_unit=u_live)
    roi = pnl_snapshot.get("overall_roi", 0.0)
    risk_mode = money_state["risk_mode"]

    strategy = "STRAT_S40_BOOST" if pattern_stats.get("s40_hit_rate", 0.0) > 32 else "BALANCED_ROI"
    exec_mode = money_state["execution"]
    money_mgr = {
        "daily_cap": money_state["daily_cap"],
        "slot_cap": money_state["slot_cap"],
    }
    return {
        "strategy": strategy,
        "risk_mode": risk_mode,
        "execution": exec_mode,
        "money_manager": money_mgr,
        "confidence": confidence,
    }


def build_mode_prediction(
    scope_df: pd.DataFrame,
    tgt_date: pd.Timestamp,
    roi_table: Dict[str, float],
    health_table: Dict[str, Dict[str, float]],
    u_live: float,
    intraday: bool,
    missing_slots: List[int],
    latest_date: pd.Timestamp,
    dynamic_topk: bool,
) -> Dict[str, object]:
    logic_preds = generate_all_logic_predictions(scope_df, tgt_date, top_k=5)
    risk_mode = "AGGRESSIVE" if roi_table and max(roi_table.values()) > 15 else "BALANCED"
    combined = combine_predictions(logic_preds, roi_table, health_table=health_table, risk_mode=risk_mode, dynamic_topk=dynamic_topk)
    flat_preds = [n for nums in combined.picks.values() for n in nums]
    andar, bahar = andar_bahar_from_predictions(flat_preds)
    andar_stake = ANDAR_BAHAR_MULTIPLIER * u_live
    bahar_stake = ANDAR_BAHAR_MULTIPLIER * u_live

    lines: List[str] = []
    slot_stakes: Dict[str, float] = {}
    confidence: Dict[str, float] = {}
    picks_by_slot: Dict[str, List[int]] = {}
    for slot_idx in [1, 2, 3, 4]:
        if intraday and tgt_date == latest_date and slot_idx not in missing_slots:
            continue
        slot_name = SLOT_NAMES[slot_idx - 1]
        picks = combined.picks.get(slot_idx, [])
        slot_scores = combined.scores.get(slot_idx, {})
        tiered = assign_tiers(slot_scores, picks, u_live)
        line, slot_stake = format_slot_line_old_style(slot_name, tiered, andar, bahar, andar_stake, bahar_stake)
        slot_cap = u_live * 25
        if slot_stake > slot_cap:
            scale = slot_cap / slot_stake
            for item in tiered:
                item["stake"] = max(1.0, item["stake"] * scale)
            andar_stake_scaled = max(1.0, andar_stake * scale)
            bahar_stake_scaled = max(1.0, bahar_stake * scale)
            line, slot_stake = format_slot_line_old_style(slot_name, tiered, andar, bahar, andar_stake_scaled, bahar_stake_scaled)
        lines.append(line)
        slot_stakes[slot_name] = slot_stake
        confidence[slot_name] = compute_slot_confidence(slot_scores)
        picks_by_slot[slot_name] = picks

    total_stake = sum(slot_stakes.values())
    return {
        "lines": lines,
        "andar": andar,
        "bahar": bahar,
        "slot_stakes": slot_stakes,
        "total_stake": total_stake,
        "confidence": confidence,
        "picks_by_slot": picks_by_slot,
        "combined": combined,
    }


def main():
    parser = argparse.ArgumentParser(description="Merged master predictor")
    parser.add_argument("--mode", choices=["NEW", "OLD", "COMPARE"], default="NEW", help="Prediction mode")
    args = parser.parse_args()

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

    perf_table, roi_table, cumulative_profit, health_table = compute_logic_performance_table(df_long)
    u_live = adjust_stake(cumulative_profit, BASE_STAKE)
    pattern_stats = compute_pattern_stats(df_long)
    pnl_snapshot_new = compute_pnl_snapshot(df_long, roi_table, dynamic_topk=True)
    pnl_snapshot_old = compute_pnl_snapshot(df_long, roi_table, dynamic_topk=False)
    top_logic = max(perf_table.values(), key=lambda p: p.roi)

    last_pred_block: Optional[Dict[str, object]] = None

    for tgt_date in target_dates:
        scope_df = df_long[df_long["date"] < tgt_date]
        if scope_df.empty:
            continue
        header = "INTRADAY" if tgt_date == latest_date else "NEXT DAY"
        print(f"\n🗓️ Prediction date: {tgt_date.date()} ({header})")

        if args.mode in {"NEW", "COMPARE"}:
            new_pred = build_mode_prediction(
                scope_df,
                tgt_date,
                roi_table,
                health_table,
                u_live,
                intraday,
                missing_slots,
                latest_date,
                dynamic_topk=True,
            )
            confidence_map = new_pred["confidence"]
            risk_summary = compute_risk_execution_summary(pnl_snapshot_new, pattern_stats, confidence_map, u_live)
            last_pred_block = new_pred

        if args.mode == "NEW":
            print("1️⃣ FINAL PREDICTIONS (NEW ENGINE)")
            for line in new_pred["lines"]:
                print(f"   {line}")
            print(f"   Total stake (incl. digits): ₹{new_pred['total_stake']:.0f}")
            print(f"   ANDAR={new_pred['andar']} | BAHAR={new_pred['bahar']}")
            print(f"   Leading logic: {top_logic.name} ({top_logic.roi:.2f}% ROI last window)")

            print("\n2️⃣ P&L SNAPSHOT (BACKTESTED)")
            print(
                f"   Overall P&L      : ₹{pnl_snapshot_new['overall_profit']:.0f} "
                f"(ROI {pnl_snapshot_new['overall_roi']:.2f}%)"
            )
            print(
                f"   Last 7 days      : ₹{pnl_snapshot_new['last7']['profit']:.0f} "
                f"(ROI {pnl_snapshot_new['last7']['roi']:.2f}%)"
            )
            print(
                f"   Last 30 days     : ₹{pnl_snapshot_new['last30']['profit']:.0f} "
                f"(ROI {pnl_snapshot_new['last30']['roi']:.2f}%)"
            )
            print(f"   Best slot        : {pnl_snapshot_new['best_slot']}")
            print(f"   Weak slot        : {pnl_snapshot_new['weak_slot']}")

            print("\n3️⃣ PATTERN & LEARNING")
            print(f"   Hits analyzed    : {pattern_stats['hits']}")
            print(
                f"   S40 family       : {pattern_stats['s40_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['s40_hits']} hits"
            )
            print(
                f"   164950 family    : {pattern_stats['pack_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['pack_hits']} hits"
            )
            print(
                f"   Near-miss count  : {pattern_stats['near_miss']} | Hot neighbors: {', '.join(pattern_stats['hot_neighbors'])}"
            )
            if pattern_stats.get("golden_pack"):
                print(f"   Golden pack      : {sorted(pattern_stats['golden_pack'])}")
            print(
                f"   Near-miss count  : {pattern_stats['near_miss']} | Hot neighbors: {', '.join(pattern_stats['hot_neighbors'])}"
            )
            if pattern_stats.get("golden_pack"):
                print(f"   Golden pack      : {sorted(pattern_stats['golden_pack'])}")

            print("\n4️⃣ RISK & EXECUTION")
            print(f"   Strategy         : {risk_summary['strategy']}")
            print(f"   Risk mode        : {risk_summary['risk_mode']}")
            print(f"   Execution mode   : {risk_summary['execution']}")
            print(
                f"   Money manager    : daily cap ₹{risk_summary['money_manager']['daily_cap']:.0f}, "
                f"single-slot cap ₹{risk_summary['money_manager']['slot_cap']:.0f}"
            )
            conf_parts = ", ".join([f"{k} {v:.1f}%" for k, v in confidence_map.items()])
            print(f"   Confidence       : {conf_parts}")

        if args.mode == "OLD":
            old_pred = build_mode_prediction(
                scope_df,
                tgt_date,
                roi_table,
                health_table,
                u_live,
                intraday,
                missing_slots,
                latest_date,
                dynamic_topk=False,
            )
            print("1️⃣ FINAL PREDICTIONS (OLD STYLE)")
            for line in old_pred["lines"]:
                print(f"   {line}")
            print(f"   Total stake (incl. digits): ₹{old_pred['total_stake']:.0f}")
            print(f"   ANDAR={old_pred['andar']} | BAHAR={old_pred['bahar']}")
            print(f"   Leading logic: {top_logic.name} ({top_logic.roi:.2f}% ROI last window)")

            print("\n2️⃣ P&L SNAPSHOT")
            print(
                f"   Overall P&L      : ₹{pnl_snapshot_old['overall_profit']:.0f} "
                f"(ROI {pnl_snapshot_old['overall_roi']:.2f}%)"
            )
            print(
                f"   Last 7 days      : ₹{pnl_snapshot_old['last7']['profit']:.0f} "
                f"(ROI {pnl_snapshot_old['last7']['roi']:.2f}%)"
            )
            print(
                f"   Last 30 days     : ₹{pnl_snapshot_old['last30']['profit']:.0f} "
                f"(ROI {pnl_snapshot_old['last30']['roi']:.2f}%)"
            )
            print(f"   Best slot        : {pnl_snapshot_old['best_slot']}")
            print(f"   Weak slot        : {pnl_snapshot_old['weak_slot']}")

            print("\n3️⃣ PATTERN & LEARNING")
            print(f"   Hits analyzed    : {pattern_stats['hits']}")
            print(
                f"   S40 family       : {pattern_stats['s40_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['s40_hits']} hits"
            )
            print(
                f"   164950 family    : {pattern_stats['pack_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['pack_hits']} hits"
            )

            old_confidence = {k: v for k, v in old_pred["confidence"].items()}
            old_risk = compute_risk_execution_summary(pnl_snapshot_old, pattern_stats, old_confidence, u_live)
            print("\n4️⃣ RISK & EXECUTION")
            print(f"   Strategy         : {old_risk['strategy']}")
            print(f"   Risk mode        : {old_risk['risk_mode']}")
            print(f"   Execution mode   : {old_risk['execution']}")
            print(
                f"   Money manager    : daily cap ₹{old_risk['money_manager']['daily_cap']:.0f}, "
                f"single-slot cap ₹{old_risk['money_manager']['slot_cap']:.0f}"
            )
            conf_parts_old = ", ".join([f"{k} {v:.1f}%" for k, v in old_confidence.items()])
            print(f"   Confidence       : {conf_parts_old}")
            last_pred_block = old_pred

        if args.mode == "COMPARE":
            old_pred = build_mode_prediction(
                scope_df,
                tgt_date,
                roi_table,
                health_table,
                u_live,
                intraday,
                missing_slots,
                latest_date,
                dynamic_topk=False,
            )
            print("=== OLD vs NEW (slot-wise) ===")
            for slot_name in SLOT_NAMES:
                if intraday and tgt_date == latest_date and SLOT_MAP[slot_name] not in missing_slots:
                    continue
                new_line = next((ln for ln in new_pred["lines"] if ln.startswith(slot_name)), f"{slot_name}: -")
                old_line = next((ln for ln in old_pred["lines"] if ln.startswith(slot_name)), f"{slot_name}: -")
                print("\n==================================================")
                print(f"SLOT: {slot_name} – OLD vs NEW")
                print("--------------------------------------------------")
                print(f"OLD: {old_line.split(': ',1)[1] if ': ' in old_line else old_line}")
                print(f"NEW: {new_line.split(': ',1)[1] if ': ' in new_line else new_line}")

                old_set = set(old_pred["picks_by_slot"].get(slot_name, []))
                new_set = set(new_pred["picks_by_slot"].get(slot_name, []))
                overlap = {two_digit(n) for n in old_set & new_set}
                old_only = {two_digit(n) for n in old_set - new_set}
                new_only = {two_digit(n) for n in new_set - old_set}
                s40_old = {two_digit(n) for n in old_set if two_digit(n) in S40}
                s40_new = {two_digit(n) for n in new_set if two_digit(n) in S40}
                p164_old = {two_digit(n) for n in old_set if two_digit(n) in PACK_164950_FAMILY}
                p164_new = {two_digit(n) for n in new_set if two_digit(n) in PACK_164950_FAMILY}
                print("\nOVERLAP & PACK VIEW:")
                print(f"  Numbers in both     : {sorted(overlap)}")
                print(f"  OLD only            : {sorted(old_only)}")
                print(f"  NEW only            : {sorted(new_only)}")
                print(f"  S40 in OLD          : {sorted(s40_old)}")
                print(f"  S40 in NEW          : {sorted(s40_new)}")
                print(f"  164950 in OLD       : {sorted(p164_old)}")
                print(f"  164950 in NEW       : {sorted(p164_new)}")

            print("\n2️⃣ P&L SNAPSHOT")
            print(
                f"   OLD  : ₹{pnl_snapshot_old['overall_profit']:.0f} "
                f"(ROI {pnl_snapshot_old['overall_roi']:.2f}%)"
            )
            print(
                f"   NEW  : ₹{pnl_snapshot_new['overall_profit']:.0f} "
                f"(ROI {pnl_snapshot_new['overall_roi']:.2f}%)"
            )

            print("\n3️⃣ PATTERN & LEARNING (shared history)")
            print(f"   Hits analyzed    : {pattern_stats['hits']}")
            print(
                f"   S40 family       : {pattern_stats['s40_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['s40_hits']} hits"
            )
            print(
                f"   164950 family    : {pattern_stats['pack_hit_rate']:.1f}% hit rate, "
                f"{pattern_stats['pack_hits']} hits"
            )
            print(
                f"   Near-miss count  : {pattern_stats['near_miss']} | Hot neighbors: {', '.join(pattern_stats['hot_neighbors'])}"
            )
            if pattern_stats.get("golden_pack"):
                print(f"   Golden pack      : {sorted(pattern_stats['golden_pack'])}")

            print("\n4️⃣ RISK & EXECUTION (new brain guiding)")
            conf_parts = ", ".join([f"{k} {v:.1f}%" for k, v in confidence_map.items()])
            risk_summary = compute_risk_execution_summary(pnl_snapshot_new, pattern_stats, confidence_map, u_live)
            print(f"   Strategy         : {risk_summary['strategy']}")
            print(f"   Risk mode        : {risk_summary['risk_mode']}")
            print(f"   Execution mode   : {risk_summary['execution']}")
            print(
                f"   Money manager    : daily cap ₹{risk_summary['money_manager']['daily_cap']:.0f}, "
                f"single-slot cap ₹{risk_summary['money_manager']['slot_cap']:.0f}"
            )
            print(f"   Confidence       : {conf_parts}")
            last_pred_block = new_pred

    if last_pred_block:
        strong_s40 = any(two_digit(x) in S40 for nums in last_pred_block["picks_by_slot"].values() for x in nums)
        if strong_s40:
            print("\n✨ S40 alignment detected in combined picks – treat as bonus confidence layer.")


if __name__ == "__main__":
    main()
