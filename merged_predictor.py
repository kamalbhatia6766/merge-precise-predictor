import json
import os
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import urllib.request

EXCEL_FILE = os.path.join(os.path.dirname(__file__), "number prediction learn.xlsx")
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "pattern_memory.json")
START_DATE = datetime(2025, 1, 1)
S40 = {"00","06","07","09","15","16","18","19","24","25","27","28","33","34","36","37","42","43","45","46","51","52","54","55","60","61","63","64","70","72","73","79","81","82","88","89","90","91","97","98"}
PACK_164950_FAMILY = {"0","1","4","5","6","9"}
SLOT_MAP = {"FRBD":1,"GZBD":2,"GALI":3,"DSWR":4}


def is_valid_2d_number(x) -> bool:
    try:
        n = int(x)
    except Exception:
        return False
    return 0 <= n <= 99


def to_2d(x) -> str:
    n = int(x)
    if n < 0 or n > 99:
        raise ValueError("2-digit number must be between 0 and 99")
    return f"{n:02d}"


def is_164950_family(num: int) -> bool:
    s = to_2d(num)
    return s[0] in PACK_164950_FAMILY and s[1] in PACK_164950_FAMILY


def generate_pack(group_left: List[str], group_right: List[str]) -> List[str]:
    return sorted({f"{a}{b}" for a in group_left for b in group_right})


def generate_2digit_pack(A,B,C,D):
    return generate_pack([to_2d(A)[0], to_2d(B)[0]],[to_2d(C)[0], to_2d(D)[0]])


def generate_3digit_pack(A,B,C,X,Y,Z):
    return generate_pack([to_2d(A)[0], to_2d(B)[0], to_2d(C)[0]],[to_2d(X)[0], to_2d(Y)[0], to_2d(Z)[0]])


def generate_ndigit_pack(*digits):
    mid = len(digits)//2
    left = [to_2d(d)[0] for d in digits[:mid]]
    right = [to_2d(d)[0] for d in digits[mid:]]
    return generate_pack(left, right)


@lru_cache(maxsize=1)
def load_excel() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    cols = {c:str(c).strip().upper() for c in df.columns}
    df.columns = [cols[c] for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else list(df.columns)[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df[(df[date_col] >= START_DATE) & (df[date_col] <= datetime.now())]
    records = []
    for _, row in df.iterrows():
        d = row[date_col]
        if pd.isna(d):
            continue
        for col, slot_id in SLOT_MAP.items():
            if col not in df.columns:
                continue
            raw = row[col]
            if pd.isna(raw):
                continue
            s = str(raw).strip()
            if not is_valid_2d_number(s):
                continue
            records.append({"date": pd.to_datetime(d), "slot": slot_id, "number": int(s)%100})
    if not records:
        return pd.DataFrame(columns=["date","slot","number"])
    out = pd.DataFrame(records)
    out = out.sort_values(["date","slot"]).reset_index(drop=True)
    return out


def load_memory() -> Dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"patterns":{},"roi":800.0,"loss_streak":0,"last_update":None,"remote_hash":None}


def save_memory(mem: Dict):
    mem["last_update"] = datetime.now().isoformat()
    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        json.dump(mem,f,indent=2)


def check_remote_update(mem: Dict):
    url = "https://raw.githubusercontent.com/kamalbhatia6766/merge-precise-predictor/main/README.md"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        remote_hash = hashlib.sha256(data).hexdigest()
        if mem.get("remote_hash") and mem["remote_hash"] != remote_hash:
            print("⚡ Detected GitHub update; consider refreshing local logic.")
        mem["remote_hash"] = remote_hash
    except Exception:
        print("⚠️  Unable to verify GitHub updates (offline)")


def discover_patterns(df: pd.DataFrame, mem: Dict) -> Dict:
    patterns = mem.get("patterns", {})
    grouped = df.groupby("date")
    for _, day in grouped:
        counts = day["number"].value_counts()
        for num, cnt in counts.items():
            key = to_2d(num)
            entry = patterns.get(key, {"hits":0,"mirrors":0,"near":0,"cross":0})
            entry["hits"] += int(cnt)
            patterns[key] = entry
        cross_numbers = [n for n,c in counts.items() if c>1]
        for n in cross_numbers:
            key = to_2d(n)
            patterns[key]["cross"] = patterns[key].get("cross",0)+1
        for num in counts.index:
            mirror = to_2d(num)[::-1]
            if mirror != to_2d(num):
                patterns.setdefault(mirror,{"hits":0,"mirrors":0,"near":0,"cross":0})
                patterns[mirror]["mirrors"] += 1
            near_list = [(num-1)%100,(num+1)%100]
            for near in near_list:
                nk = to_2d(near)
                patterns.setdefault(nk,{"hits":0,"mirrors":0,"near":0,"cross":0})
                patterns[nk]["near"] += 1
    mem["patterns"] = patterns
    return patterns


def recency_scores(df: pd.DataFrame) -> Dict[int, Dict[str,float]]:
    scores = {slot:{} for slot in SLOT_MAP.values()}
    for slot in SLOT_MAP.values():
        sub = df[df["slot"]==slot].sort_values("date")
        if sub.empty:
            continue
        weights = np.geomspace(1.0,0.1,len(sub))
        weights = weights/weights.sum()
        for w, num in zip(weights, sub["number"]):
            key = to_2d(num)
            scores[slot][key] = scores[slot].get(key,0.0)+float(w)
    return scores


def frequency_scores(df: pd.DataFrame, window_days:int=30) -> Dict[int, Dict[str,float]]:
    scores = {slot:{} for slot in SLOT_MAP.values()}
    if df.empty:
        return scores
    cutoff = df["date"].max() - timedelta(days=window_days)
    recent = df[df["date"]>=cutoff]
    for slot in SLOT_MAP.values():
        sub = recent[recent["slot"]==slot]
        vc = sub["number"].value_counts()
        if vc.empty:
            continue
        mx = vc.max()
        for num, cnt in vc.items():
            scores[slot][to_2d(num)] = cnt/mx
    return scores


def cross_clock_scores(df: pd.DataFrame) -> Dict[str,float]:
    score = {}
    grouped = df.groupby("date")
    for _, day in grouped:
        dupes = day["number"].value_counts()
        for num, cnt in dupes.items():
            if cnt > 1:
                key = to_2d(num)
                score[key] = score.get(key,0.0) + cnt
    return score


def s40_scores() -> Dict[str,float]:
    return {n:1.0 for n in S40}


def mirror_scores(df: pd.DataFrame) -> Dict[str,float]:
    score = {}
    vc = df["number"].value_counts()
    for num, cnt in vc.items():
        m = to_2d(num)[::-1]
        score[m] = score.get(m,0.0) + float(cnt)
    return score


def near_miss_scores(df: pd.DataFrame) -> Dict[str,float]:
    score = {}
    vc = df["number"].value_counts()
    for num, cnt in vc.items():
        for neigh in [(num-1)%100,(num+1)%100]:
            key = to_2d(neigh)
            score[key] = score.get(key,0.0) + float(cnt)*0.5
    return score


def family_scores() -> Dict[str,float]:
    return {f"{a}{b}":1.0 for a in PACK_164950_FAMILY for b in PACK_164950_FAMILY}


def pattern_memory_scores(patterns: Dict[str,Dict[str,int]]) -> Dict[str,float]:
    scores = {}
    for num, stats in patterns.items():
        hits = stats.get("hits",0)
        near = stats.get("near",0)
        cross = stats.get("cross",0)
        mir = stats.get("mirrors",0)
        scores[num] = hits*0.6 + near*0.2 + cross*0.15 + mir*0.05
    return scores


def blend_scores(weighted_sources: List[Tuple[Dict[str,float], float]]) -> Dict[str,float]:
    agg = {}
    for src, w in weighted_sources:
        for num, val in src.items():
            agg[num] = agg.get(num,0.0) + val*w
    return agg


def normalize_scores(scores: Dict[str,float]) -> Dict[str,float]:
    if not scores:
        return {}
    arr = np.array(list(scores.values()), dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return {k:0.0 for k in scores}
    return {k:(v-mn)/(mx-mn) for k,v in scores.items()}


def backtest_roi(df: pd.DataFrame, days:int=60, lookback:int=30, picks:int=5) -> Tuple[float,int,int]:
    if df.empty:
        return 1000.0,0,0
    df = df.sort_values(["date","slot"]).reset_index(drop=True)
    end_cutoff = df["date"].max()
    start_cutoff = end_cutoff - timedelta(days=days)
    df = df[df["date"]>=start_cutoff]
    hits = 0
    total_bets = 0
    for idx, row in df.iterrows():
        current_date = row["date"]
        slot = row["slot"]
        hist = df[(df["date"] < current_date) & (df["date"] >= current_date - timedelta(days=lookback))]
        if hist.empty:
            continue
        freq = hist[hist["slot"]==slot]["number"].value_counts()
        if freq.empty:
            continue
        top = [to_2d(n) for n in freq.head(picks).index]
        total_bets += picks
        actual = to_2d(row["number"])
        if actual in top:
            hits += 1
    if total_bets == 0:
        return 1000.0,hits,total_bets
    profit = hits*90 - total_bets
    roi = (profit/total_bets)*100
    return roi,hits,total_bets


def determine_output_count(roi: float) -> int:
    if roi < 400:
        return 3
    if roi < 800:
        return 4
    return 5


def calculate_stakes(preds: List[str], confidence: Dict[str,float], loss_streak:int) -> Dict[str,float]:
    base = min(8, 2 ** max(0, loss_streak))
    stakes = {}
    for num in preds:
        stakes[num] = round(base * (1 + confidence.get(num,0.0)), 2)
    return stakes


def select_andar_bahar(preds: List[str]) -> Tuple[str,str]:
    if not preds:
        return "0","0"
    tens = [p[0] for p in preds]
    ones = [p[1] for p in preds]
    andar = max(set(tens), key=tens.count)
    bahar = max(set(ones), key=ones.count)
    return andar, bahar


def build_confidence(scores: Dict[str,float]) -> Dict[str,float]:
    return normalize_scores(scores)


def predict_for_slot(slot:int, df: pd.DataFrame, patterns: Dict[str,Dict[str,int]], roi: float, mem: Dict) -> Tuple[List[str], Dict[str,float]]:
    rec = recency_scores(df)
    freq = frequency_scores(df)
    cross = cross_clock_scores(df)
    mirrors = mirror_scores(df)
    near = near_miss_scores(df)
    fam = family_scores()
    s40 = s40_scores()
    pat = pattern_memory_scores(patterns)
    base_weights = [
        (rec.get(slot,{}), 0.22),
        (freq.get(slot,{}), 0.2),
        (pat, 0.14),
        (cross, 0.1),
        (mirrors,0.08),
        (near,0.08),
        (fam,0.09),
        (s40,0.09)
    ]
    if roi < 500:
        base_weights.append((s40,0.1))
    blended = blend_scores(base_weights)
    blended = normalize_scores(blended)
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    out_count = determine_output_count(roi)
    preds = [n for n,_ in ranked[:out_count]]
    confidence = build_confidence({n:s for n,s in ranked[:out_count]})
    return preds, confidence


def summarize_projection(preds: List[str], stakes: Dict[str,float], andar: str, bahar: str, confidence: Dict[str,float]):
    print("Numbers:", ", ".join(preds))
    print("Stakes:", ", ".join([f"{n}:₹{stakes[n]}" for n in preds]))
    print(f"Andar: {andar}  Bahar: {bahar}")
    avg_conf = sum(confidence.values())/len(confidence) if confidence else 0.0
    est_profit = sum(90*stakes[n] for n in preds)*avg_conf - sum(stakes.values())
    print(f"Confidence: {avg_conf*100:.1f}% | Expected P/L: ₹{est_profit:.1f}")


def evaluate_today_slots(df: pd.DataFrame) -> Tuple[List[int], datetime]:
    today = datetime.now().date()
    todays = df[df["date"].dt.date == today]
    filled_slots = set(todays["slot"].unique())
    missing = [s for s in SLOT_MAP.values() if s not in filled_slots]
    target_date = datetime.now().date() if missing else (today + timedelta(days=1))
    return missing if missing else list(SLOT_MAP.values()), datetime.combine(target_date, datetime.min.time())


def main():
    df = load_excel()
    mem = load_memory()
    check_remote_update(mem)
    patterns = discover_patterns(df, mem)
    roi, hits, bets = backtest_roi(df)
    mem["roi"] = roi
    target_slots, target_date = evaluate_today_slots(df)
    print(f"Data window: {df['date'].min().date() if not df.empty else 'n/a'} → {df['date'].max().date() if not df.empty else 'n/a'}")
    print(f"Backtest ROI: {roi:.1f}% ({hits} hits / {bets} bets)")
    day_label = target_date.strftime('%d-%m-%Y')
    for slot in target_slots:
        preds, conf = predict_for_slot(slot, df, patterns, roi, mem)
        stakes = calculate_stakes(preds, conf, mem.get("loss_streak",0))
        andar, bahar = select_andar_bahar(preds)
        print(f"\nPredictions for {day_label} - Clock {slot}")
        summarize_projection(preds, stakes, andar, bahar, conf)
    if roi < 500:
        mem["loss_streak"] = mem.get("loss_streak",0)+1
    else:
        mem["loss_streak"] = 0
    save_memory(mem)


if __name__ == "__main__":
    main()
