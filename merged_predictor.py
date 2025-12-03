import os, sys, json, math, random, itertools, collections, warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

EXCEL_FILE = os.path.join(os.path.dirname(__file__), "number prediction learn.xlsx")
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "pattern_memory.json")
SLOT_ORDER = ["FRBD", "GZBD", "GALI", "DSWR"]
SLOT_ID = {k: i + 1 for i, k in enumerate(SLOT_ORDER)}
S40 = {"00","06","07","09","15","16","18","19","24","25","27","28","33","34","36","37","42","43","45","46","51","52","54","55","60","61","63","64","70","72","73","79","81","82","88","89","90","91","97","98"}
PACK_164950_FAMILY = {"0","1","4","5","6","9"}


def is_valid_2d_number(x):
    try:
        n = int(x)
        return 0 <= n <= 99
    except Exception:
        return False


def to_2d(x):
    n = int(x)
    if n < 0 or n > 99:
        raise ValueError("2-digit number must be between 0 and 99")
    return f"{n:02d}"


def is_164950_family(num):
    s = to_2d(num)
    return s[0] in PACK_164950_FAMILY and s[1] in PACK_164950_FAMILY


def generate_pack(group_left, group_right):
    return sorted({f"{a}{b}" for a in group_left for b in group_right})


def generate_2digit_pack(A, B, C, D):
    return generate_pack([to_2d(A)[0], to_2d(B)[0]], [to_2d(C)[0], to_2d(D)[0]])


def generate_3digit_pack(A, B, C, X, Y, Z):
    return generate_pack([to_2d(A)[0], to_2d(B)[0], to_2d(C)[0]], [to_2d(X)[0], to_2d(Y)[0], to_2d(Z)[0]])


##############################
# DATA LOADER & STATUS CHECK #
##############################

def load_excel_data():
    if not os.path.exists(EXCEL_FILE):
        return pd.DataFrame(columns=["date", "slot", "number"])
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    cols = {c: str(c).strip().upper() for c in df.columns}
    df.columns = [cols[c] for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else list(df.columns)[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    records = []
    for _, row in df.iterrows():
        d = row[date_col]
        if pd.isna(d):
            continue
        for slot in SLOT_ORDER:
            if slot not in df.columns:
                continue
            raw = row[slot]
            if pd.isna(raw) or not is_valid_2d_number(raw):
                continue
            records.append({"date": pd.to_datetime(d), "slot": SLOT_ID[slot], "slot_name": slot, "number": int(raw) % 100})
    out = pd.DataFrame(records)
    return out.sort_values(["date", "slot"]).reset_index(drop=True)


def check_clock_status(df):
    today = datetime.now().date()
    today_rows = df[df["date"].dt.date == today]
    filled = set(today_rows["slot_name"].unique())
    missing = [s for s in SLOT_ORDER if s not in filled]
    target_date = today if missing else today + timedelta(days=1)
    return missing if missing else SLOT_ORDER, target_date


####################################
# CORE VALIDATION & UTILITY SCORES #
####################################

def recency_scores(df):
    scores = {slot: {} for slot in SLOT_ORDER}
    for slot in SLOT_ORDER:
        sub = df[df["slot_name"] == slot].sort_values("date")
        if sub.empty:
            continue
        weights = np.geomspace(1.0, 0.12, len(sub))
        weights = weights / weights.sum()
        for w, num in zip(weights, sub["number"]):
            key = to_2d(num)
            scores[slot][key] = scores[slot].get(key, 0.0) + float(w)
    return scores


def frequency_scores(df, window_days=40):
    scores = {slot: {} for slot in SLOT_ORDER}
    if df.empty:
        return scores
    cutoff = df["date"].max() - timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]
    for slot in SLOT_ORDER:
        sub = recent[recent["slot_name"] == slot]
        vc = sub["number"].value_counts()
        if vc.empty:
            continue
        mx = vc.max()
        for num, cnt in vc.items():
            scores[slot][to_2d(num)] = cnt / mx
    return scores


def cross_clock_scores(df):
    score = {}
    grouped = df.groupby("date")
    for _, day in grouped:
        dupes = day["number"].value_counts()
        for num, cnt in dupes.items():
            if cnt > 1:
                key = to_2d(num)
                score[key] = score.get(key, 0.0) + float(cnt)
    return score


def mirror_scores(df):
    score = {}
    for num, cnt in df["number"].value_counts().items():
        m = to_2d(num)[::-1]
        score[m] = score.get(m, 0.0) + float(cnt)
    return score


def near_miss_scores(df):
    score = {}
    for num, cnt in df["number"].value_counts().items():
        for neigh in [(num - 1) % 100, (num + 1) % 100]:
            key = to_2d(neigh)
            score[key] = score.get(key, 0.0) + float(cnt) * 0.5
    return score


def family_scores():
    return {f"{a}{b}": 1.0 for a in PACK_164950_FAMILY for b in PACK_164950_FAMILY}


def s40_scores():
    return {n: 1.0 for n in S40}


def pattern_memory_scores(patterns):
    scores = {}
    for num, stats in patterns.items():
        scores[num] = stats.get("hits", 0) * 0.55 + stats.get("near", 0) * 0.15 + stats.get("cross", 0) * 0.2 + stats.get("mirrors", 0) * 0.1
    return scores


def normalize_scores(scores):
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


#############################
# PATTERN MEMORY MANAGEMENT #
#############################

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "patterns": {},
        "script_weights": {f"scr{i}": 1.0 for i in range(1, 10)},
        "loss_streak": 0,
        "roi_trend": [900.0],
        "slump_guard": {slot: 0 for slot in SLOT_ORDER},
        "memory_rules": {"S40": list(S40), "family": list(PACK_164950_FAMILY)},
        "hit_memory": {},
    }


def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)


def update_pattern_memory(df, mem):
    patterns = mem.get("patterns", {})
    grouped = df.groupby("date")
    for _, day in grouped:
        counts = day["number"].value_counts()
        for num, cnt in counts.items():
            key = to_2d(num)
            entry = patterns.get(key, {"hits": 0, "near": 0, "cross": 0, "mirrors": 0})
            entry["hits"] += int(cnt)
            patterns[key] = entry
        for num in counts.index:
            mirror = to_2d(num)[::-1]
            patterns.setdefault(mirror, {"hits": 0, "near": 0, "cross": 0, "mirrors": 0})
            patterns[mirror]["mirrors"] += 1
            for near in [(num - 1) % 100, (num + 1) % 100]:
                nk = to_2d(near)
                patterns.setdefault(nk, {"hits": 0, "near": 0, "cross": 0, "mirrors": 0})
                patterns[nk]["near"] += 1
        dupes = counts[counts > 1]
        for num, cnt in dupes.items():
            key = to_2d(num)
            patterns.setdefault(key, {"hits": 0, "near": 0, "cross": 0, "mirrors": 0})
            patterns[key]["cross"] += int(cnt)
    mem["patterns"] = patterns
    return patterns


def adjust_script_weights(mem, slot_stats):
    weights = mem.get("script_weights", {f"scr{i}": 1.0 for i in range(1, 10)})
    for scr, info in slot_stats.items():
        change = 0.02 if info.get("roi", 0) > 500 else -0.02
        weights[scr] = max(0.2, min(1.8, weights.get(scr, 1.0) + change))
    mem["script_weights"] = weights
    return weights


def detect_cross_hits(df):
    cross = collections.defaultdict(int)
    grouped = df.groupby("date")
    for _, day in grouped:
        per_slot = {row.slot_name: to_2d(row.number) for row in day.itertuples(index=False)}
        for a, b in itertools.permutations(per_slot.items(), 2):
            cross[(a[0], b[0], a[1])] += 1
    return cross


###############################
# 9 PREDICTION ALGORITHMS (AI) #
###############################

def scr1_logic(df, slot):
    scores = recency_scores(df).get(slot, {})
    return sorted(scores, key=scores.get, reverse=True)


def scr2_logic(df, slot):
    freq = frequency_scores(df, 25).get(slot, {})
    return sorted(freq, key=freq.get, reverse=True)


def scr3_logic(df, slot):
    cross = cross_clock_scores(df)
    rec = recency_scores(df).get(slot, {})
    mix = {k: rec.get(k, 0) * 0.6 + cross.get(k, 0) * 0.4 for k in set(rec) | set(cross)}
    return sorted(mix, key=mix.get, reverse=True)


def scr4_logic(df, slot):
    mirrors = mirror_scores(df)
    near = near_miss_scores(df)
    mix = {k: mirrors.get(k, 0) + near.get(k, 0) for k in set(mirrors) | set(near)}
    return sorted(mix, key=mix.get, reverse=True)


def scr5_logic(df, slot):
    fam = family_scores()
    rec = recency_scores(df).get(slot, {})
    mix = {k: rec.get(k, 0) * 0.7 + fam.get(k, 0) * 0.3 for k in set(rec) | set(fam)}
    return sorted(mix, key=mix.get, reverse=True)


def scr6_logic(df, slot):
    s40 = s40_scores()
    freq = frequency_scores(df, 60).get(slot, {})
    mix = {k: freq.get(k, 0) * 0.5 + s40.get(k, 0) * 0.5 for k in set(freq) | set(s40)}
    return sorted(mix, key=mix.get, reverse=True)


def scr7_logic(df, slot):
    pat = pattern_memory_scores(update_pattern_memory(df, load_memory())).copy()
    rec = recency_scores(df).get(slot, {})
    mix = {k: pat.get(k, 0) * 0.6 + rec.get(k, 0) * 0.4 for k in set(pat) | set(rec)}
    return sorted(mix, key=mix.get, reverse=True)


def scr8_logic(df, slot):
    packs = generate_3digit_pack(1, 6, 4, 9, 5, 0)
    bonus = {p: 1.0 for p in packs}
    rec = recency_scores(df).get(slot, {})
    mix = {k: rec.get(k, 0) + bonus.get(k, 0) for k in set(rec) | set(bonus)}
    return sorted(mix, key=mix.get, reverse=True)


def scr9_logic(df, slot):
    all_scores = [recency_scores(df).get(slot, {}), frequency_scores(df).get(slot, {}), pattern_memory_scores(load_memory().get("patterns", {}))]
    agg = {}
    for s in all_scores:
        for k, v in s.items():
            agg[k] = agg.get(k, 0) + v
    return sorted(agg, key=agg.get, reverse=True)


###########################
# INTELLIGENT AGGREGATION #
###########################

def aggregate_predictions(df, mem, slot_id):
    algos = [scr1_logic, scr2_logic, scr3_logic, scr4_logic, scr5_logic, scr6_logic, scr7_logic, scr8_logic, scr9_logic]
    weights = mem.get("script_weights", {f"scr{i}": 1.0 for i in range(1, 10)})
    votes = collections.Counter()
    slot_name = SLOT_ORDER[slot_id - 1]
    for idx, algo in enumerate(algos, start=1):
        seq = algo(df, slot_name)
        weight = weights.get(f"scr{idx}", 1.0)
        for rank, num in enumerate(seq[:10], start=1):
            votes[num] += weight * (1.0 / rank)
    ranked = [n for n, _ in votes.most_common()]
    roi_trend = np.mean(mem.get("roi_trend", [900]))
    count = decide_number_count(roi_trend)
    preds = ranked[:count]
    confidence = normalize_scores({n: votes[n] for n in preds})
    return preds, confidence


def decide_number_count(roi_trend):
    if roi_trend < 300:
        return 5
    if roi_trend < 600:
        return 4
    return 3


def select_and_bahar(predictions):
    if not predictions:
        return "0", "0"
    tens = [p[0] for p in predictions]
    ones = [p[1] for p in predictions]
    return max(set(tens), key=tens.count), max(set(ones), key=ones.count)


############################
# FORENSIC AUDIT & PREVENT #
############################

def audit_date_alignment(df):
    issues = []
    grouped = df.groupby("date")
    for d, day in grouped:
        expected = set(SLOT_ORDER)
        actual = set(day["slot_name"])
        if actual != expected:
            issues.append(f"{d.date()} slots mismatch {sorted(expected-actual)} missing")
    return issues


def analyze_frbd_slump(df):
    frbd = df[df["slot_name"] == "FRBD"].sort_values("date")
    if frbd.empty:
        return {"fracture": None, "curve": []}
    curve = []
    profit = 0
    for row in frbd.itertuples(index=False):
        profit += 90 - 5
        curve.append((row.date.date(), profit))
    diffs = [curve[i + 1][1] - curve[i][1] for i in range(len(curve) - 1)]
    fracture = None
    for i, d in enumerate(diffs):
        if d < 0:
            fracture = curve[i + 1][0]
            break
    return {"fracture": fracture, "curve": curve[-10:]}


def audit_risk_weights(mem):
    weights = mem.get("script_weights", {})
    capped = {k: min(1.8, max(0.2, v)) for k, v in weights.items()}
    throttle = {slot: min(1.0, 1.0 - mem.get("slump_guard", {}).get(slot, 0) * 0.1) for slot in SLOT_ORDER}
    return capped, throttle


def slump_detector(mem, roi):
    trend = mem.get("roi_trend", [])[-8:]
    trend.append(roi)
    mem["roi_trend"] = trend[-12:]
    if roi < 300:
        for slot in SLOT_ORDER:
            mem.setdefault("slump_guard", {})[slot] = min(5, mem.get("slump_guard", {}).get(slot, 0) + 1)
    else:
        mem["slump_guard"] = {s: max(0, v - 1) for s, v in mem.get("slump_guard", {}).items()}


########################
# BACKTEST & RECOMPUTE #
########################

def run_backtest(df, days=30, picks=4):
    if df.empty:
        return {"roi": 1000.0, "hits": 0, "bets": 0, "slot_roi": {}}
    df = df.sort_values(["date", "slot"]).reset_index(drop=True)
    end_cutoff = df["date"].max()
    start_cutoff = end_cutoff - timedelta(days=days)
    df = df[df["date"] >= start_cutoff]
    hits = bets = 0
    slot_perf = {s: {"hits": 0, "bets": 0} for s in SLOT_ORDER}
    for idx, row in df.iterrows():
        hist = df[(df["date"] < row["date"]) & (df["slot"] == row["slot"])]
        if hist.empty:
            continue
        freq = hist["number"].value_counts()
        top = [to_2d(n) for n in freq.head(picks).index]
        bets += picks
        slot_name = SLOT_ORDER[row["slot"] - 1]
        slot_perf[slot_name]["bets"] += picks
        if to_2d(row["number"]) in top:
            hits += 1
            slot_perf[slot_name]["hits"] += 1
    roi = ((hits * 90 - bets) / bets * 100) if bets else 1000.0
    slot_roi = {k: (((v["hits"] * 90 - v["bets"]) / v["bets"] * 100) if v["bets"] else 0.0) for k, v in slot_perf.items()}
    return {"roi": roi, "hits": hits, "bets": bets, "slot_roi": slot_roi}


def recompute_if_needed(df, mem, base_roi):
    if base_roi >= 500:
        return base_roi
    alt = run_backtest(df, days=20, picks=5)
    mem["roi_trend"].append(max(base_roi, alt["roi"]))
    return max(base_roi, alt["roi"])


###############################
# MONEY & STAKE MANAGEMENT    #
###############################

def loss_recovery_engine(mem):
    streak = mem.get("loss_streak", 0)
    return min(16, 2 ** streak)


def calculate_dynamic_stakes(predictions, confidence, risk_throttle, mem):
    base = loss_recovery_engine(mem)
    stakes = {}
    for num in predictions:
        adj = 1 + confidence.get(num, 0.0)
        stakes[num] = round(base * adj * risk_throttle, 2)
    return stakes


def apply_daily_caps(stakes, cap=200, single_cap=70):
    total = sum(stakes.values())
    if total > cap:
        factor = cap / total
        stakes = {k: round(v * factor, 2) for k, v in stakes.items()}
    return {k: min(single_cap, v) for k, v in stakes.items()}


def expected_pl(stakes, confidence):
    avg_conf = np.mean(list(confidence.values())) if confidence else 0.0
    return sum(90 * stakes[n] for n in stakes) * avg_conf - sum(stakes.values()), avg_conf


def andar_bahar_stake(base):
    return round(max(1.0, base * 0.2), 2)


#############################
# REPORTING & PRESENTATION  #
#############################

def confidence_label(score):
    if score >= 0.8:
        return "VERY_HIGH"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def render_slot(slot, day_label, preds, stakes, andar, bahar, conf, throttle):
    scores = {n: int(conf.get(n, 0.0) * 100) for n in preds}
    label_score = int(np.mean(list(scores.values()))) if scores else 0
    tag = confidence_label(label_score / 100)
    stakes_txt = ", ".join([f"{n}(₹{stakes[n]})" for n in preds]) if preds else "-"
    num_txt = ", ".join([f"{n}({chr(65+i)})" for i, n in enumerate(preds)]) if preds else "-"
    exp_pl, avg_conf = expected_pl(stakes, conf)
    print(f"\nPredictions for {day_label} - {slot}")
    print(f"Numbers: {num_txt}")
    print(f"Stakes: {stakes_txt}")
    print(f"Andar: {andar} Bahar: {bahar} (₹{andar_bahar_stake(throttle):.2f} each)")
    print(f"Confidence: {tag} ({label_score}) | Expected P/L: ₹{exp_pl:.1f}")


def render_pl_snapshot(bt, last7):
    overall_pl = bt["hits"] * 90 - bt["bets"]
    last_pl = last7["hits"] * 90 - last7["bets"]
    best = max(bt["slot_roi"].items(), key=lambda x: x[1]) if bt["slot_roi"] else ("n/a", 0)
    worst = min(bt["slot_roi"].items(), key=lambda x: x[1]) if bt["slot_roi"] else ("n/a", 0)
    print("\nP&L SNAPSHOT:")
    print(f"Overall P&L: ₹{overall_pl:.0f} (ROI {bt['roi']:.1f}%)")
    print(f"Last 7 days: ₹{last_pl:.0f} (ROI {last7['roi']:.1f}%)")
    print(f"Best slot: {best[0]} ₹{(best[1]/100)*bt['slot_roi'].get(best[0],0):.0f}")
    print(f"Weak slot: {worst[0]} ₹{(worst[1]/100)*bt['slot_roi'].get(worst[0],0):.0f}")


def render_pattern_learning(df, mem, cross_hits):
    total = len(df)
    s40_hits = df[df["number"].apply(lambda n: to_2d(n) in S40)]
    fam_hits = df[df["number"].apply(is_164950_family)]
    s40_rate = (len(s40_hits) / total * 100) if total else 0.0
    fam_rate = (len(fam_hits) / total * 100) if total else 0.0
    print("\nPATTERN LEARNING:")
    print(f"S40 family: {s40_rate:.1f}% hit rate, {len(s40_hits)} hits")
    print(f"164950 family: {fam_rate:.1f}% hit rate, {len(fam_hits)} hits")
    top_cross = sorted(cross_hits.items(), key=lambda x: x[1], reverse=True)
    if top_cross:
        a, b, num = top_cross[0][0]
        print(f"Cross-slot patterns: {a}→{b} ({num}) x{top_cross[0][1]}")
    else:
        print("Cross-slot patterns: none yet")


def render_risk_execution(strategy, risk_mode, execution, caps):
    print("\nRISK & EXECUTION:")
    print(f"Strategy: {strategy}")
    print(f"Risk mode: {risk_mode}")
    print(f"Execution: {execution}")
    print(f"Daily cap: ₹{caps['daily']} , Single cap: ₹{caps['single']}")


#############################
# MAIN EXECUTION PIPELINE   #
#############################

def main():
    df = load_excel_data()
    mem = load_memory()
    patterns = update_pattern_memory(df, mem)
    base_bt = run_backtest(df, days=30, picks=4)
    last7_bt = run_backtest(df, days=7, picks=4)
    roi = recompute_if_needed(df, mem, base_bt["roi"])
    slump_detector(mem, roi)
    capped_weights, throttle_map = audit_risk_weights(mem)
    mem["script_weights"] = capped_weights
    slots_to_predict, target_date = check_clock_status(df)
    cross_hits = detect_cross_hits(df)

    start_label = df["date"].min().date() if not df.empty else "n/a"
    end_label = df["date"].max().date() if not df.empty else "n/a"
    print(f"Data window: {start_label} → {end_label}")
    print(f"Backtest ROI: {base_bt['roi']:.1f}% ({base_bt['hits']} hits / {base_bt['bets']} bets)")

    slot_stats = {}
    day_label = target_date.isoformat()
    for slot in slots_to_predict:
        preds, conf = aggregate_predictions(df, mem, SLOT_ID[slot])
        risk_throttle = throttle_map.get(slot, 1.0)
        stakes = calculate_dynamic_stakes(preds, conf, risk_throttle, mem)
        stakes = apply_daily_caps(stakes)
        andar, bahar = select_and_bahar(preds)
        render_slot(slot, day_label, preds, stakes, andar, bahar, conf, risk_throttle)
        exp_pl, _ = expected_pl(stakes, conf)
        slot_stats[f"scr{SLOT_ID[slot]}"] = {"roi": roi + exp_pl}
    adjust_script_weights(mem, slot_stats)

    render_pl_snapshot(base_bt, last7_bt)
    render_pattern_learning(df, mem, cross_hits)
    render_risk_execution("STRAT_S40_BOOST", "DEFENSIVE" if roi < 600 else "BALANCED", "GO_LIVE_LIGHT" if roi < 600 else "GO_LIVE", {"daily": 200, "single": 70})

    issues = audit_date_alignment(df)
    slump = analyze_frbd_slump(df)
    if issues:
        print("\nAUDIT ALERTS:")
        for i in issues:
            print("-", i)
    if slump.get("fracture"):
        print(f"FRBD slump detected at {slump['fracture']}")
    if roi < 500:
        mem["loss_streak"] = mem.get("loss_streak", 0) + 1
    else:
        mem["loss_streak"] = 0
    save_memory(mem)


if __name__ == "__main__":
    main()
