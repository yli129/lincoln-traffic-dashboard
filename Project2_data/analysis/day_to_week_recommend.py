"""
Day-to-Week recommendation (annual factors only).

Design
------
* Reference intersection A: full-year data -> annual weekday factors.
* Target intersection B (B != A): one day's count per week is used to estimate
  B's Weekly ADT; the truth is the real Mon-Sun mean of that week.
* All ordered pairs (A, B) of the intersections are tested
  (3 intersections -> 6 pairs).
* 6 sections (5 motor-vehicle sections + All Motor Vehicles) x 5 directions
  (4 approaches + Total).

Products
--------
1. predict_weekly_adt(): one day's volume -> Weekly ADT
2. recommend(): which weekday to collect on, with expected error
3. main(): writes the CSV tables used by the dashboard page

Run
---
    python Project2_data/analysis/day_to_week_recommend.py

Reads  Project2_data/cleaned_data/*_Binned_*.csv
Writes Project2_data/outputs/day_to_week/*.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # shared code lives in src/

from day_to_week import (
    prepare_daily,
    select_daily_volume,
    make_week_table,
    calculate_annual_factors,
    estimate_weekly_adt,
    _week_arrays,
    WEEKDAY_NAMES,
)
from shared_traffic_utils import DIRECTIONS

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
CLEANED_FOLDER = PROJECT_FOLDER / "cleaned_data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "day_to_week"

SECTIONS = [
    "All Motor Vehicles",
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]
DIRECTION_OPTIONS = ["Total"] + DIRECTIONS

# Reliability of the single-day estimate, judged from the best weekday's WAPE
HIGH_WAPE = 10.0
MODERATE_WAPE = 20.0
MIN_WEEKLY_ADT = 10.0      # below this average daily volume: no recommendation
TIE_ALPHA = 0.05
TIE_REL_MARGIN = 1.25      # an equivalent day may have up to 25% higher WAPE ...
TIE_ABS_MARGIN = 1.0       # ... or 1 percentage point, whichever is larger


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------
def get_intersection_files(cleaned_folder=CLEANED_FOLDER):
    """Find CSV files and group them by intersection."""
    groups = {}
    for file in sorted(Path(cleaned_folder).glob("*_Binned_*.csv")):
        name = file.name.split("_Binned_")[0]
        groups.setdefault(name, []).append(file)
    return groups


class WeeklyCache:
    """Loads each (intersection, section) once and builds week tables."""

    def __init__(self, intersection_files):
        self.files = intersection_files
        self._daily = {}

    def weekly(self, name, section, direction):
        key = (name, section)
        if key not in self._daily:
            self._daily[key] = prepare_daily(self.files[name], section)
        return make_week_table(
            select_daily_volume(self._daily[key], section, direction)
        )


# ----------------------------------------------------------------------
# Part 1: prediction
# ----------------------------------------------------------------------
def predict_weekly_adt(weekly_A, weekday, volume):
    """
    Estimate Weekly ADT from one day's count.

    weekly_A : week table of the reference intersection (full year)
    weekday  : 0=Mon ... 6=Sun (weekday on which `volume` was counted)
    volume   : counted daily volume
    Returns (estimated_adt, factor).
    """
    factors = calculate_annual_factors(weekly_A).set_index("Weekday")["Factor"]
    factor = factors.get(weekday, np.nan)
    if pd.isna(factor):
        raise ValueError("Factor not available for this weekday.")
    return volume * factor, factor


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def run_tests(cache, sections=SECTIONS, directions=DIRECTION_OPTIONS,
              verbose=True):
    """Per-week estimation records for every ordered pair, section, direction."""
    names = list(cache.files)
    pairs = [(a, b) for a in names for b in names if a != b]

    frames = []
    for a, b in pairs:
        for section in sections:
            for direction in directions:
                try:
                    wa = cache.weekly(a, section, direction)
                    wb = cache.weekly(b, section, direction)
                    res = estimate_weekly_adt(wa, wb, "Annual",
                                              leave_one_out=False, min_train_weeks=2)
                except ValueError:
                    continue
                res = res.dropna(subset=["Estimated"])
                res = res[res["Actual"] > 0]
                if res.empty:
                    continue
                res.insert(0, "Direction", direction)
                res.insert(0, "Section", section)
                res.insert(0, "B", b)
                res.insert(0, "A", a)
                frames.append(res[["A", "B", "Section", "Direction", "Week", "Weekday",
                                   "Sample", "Factor", "Estimated", "Actual",
                                   "Err", "AE", "APE"]])
        if verbose:
            print(f"tested {a} -> {b}")

    return pd.concat(frames, ignore_index=True)


def build_factor_table(cache, sections=SECTIONS, directions=DIRECTION_OPTIONS):
    """Annual factor of every weekday for each (reference, section, direction)."""
    rows = []
    for name in cache.files:
        for section in sections:
            for direction in directions:
                try:
                    weekly = cache.weekly(name, section, direction)
                    factors = calculate_annual_factors(weekly)
                    _, _, adt, _ = _week_arrays(weekly)
                except ValueError:
                    continue
                factors["A"] = name
                factors["Section"] = section
                factors["Direction"] = direction
                factors["Mean_Weekly_ADT"] = adt.mean()
                rows.append(factors)
    out = pd.concat(rows, ignore_index=True)
    return out[["A", "Section", "Direction", "Weekday", "Factor",
                "N_Weeks", "Mean_Weekly_ADT"]]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def weekday_metrics(results, by):
    """MAE, MAPE, WAPE, bias and P90 APE per weekday, grouped by `by`."""
    g = results.groupby(list(by) + ["Weekday"])
    out = g.agg(
        MAE=("AE", "mean"),
        MAPE=("APE", "mean"),
        P90_APE=("APE", lambda s: np.nanpercentile(s, 90)),
        Sum_AE=("AE", "sum"),
        Sum_Err=("Err", "sum"),
        Sum_Actual=("Actual", "sum"),
        Weekly_ADT=("Actual", "mean"),
        N=("AE", "size"),
    ).reset_index()
    out["WAPE"] = out["Sum_AE"] / out["Sum_Actual"] * 100
    out["Bias_pct"] = out["Sum_Err"] / out["Sum_Actual"] * 100
    out["Day"] = out["Weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    return out.drop(columns=["Sum_AE", "Sum_Err", "Sum_Actual"])


def _reliability(best_wape, weekly_adt):
    if not np.isfinite(best_wape) or weekly_adt < MIN_WEEKLY_ADT:
        return "Insufficient volume"
    if best_wape <= HIGH_WAPE:
        return "High"
    if best_wape <= MODERATE_WAPE:
        return "Moderate"
    return "Low"


def _tied_days(group, best_day, day_wape, alpha=TIE_ALPHA):
    """
    Weekdays that are practically as good as the best day.

    A day counts as equivalent only if BOTH hold:
      1. its paired APE is not significantly worse than the best day
         (one-sided Wilcoxon test), and
      2. its WAPE is within a small margin of the best day:
         at most max(best * TIE_REL_MARGIN, best + TIE_ABS_MARGIN).
    Condition 2 is needed because "not significantly worse" can also mean
    "too noisy to tell", which would otherwise let clearly worse days in.
    """
    from scipy.stats import wilcoxon

    best_wape = day_wape[best_day]
    limit = max(best_wape * TIE_REL_MARGIN, best_wape + TIE_ABS_MARGIN)

    wide = group.pivot_table(index=["A", "B", "Week"], columns="Weekday", values="APE")
    tied = [best_day]
    for d in range(7):
        if d == best_day or d not in wide or best_day not in wide:
            continue
        if not (day_wape.get(d, np.inf) <= limit):
            continue
        diff = (wide[d] - wide[best_day]).dropna()
        if len(diff) < 10 or (diff == 0).all():
            tied.append(d)
            continue
        # one-sided: is day d worse (larger APE) than the best day?
        p = wilcoxon(diff, alternative="greater").pvalue
        if p >= alpha:
            tied.append(d)
    return sorted(tied)


def recommend(results, by):
    """
    One row per group in `by` (e.g. Section+Direction, or A+Section+Direction).

    Best_Day      weekday with the lowest pooled WAPE
    Equivalent    weekdays statistically tied with the best day
    Agreement     share of (A,B) pairs whose own best day is inside Equivalent
    Reliability   High / Moderate / Low / Insufficient volume
    """
    metrics = weekday_metrics(results, by)
    rows = []
    for key, g in metrics.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        best = g.loc[g["WAPE"].idxmin()]
        best_day = int(best["Weekday"])

        mask = np.ones(len(results), dtype=bool)
        for col, val in zip(by, key):
            mask &= (results[col] == val).to_numpy()
        sub = results[mask]

        pair_best = (
            weekday_metrics(sub, ["A", "B"])
            .loc[lambda d: d.groupby(["A", "B"])["WAPE"].idxmin()]["Weekday"]
        )
        day_wape = g.set_index("Weekday")["WAPE"].to_dict()
        tied = _tied_days(sub, best_day, day_wape)
        agreement = pair_best.isin(tied).mean() * 100
        rel = _reliability(best["WAPE"], best["Weekly_ADT"])

        row = dict(zip(by, key))
        row.update({
            "Best_Day": WEEKDAY_NAMES[best_day],
            "Best_WAPE": best["WAPE"],
            "Best_MAE": best["MAE"],
            "Best_P90_APE": best["P90_APE"],
            "Equivalent_Days": ", ".join(WEEKDAY_NAMES[d] for d in tied),
            "Worst_Day": WEEKDAY_NAMES[int(g.loc[g["WAPE"].idxmax(), "Weekday"])],
            "Worst_WAPE": g["WAPE"].max(),
            "Agreement_pct": agreement,
            "N_Pairs": sub[["A", "B"]].drop_duplicates().shape[0],
            "Mean_Weekly_ADT": best["Weekly_ADT"],
            "Reliability": rel,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def advice_text(row):
    """One-line sentence for the dashboard."""
    if row["Reliability"] == "Insufficient volume":
        return ("Traffic volume is too low for a reliable day-to-week estimate; "
                "collect a full week instead.")
    others = [d.strip() for d in row["Equivalent_Days"].split(",")
              if d.strip() != row["Best_Day"]]
    alt = f" (also acceptable: {', '.join(others)})" if others else ""
    base = (f"Collect on {row['Best_Day']}{alt}; "
            f"typical error about {row['Best_WAPE']:.1f}% (WAPE).")
    if row["Reliability"] == "Low":
        base += " Accuracy is low for this case; treat the estimate with caution."
    return base


# ----------------------------------------------------------------------
def main(cleaned_folder=CLEANED_FOLDER, output_folder=OUTPUT_FOLDER):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = get_intersection_files(cleaned_folder)
    if len(files) < 2:
        raise SystemExit(f"Need at least two intersections in {cleaned_folder}")

    cache = WeeklyCache(files)
    results = run_tests(cache)

    build_factor_table(cache).to_csv(output_folder / "annual_factors.csv", index=False)

    weekday_metrics(results, ["A", "B", "Section", "Direction"]).to_csv(
        output_folder / "weekday_metrics_by_pair.csv", index=False)
    weekday_metrics(results, ["A", "Section", "Direction"]).to_csv(
        output_folder / "weekday_metrics_by_reference.csv", index=False)
    weekday_metrics(results, ["Section", "Direction"]).to_csv(
        output_folder / "weekday_metrics_pooled.csv", index=False)

    pooled = recommend(results, ["Section", "Direction"])
    pooled["Advice"] = pooled.apply(advice_text, axis=1)
    pooled.to_csv(output_folder / "recommendation_pooled.csv", index=False)

    by_ref = recommend(results, ["A", "Section", "Direction"])
    by_ref["Advice"] = by_ref.apply(advice_text, axis=1)
    by_ref.to_csv(output_folder / "recommendation_by_reference.csv", index=False)

    print(f"saved to {output_folder}")


if __name__ == "__main__":
    main()
