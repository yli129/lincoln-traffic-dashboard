"""
Day-to-Week estimation.

Estimate Weekly ADT (mean of Mon-Sun) from a single day's count:

    Weekly_ADT_hat = Daily_Volume(weekday d) * Factor(d)

    Annual factor  (d)    = sum(Weekly_ADT) / sum(Volume on weekday d)
                            over all complete reference weeks of the year
    Monthly factor (m, d) = same ratio, but only over reference weeks whose
                            weekday-d date falls in month m

Evaluation
----------
* Reference intersection A supplies the factors, target intersection B
  supplies the "truth" and the single-day sample.
* If A and B are the same intersection, the test week is removed from the
  factor calculation (leave-one-week-out), otherwise the factor would be
  built from the very week it is tested on.
"""

import numpy as np
import pandas as pd

from shared_traffic_utils import (
    load_intersection,
    aggregate_daily,
    MOTOR_SECTIONS,
    ALL_MOTOR_VEHICLES,
    DIRECTIONS,
    VEHICLE_MOVEMENTS,
)

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
METHODS = ("Annual", "Monthly")


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def prepare_daily(files, section):
    """Load traffic data and aggregate complete days."""
    if section == ALL_MOTOR_VEHICLES:
        sections = MOTOR_SECTIONS
    else:
        sections = [section]

    df = load_intersection(files, sections)
    daily = aggregate_daily(df)

    if section == ALL_MOTOR_VEHICLES:
        # A day is usable only if all motor sections are complete that day,
        # otherwise the sum would silently miss a vehicle class.
        n_sections = daily.groupby("Date")["Section"].nunique()
        full_dates = n_sections[n_sections == len(sections)].index
        daily = daily[daily["Date"].isin(full_dates)]

        traffic_columns = daily.select_dtypes(include="number").columns
        daily = (
            daily.groupby("Date")[traffic_columns]
            .sum(min_count=1)
            .reset_index()
        )

    return daily


def _split_column(col):
    """'Southbound_U-Turn' -> ('Southbound', 'U-Turn')."""
    direction, _, movement = col.partition("_")
    return direction, movement


def _traffic_columns(daily, section, direction, movement):
    is_ped = section == "Pedestrians"
    directions = DIRECTIONS if direction == "Total" else [direction]

    columns = []
    for col in daily.columns:
        d, m = _split_column(col)
        if d not in directions:
            continue
        if not is_ped and m not in VEHICLE_MOVEMENTS:
            continue
        if movement != "Total" and m != movement:
            continue
        columns.append(col)
    return columns


def available_movements(daily, section):
    """Movement names present in the data (works for pedestrians too)."""
    found = {
        _split_column(c)[1]
        for c in _traffic_columns(daily, section, "Total", "Total")
    }
    if section == "Pedestrians":
        return sorted(found)
    return [m for m in VEHICLE_MOVEMENTS if m in found]


def select_daily_volume(daily, section, direction="Total", movement="Total"):
    """Select daily traffic volume by direction and movement."""
    columns = _traffic_columns(daily, section, direction, movement)
    if not columns:
        raise ValueError(
            f"No columns for {section} / {direction} / {movement}"
        )

    result = daily[["Date"]].copy()
    result["Volume"] = daily[columns].sum(axis=1, min_count=1)
    return result


def make_week_table(daily_volume, exclude_dates=None):
    """
    Organize daily traffic into Monday-Sunday weeks (rows = week start,
    columns = 0..6). Dates in exclude_dates (e.g. holidays) are blanked,
    which removes the whole week from Weekly-ADT calculations.
    """
    df = daily_volume.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if exclude_dates is not None:
        df = df[~df["Date"].isin(pd.to_datetime(list(exclude_dates)))]

    df["Weekday"] = df["Date"].dt.weekday
    df["Week"] = df["Date"] - pd.to_timedelta(df["Weekday"], unit="D")

    weekly = df.pivot(index="Week", columns="Weekday", values="Volume")
    return weekly.reindex(columns=range(7)).sort_index()


def _week_arrays(weekly):
    """Complete weeks only -> (week starts, volumes WxD, ADT W, months WxD)."""
    full = weekly.dropna(how="any")
    if full.empty:
        raise ValueError("No complete weeks available.")

    weeks = pd.DatetimeIndex(full.index)
    volumes = full.to_numpy(dtype=float)
    adt = volumes.mean(axis=1)

    offsets = np.arange(7) * np.timedelta64(1, "D")
    dates = weeks.to_numpy()[:, None] + offsets[None, :]
    months = pd.DatetimeIndex(dates.ravel()).month.to_numpy().reshape(volumes.shape)

    return weeks, volumes, adt, months


# ----------------------------------------------------------------------
# Factor tables (for display / export)
# ----------------------------------------------------------------------
def calculate_annual_factors(weekly):
    """Annual factor for each weekday, using all complete weeks."""
    weeks, X, adt, _ = _week_arrays(weekly)
    den = X.sum(axis=0)
    factor = adt.sum() / np.where(den > 0, den, np.nan)

    return pd.DataFrame({
        "Weekday": range(7),
        "Factor": factor,
        "N_Weeks": len(weeks),
    })


def calculate_monthly_factors(weekly):
    """Factor for each (month, weekday), using all complete weeks."""
    weeks, X, adt, months = _week_arrays(weekly)
    records = []

    for m in range(1, 13):
        for d in range(7):
            mask = months[:, d] == m
            den = X[mask, d].sum()
            factor = adt[mask].sum() / den if mask.any() and den > 0 else np.nan
            records.append({
                "Month": m, "Weekday": d,
                "Factor": factor, "N_Weeks": int(mask.sum()),
            })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------
def estimate_weekly_adt(weekly_A, weekly_B, method,
                        leave_one_out=False, min_train_weeks=2):
    """
    Estimate every (week, weekday) of B from factors built on A.

    Parameters
    ----------
    method : "Annual" or "Monthly"
    leave_one_out : exclude the test week from A's factors. Use True when
        A and B are the same intersection.
    min_train_weeks : minimum number of weeks behind a factor; cells with
        fewer weeks give NaN (matters for Monthly, ~4 weeks per cell).

    Returns
    -------
    pd.DataFrame, one row per (Week, Weekday) with the estimate and errors.
    """
    wa, Xa, adt_a, months_a = _week_arrays(weekly_A)
    wb, Xb, adt_b, months_b = _week_arrays(weekly_B)
    n_test = len(wb)
    cols = np.arange(7)[None, :]

    if method == "Annual":
        num = np.full((n_test, 7), adt_a.sum())
        den = np.tile(Xa.sum(axis=0), (n_test, 1))
        cnt = np.full((n_test, 7), float(len(wa)))

    elif method == "Monthly":
        m_num = np.zeros((12, 7))
        m_den = np.zeros((12, 7))
        m_cnt = np.zeros((12, 7))
        for d in range(7):
            for m in range(1, 13):
                mask = months_a[:, d] == m
                m_num[m - 1, d] = adt_a[mask].sum()
                m_den[m - 1, d] = Xa[mask, d].sum()
                m_cnt[m - 1, d] = mask.sum()

        idx = months_b - 1
        num = m_num[idx, cols]
        den = m_den[idx, cols]
        cnt = m_cnt[idx, cols]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Leave-one-week-out: remove the test week's own contribution
    if leave_one_out:
        pos = wa.get_indexer(wb)
        hit = pos >= 0
        if hit.any():
            p = pos[hit]
            num[hit] -= adt_a[p][:, None]
            den[hit] -= Xa[p]
            cnt[hit] -= 1

    valid = (cnt >= min_train_weeks) & (den > 0)
    factor = np.full((n_test, 7), np.nan)
    factor[valid] = num[valid] / den[valid]

    estimated = Xb * factor
    actual = np.repeat(adt_b[:, None], 7, axis=1)
    err = estimated - actual
    ae = np.abs(err)
    ape = np.where(actual > 0, ae / np.where(actual > 0, actual, 1) * 100, np.nan)

    return pd.DataFrame({
        "Method": method,
        "Week": np.repeat(wb.to_numpy(), 7),
        "Weekday": np.tile(np.arange(7), n_test),
        "Month": months_b.ravel(),
        "Sample": Xb.ravel(),
        "Factor": factor.ravel(),
        "Estimated": estimated.ravel(),
        "Actual": actual.ravel(),
        "N_Train": cnt.ravel(),
        "Err": err.ravel(),
        "AE": ae.ravel(),
        "APE": ape.ravel(),
    })


def calculate_annual_ape(weekly_A, weekly_B, leave_one_out=False, **kw):
    return estimate_weekly_adt(weekly_A, weekly_B, "Annual", leave_one_out, **kw)


def calculate_monthly_ape(weekly_A, weekly_B, leave_one_out=False, **kw):
    return estimate_weekly_adt(weekly_A, weekly_B, "Monthly", leave_one_out, **kw)


def align_methods(results):
    """Keep only (Week, Weekday) cases that every method could estimate,
    so the methods are compared on exactly the same samples."""
    ok = results.dropna(subset=["Estimated"])
    n_methods = results["Method"].nunique()
    n_have = ok.groupby(["Week", "Weekday"])["Method"].transform("nunique")
    return ok[n_have == n_methods].reset_index(drop=True)


def evaluate(weekly_A, weekly_B, leave_one_out=False,
             min_train_weeks=2, min_weekly_adt=0):
    """Run both methods on identical samples. Weeks with a Weekly ADT below
    min_weekly_adt are dropped (guards MAPE against near-zero actuals)."""
    results = pd.concat(
        [estimate_weekly_adt(weekly_A, weekly_B, m, leave_one_out, min_train_weeks)
         for m in METHODS],
        ignore_index=True,
    )
    results = align_methods(results)
    keep = (results["Actual"] > 0) & (results["Actual"] >= min_weekly_adt)
    return results[keep].reset_index(drop=True)
