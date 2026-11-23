"""Build rolling-window AADT error curves for traffic movements.

For each ordered reference/target pair and each movement measure, the model
tests every possible consecutive calibration window from 1 to 90 days. This
avoids making the result depend on one arbitrarily selected starting date.
"""

from itertools import permutations

import numpy as np
import pandas as pd

from config import ERROR_CURVE_MAX_DAYS
from data_loader import measure_group


def _shared_daily_data(
    reference: dict[str, pd.DataFrame],
    target: dict[str, pd.DataFrame],
    measure: str,
) -> pd.DataFrame:
    """Align one daily measure and identify uninterrupted calendar segments."""

    paired = pd.concat(
        [
            reference["daily"][measure].rename("Reference"),
            target["daily"][measure].rename("Target"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    paired = paired.sort_index()

    if paired.empty:
        return paired

    # A rolling count may not silently jump across an excluded/missing date.
    starts_new_segment = paired.index.to_series().diff().ne(pd.Timedelta(days=1))
    paired["Segment"] = starts_new_segment.cumsum().to_numpy()
    return paired


def build_error_curve(
    reference: dict[str, pd.DataFrame],
    target: dict[str, pd.DataFrame],
    measure: str,
    max_days: int = ERROR_CURVE_MAX_DAYS,
) -> pd.DataFrame:
    """Return rolling-window AADT error statistics for one movement measure.

    A separate scale factor is calculated for every possible consecutive
    window of a given length. The curve reports the distribution of the AADT
    errors across those possible starting dates.
    """

    paired = _shared_daily_data(reference, target, measure)
    if paired.empty:
        return pd.DataFrame()

    reference_aadt = paired["Reference"].mean()
    actual_target_aadt = paired["Target"].mean()
    if reference_aadt <= 0 or actual_target_aadt <= 0:
        return pd.DataFrame()

    rows: list[dict] = []
    longest_segment = int(paired.groupby("Segment").size().max())
    last_day = min(max_days, longest_segment)

    for calibration_days in range(1, last_day + 1):
        scale_factors: list[float] = []

        for _, segment in paired.groupby("Segment", sort=False):
            if len(segment) < calibration_days:
                continue

            reference_totals = segment["Reference"].rolling(
                calibration_days, min_periods=calibration_days
            ).sum()
            target_totals = segment["Target"].rolling(
                calibration_days, min_periods=calibration_days
            ).sum()

            valid = reference_totals.gt(0) & reference_totals.notna()
            factors = (target_totals.loc[valid] / reference_totals.loc[valid]).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            scale_factors.extend(factors.astype(float).tolist())

        if not scale_factors:
            continue

        factors_array = np.asarray(scale_factors, dtype=float)
        estimated_aadt = reference_aadt * factors_array
        percent_errors = (
            np.abs(estimated_aadt - actual_target_aadt)
            / actual_target_aadt
            * 100.0
        )

        rows.append(
            {
                "Measure_Group": measure_group(measure),
                "Measure": measure,
                "Calibration_Days": calibration_days,
                "Nominal_15_Minute_Observations": calibration_days * 96,
                "Calibration_Windows": len(percent_errors),
                "Shared_Valid_Dates": len(paired),
                "Reference_AADT": reference_aadt,
                "Actual_Target_AADT": actual_target_aadt,
                "Mean_Scale_Factor": factors_array.mean(),
                "Mean_Estimated_Target_AADT": estimated_aadt.mean(),
                "Mean_Error": percent_errors.mean(),
                "Median_Error": np.median(percent_errors),
                "P10_Error": np.percentile(percent_errors, 10),
                "P90_Error": np.percentile(percent_errors, 90),
                "Minimum_Error": percent_errors.min(),
                "Maximum_Error": percent_errors.max(),
            }
        )

    return pd.DataFrame(rows)


def run_all_error_curves(
    locations: dict[str, dict[str, pd.DataFrame]],
    max_days: int = ERROR_CURVE_MAX_DAYS,
) -> pd.DataFrame:
    """Evaluate all ordered site pairs, measures, and daily count lengths."""

    curves: list[pd.DataFrame] = []
    for reference_name, target_name in permutations(sorted(locations), 2):
        reference = locations[reference_name]
        target = locations[target_name]

        common_measures = reference["daily"].columns.intersection(
            target["daily"].columns
        )
        for measure in common_measures:
            curve = build_error_curve(
                reference,
                target,
                measure,
                max_days=max_days,
            )
            if curve.empty:
                continue

            curve.insert(0, "Target_Location", target_name)
            curve.insert(0, "Reference_Location", reference_name)
            curves.append(curve)

    columns = [
        "Reference_Location",
        "Target_Location",
        "Measure_Group",
        "Measure",
        "Calibration_Days",
        "Nominal_15_Minute_Observations",
        "Calibration_Windows",
        "Shared_Valid_Dates",
        "Reference_AADT",
        "Actual_Target_AADT",
        "Mean_Scale_Factor",
        "Mean_Estimated_Target_AADT",
        "Mean_Error",
        "Median_Error",
        "P10_Error",
        "P90_Error",
        "Minimum_Error",
        "Maximum_Error",
    ]
    if not curves:
        return pd.DataFrame(columns=columns)
    return pd.concat(curves, ignore_index=True)[columns]
