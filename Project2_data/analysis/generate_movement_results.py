"""Generate all direction- and turn-specific AADT result files.

Run from the Project2_data folder:

    python3 analysis/generate_movement_results.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aadt_estimation import run_all_error_curves  # noqa: E402
from config import CLEANED_DATA_DIR, OUTPUTS_DIR  # noqa: E402
from data_loader import load_all_locations  # noqa: E402


def make_threshold_summary(error_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize first 5%/10% crossings and the best tested point."""

    group_columns = [
        "Reference_Location",
        "Target_Location",
        "Measure_Group",
        "Measure",
    ]
    rows = []
    for keys, curve in error_curves.groupby(group_columns, sort=True):
        curve = curve.sort_values("Calibration_Days")
        row = dict(zip(group_columns, keys))

        for threshold in (5, 10):
            reached = curve.loc[curve["Mean_Error"].le(threshold)]
            days = int(reached.iloc[0]["Calibration_Days"]) if not reached.empty else np.nan
            row[f"Days_to_{threshold}pct"] = days
            row[f"Observations_to_{threshold}pct"] = (
                int(days * 96) if pd.notna(days) else np.nan
            )

        best = curve.loc[curve["Mean_Error"].idxmin()]
        row["Best_Mean_Error"] = best["Mean_Error"]
        row["Best_Calibration_Days"] = int(best["Calibration_Days"])
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Read cleaned data, build error curves, and save reproducible outputs."""

    locations, quality_report = load_all_locations(CLEANED_DATA_DIR)
    error_curves = run_all_error_curves(locations)
    threshold_summary = make_threshold_summary(error_curves)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    curves_file = OUTPUTS_DIR / "movement_aadt_error_curves.csv"
    thresholds_file = OUTPUTS_DIR / "movement_aadt_threshold_summary.csv"
    quality_file = OUTPUTS_DIR / "movement_data_quality.csv"
    error_curves.to_csv(curves_file, index=False)
    threshold_summary.to_csv(thresholds_file, index=False)
    quality_report.to_csv(quality_file, index=False)

    print("\nValid dates by intersection")
    print(
        quality_report[
            ["Location", "Valid_Dates", "Excluded_Dates"]
        ].to_string(index=False)
    )
    print(f"\nSaved {len(error_curves):,} curve points to {curves_file}")
    print(f"Saved threshold summary to {thresholds_file}")
    print(f"Saved data-quality summary to {quality_file}")


if __name__ == "__main__":
    main()
