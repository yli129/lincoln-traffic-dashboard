"""Generate Average Daily Traffic (ADT) results for every intersection.

Run from the Project2_data folder:

    python3 analysis/generate_adt_results.py

The existing data_loader handles timestamp conversion to America/Chicago,
validates complete dates, combines the five motor-vehicle sections, and
creates daily movement totals. This script averages those valid daily totals.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CLEANED_DATA_DIR, OUTPUTS_DIR  # noqa: E402
from data_loader import load_all_locations, measure_group  # noqa: E402


def main() -> None:
    """Calculate and save ADT for every available intersection."""
    locations, quality_report = load_all_locations(CLEANED_DATA_DIR)

    daily_output = []
    adt_rows = []

    for location, results in sorted(locations.items()):
        daily = results["daily"].copy()

        if daily.empty:
            print(f"Skipping {location}: no valid daily observations.")
            continue

        # Preserve every valid daily total for inspection and later analysis.
        daily = daily.sort_index()
        daily.insert(0, "Location", location)
        daily_output.append(daily.reset_index())

        # ADT = arithmetic mean of the valid daily traffic totals.
        measures = results["daily"]
        for measure in measures.columns:
            values = pd.to_numeric(measures[measure], errors="coerce").dropna()

            adt_rows.append(
                {
                    "Location": location,
                    "Measure_Group": measure_group(measure),
                    "Measure": measure,
                    "ADT": values.mean(),
                    "Valid_Days": int(values.count()),
                    "Total_Observed_Volume": values.sum(),
                }
            )

    if not adt_rows:
        raise RuntimeError(
            "No ADT results were created. Check the cleaned data and quality report."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    adt_summary = pd.DataFrame(adt_rows).sort_values(
        ["Location", "Measure_Group", "Measure"]
    )
    daily_volumes = pd.concat(daily_output, ignore_index=True)

    adt_file = OUTPUTS_DIR / "average_daily_traffic.csv"
    daily_file = OUTPUTS_DIR / "daily_traffic_volumes.csv"
    quality_file = OUTPUTS_DIR / "adt_data_quality.csv"

    adt_summary.to_csv(adt_file, index=False)
    daily_volumes.to_csv(daily_file, index=False)
    quality_report.to_csv(quality_file, index=False)

    print("\nAVERAGE DAILY TRAFFIC (ADT)")
    print(
        adt_summary[
            ["Location", "Measure", "ADT", "Valid_Days"]
        ].to_string(index=False)
    )

    print("\nVALID DAYS BY INTERSECTION")
    print(
        quality_report[
            ["Location", "Valid_Dates", "Excluded_Dates"]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {adt_file}")
    print(f"  {daily_file}")
    print(f"  {quality_file}")


if __name__ == "__main__":
    main()
