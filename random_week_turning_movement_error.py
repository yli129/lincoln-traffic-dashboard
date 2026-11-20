"""
RANDOM-WEEK TURNING-MOVEMENT MONTHLY ESTIMATOR
================================================

Purpose:
    Select a month and intersection, randomly select one complete 7-day
    period from that month, and use that week's RIGHT-turn and LEFT-turn
    totals to estimate the month's total right and left turns.

    The estimate is compared with the actual monthly totals. The program
    also repeats the random-week experiment many times to show how much
    the error can vary depending on which week is selected.

Input:
    - Project2_data.zip and/or *Binned_*.csv files in the same folder.
    - Supports the older CSV format with columns named Right and Left.
    - Supports the newer Binned CSV "Lights" section, with direction/movement
      columns such as North_Right and South_Left.

Requirements:
    pip install pandas matplotlib

Outputs:
    - random_week_turn_estimates.csv
    - random_week_simulation_results.csv
    - random_week_error_summary.csv
    - random_week_error_distribution.png
    - random_week_error_curve.png

Important:
    A valid month must have one recorded daily total for every calendar day.
    A sampled week must contain seven consecutive, complete days.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, date, timedelta
import csv
import zipfile
import random
import calendar

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ZIP_FILE = BASE_DIR / "Project2_data.zip"
BINNED_PATTERN = "*Binned_*.csv"

# Number of random 7-day samples used to study the range of possible errors.
# The program also prints one randomly selected sample.
SIMULATION_TRIALS = 1000

# Set to an integer (e.g. 42) to make the random results repeatable.
# Set to None to get a different random selection each run.
RANDOM_SEED = 42


# ------------------------------------------------------------
# CSV PARSING
# ------------------------------------------------------------

def to_number(value):
    """Convert a traffic count to a number; treat blanks as zero."""
    try:
        value = str(value).strip().replace(",", "")
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0


def find_location(rows, fallback_name):
    for row in rows:
        if len(row) >= 2 and row[0].strip().lower() == "location":
            return row[1].strip()
    return Path(fallback_name).stem


def parse_traffic_csv(csv_text, source_name):
    """
    Return (intersection_name, dataframe) with:
        Time, Right, Left

    The parser supports both the older movement-column layout and the
    newer Binned export with a Lights section.
    """
    rows = list(csv.reader(csv_text.splitlines()))
    location = find_location(rows, source_name)

    # Newer Binned format: find the Lights section and its two header rows.
    lights_index = next(
        (i for i, row in enumerate(rows)
         if row and row[0].strip().lower() == "lights"),
        None
    )

    if lights_index is not None and len(rows) > lights_index + 3:
        direction_row = rows[lights_index + 2]
        movement_row = rows[lights_index + 3]

        column_types = []
        current_direction = ""

        for col_index in range(1, len(movement_row)):
            if col_index < len(direction_row):
                direction_text = direction_row[col_index].strip()
                if direction_text:
                    current_direction = direction_text

            movement = movement_row[col_index].strip().lower()

            if movement == "right":
                column_types.append((col_index, "Right"))
            elif movement == "left":
                column_types.append((col_index, "Left"))

        records = []
        for row in rows[lights_index + 4:]:
            if not row or not row[0].strip():
                continue

            try:
                timestamp = datetime.strptime(
                    row[0].strip()[:19], "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                continue

            right_total = sum(
                to_number(row[i]) for i, kind in column_types
                if kind == "Right" and i < len(row)
            )
            left_total = sum(
                to_number(row[i]) for i, kind in column_types
                if kind == "Left" and i < len(row)
            )

            records.append((timestamp, right_total, left_total))

        if records:
            return location, pd.DataFrame(
                records, columns=["Time", "Right", "Left"]
            )

    # Older format: one or more columns named Right and Left.
    header_index = None
    for i, row in enumerate(rows):
        if (
            row
            and row[0].strip().lower() == "start time"
            and any(cell.strip().lower() == "right" for cell in row)
            and any(cell.strip().lower() == "left" for cell in row)
        ):
            header_index = i
            break

    if header_index is None:
        return location, pd.DataFrame(columns=["Time", "Right", "Left"])

    headers = rows[header_index]
    right_columns = [
        i for i, name in enumerate(headers)
        if name.strip().lower() == "right"
    ]
    left_columns = [
        i for i, name in enumerate(headers)
        if name.strip().lower() == "left"
    ]

    records = []
    for row in rows[header_index + 1:]:
        if not row or not row[0].strip():
            continue

        try:
            timestamp = datetime.strptime(
                row[0].strip()[:19], "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            continue

        right_total = sum(
            to_number(row[i]) for i in right_columns if i < len(row)
        )
        left_total = sum(
            to_number(row[i]) for i in left_columns if i < len(row)
        )

        records.append((timestamp, right_total, left_total))

    return location, pd.DataFrame(
        records, columns=["Time", "Right", "Left"]
    )


def load_intersections():
    """Load ZIP contents and standalone Binned CSVs, grouped by intersection."""
    sources = []

    if ZIP_FILE.exists():
        with zipfile.ZipFile(ZIP_FILE, "r") as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    try:
                        content = archive.read(name).decode("utf-8-sig")
                        sources.append((name, content))
                    except Exception as exc:
                        print(f"Could not read {name} in ZIP: {exc}")

    for path in sorted(BASE_DIR.glob(BINNED_PATTERN)):
        try:
            sources.append((path.name, path.read_text(encoding="utf-8-sig")))
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")

    if not sources:
        raise FileNotFoundError(
            "No input data found. Put Project2_data.zip and/or "
            "*Binned_*.csv files beside this Python script."
        )

    by_intersection = defaultdict(list)

    for source_name, content in sources:
        location, frame = parse_traffic_csv(content, source_name)
        if frame.empty:
            continue

        frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
        frame = frame.dropna(subset=["Time"])
        by_intersection[location].append(frame)

    intersections = {}

    for location, frames in by_intersection.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("Time")

        # Remove duplicate timestamps that may occur when overlapping exports
        # or duplicate copies of the same CSV are present.
        combined = combined.drop_duplicates(subset=["Time"], keep="last")

        combined["Date"] = combined["Time"].dt.normalize()

        daily = (
            combined.groupby("Date")[["Right", "Left"]]
            .sum()
            .sort_index()
        )
        daily.index = pd.to_datetime(daily.index)
        intersections[location] = daily

    if not intersections:
        raise ValueError(
            "Files were found, but no usable Right/Left traffic data "
            "could be parsed."
        )

    return intersections


# ------------------------------------------------------------
# MONTH AND RANDOM-WEEK ANALYSIS
# ------------------------------------------------------------

def get_complete_month(daily, year, month):
    """Return the month's daily data only if every calendar date is present."""
    number_of_days = calendar.monthrange(year, month)[1]
    first_day = pd.Timestamp(year=year, month=month, day=1)
    expected_dates = pd.date_range(first_day, periods=number_of_days, freq="D")

    month_data = daily.reindex(expected_dates)

    missing_dates = month_data[["Right", "Left"]].isna().any(axis=1)
    if missing_dates.any():
        missing = [
            timestamp.strftime("%Y-%m-%d")
            for timestamp in month_data.index[missing_dates]
        ]
        raise ValueError(
            "This month does not have a recorded daily total for every date.\n"
            "Missing dates: " + ", ".join(missing) + "\n"
            "A full-month actual total is needed to calculate a valid error."
        )

    return month_data


def possible_week_starts(month_data):
    """List every start date for a complete 7-consecutive-day sample."""
    starts = []
    for i in range(len(month_data) - 6):
        week = month_data.iloc[i:i + 7]
        if len(week) == 7 and not week[["Right", "Left"]].isna().any().any():
            starts.append(month_data.index[i])
    return starts


def estimate_from_week(month_data, week_start):
    """Estimate the monthly right/left totals from one 7-day sample."""
    week_end = week_start + pd.Timedelta(days=6)
    week = month_data.loc[week_start:week_end]

    days_in_month = len(month_data)
    actual_right = month_data["Right"].sum()
    actual_left = month_data["Left"].sum()

    estimated_right = week["Right"].mean() * days_in_month
    estimated_left = week["Left"].mean() * days_in_month

    right_signed_error = (
        (estimated_right - actual_right) / actual_right * 100
        if actual_right else float("nan")
    )
    left_signed_error = (
        (estimated_left - actual_left) / actual_left * 100
        if actual_left else float("nan")
    )

    return {
        "Week_Start": week_start.date().isoformat(),
        "Week_End": week_end.date().isoformat(),
        "Sample_Days": len(week),
        "Actual_Monthly_Right": actual_right,
        "Estimated_Monthly_Right": estimated_right,
        "Right_Signed_Error_Percent": right_signed_error,
        "Right_Absolute_Error_Percent": abs(right_signed_error),
        "Actual_Monthly_Left": actual_left,
        "Estimated_Monthly_Left": estimated_left,
        "Left_Signed_Error_Percent": left_signed_error,
        "Left_Absolute_Error_Percent": abs(left_signed_error),
    }


def run_analysis():
    rng = random.Random(RANDOM_SEED)
    intersections = load_intersections()

    print("\nAVAILABLE INTERSECTIONS")
    print("=" * 65)
    locations = list(intersections.keys())
    for i, location in enumerate(locations, start=1):
        print(f"{i}. {location}")

    while True:
        try:
            choice = int(input("\nSelect an intersection by number: "))
            if 1 <= choice <= len(locations):
                break
        except ValueError:
            pass
        print("Please enter one of the listed numbers.")

    location = locations[choice - 1]
    daily = intersections[location]

    available_years = sorted(set(daily.index.year))
    print("\nYears available:", ", ".join(map(str, available_years)))

    year = int(input("Enter the year to analyze (e.g. 2025): "))
    month = int(input("Enter the month number (1=January, 12=December): "))

    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")

    month_data = get_complete_month(daily, year, month)
    starts = possible_week_starts(month_data)

    if not starts:
        raise ValueError("No complete 7-day sample is available in this month.")

    actual_right = month_data["Right"].sum()
    actual_left = month_data["Left"].sum()

    # Pick one random week and show the estimate requested by the user.
    selected_start = rng.choice(starts)
    selected_result = estimate_from_week(month_data, selected_start)

    print("\n" + "=" * 72)
    print("ONE RANDOM WEEK — MONTHLY TURNING-MOVEMENT ESTIMATE")
    print("=" * 72)
    print(f"Intersection: {location}")
    print(f"Month: {calendar.month_name[month]} {year}")
    print(
        f"Selected week: {selected_result['Week_Start']} "
        f"through {selected_result['Week_End']}"
    )

    print("\nRIGHT TURNS")
    print(f"  Actual monthly total:    {actual_right:,.0f}")
    print(
        "  Estimated monthly total: "
        f"{selected_result['Estimated_Monthly_Right']:,.0f}"
    )
    print(
        "  Percentage error:        "
        f"{selected_result['Right_Signed_Error_Percent']:+.2f}%"
    )
    print(
        "  Absolute percentage error: "
        f"{selected_result['Right_Absolute_Error_Percent']:.2f}%"
    )

    print("\nLEFT TURNS")
    print(f"  Actual monthly total:    {actual_left:,.0f}")
    print(
        "  Estimated monthly total: "
        f"{selected_result['Estimated_Monthly_Left']:,.0f}"
    )
    print(
        "  Percentage error:        "
        f"{selected_result['Left_Signed_Error_Percent']:+.2f}%"
    )
    print(
        "  Absolute percentage error: "
        f"{selected_result['Left_Absolute_Error_Percent']:.2f}%"
    )

    # Repeat the experiment using random valid weeks.
    simulation_rows = []
    for trial in range(1, SIMULATION_TRIALS + 1):
        start = rng.choice(starts)
        result = estimate_from_week(month_data, start)
        result["Trial"] = trial
        simulation_rows.append(result)

    simulations = pd.DataFrame(simulation_rows)

    summary = []
    for movement in ("Right", "Left"):
        errors = simulations[f"{movement}_Absolute_Error_Percent"]
        signed = simulations[f"{movement}_Signed_Error_Percent"]
        summary.append({
            "Movement": movement,
            "Trials": len(simulations),
            "Mean_Absolute_Error_Percent": errors.mean(),
            "Median_Absolute_Error_Percent": errors.median(),
            "Best_Absolute_Error_Percent": errors.min(),
            "Worst_Absolute_Error_Percent": errors.max(),
            "Mean_Signed_Error_Percent": signed.mean(),
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 72)
    print(f"RANDOM-WEEK SIMULATION SUMMARY ({SIMULATION_TRIALS:,} trials)")
    print("=" * 72)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Save detailed results.
    selected_df = pd.DataFrame([selected_result])
    selected_df.insert(0, "Intersection", location)
    selected_df.insert(1, "Year", year)
    selected_df.insert(2, "Month", calendar.month_name[month])
    selected_df.to_csv(BASE_DIR / "random_week_turn_estimates.csv", index=False)

    simulations.insert(0, "Intersection", location)
    simulations.insert(1, "Year", year)
    simulations.insert(2, "Month", calendar.month_name[month])
    simulations.to_csv(
        BASE_DIR / "random_week_simulation_results.csv", index=False
    )

    summary_df.insert(0, "Intersection", location)
    summary_df.insert(1, "Year", year)
    summary_df.insert(2, "Month", calendar.month_name[month])
    summary_df.to_csv(
        BASE_DIR / "random_week_error_summary.csv", index=False
    )

    # Graph 1: distribution of absolute errors across random weeks.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(
        simulations["Right_Absolute_Error_Percent"],
        bins=25,
        edgecolor="black"
    )
    axes[0].axvline(
        simulations["Right_Absolute_Error_Percent"].mean(),
        linestyle="--",
        label="Mean absolute error"
    )
    axes[0].set_title("Right-turn monthly estimate error")
    axes[0].set_xlabel("Absolute percentage error (%)")
    axes[0].set_ylabel("Number of random weeks")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(
        simulations["Left_Absolute_Error_Percent"],
        bins=25,
        edgecolor="black"
    )
    axes[1].axvline(
        simulations["Left_Absolute_Error_Percent"].mean(),
        linestyle="--",
        label="Mean absolute error"
    )
    axes[1].set_title("Left-turn monthly estimate error")
    axes[1].set_xlabel("Absolute percentage error (%)")
    axes[1].set_ylabel("Number of random weeks")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Random 7-Day Samples — {calendar.month_name[month]} {year}\n"
        f"{location}"
    )
    fig.tight_layout()
    fig.savefig(
        BASE_DIR / "random_week_error_distribution.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Graph 2: running mean absolute error as more random trials are added.
    fig, ax = plt.subplots(figsize=(10, 6))

    trial_numbers = range(1, len(simulations) + 1)
    right_curve = (
        simulations["Right_Absolute_Error_Percent"]
        .expanding().mean()
    )
    left_curve = (
        simulations["Left_Absolute_Error_Percent"]
        .expanding().mean()
    )

    ax.plot(trial_numbers, right_curve, label="Right turns")
    ax.plot(trial_numbers, left_curve, label="Left turns")
    ax.set_title("Running Mean Absolute Error Across Random Weeks")
    ax.set_xlabel("Number of random-week trials")
    ax.set_ylabel("Running mean absolute percentage error (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        BASE_DIR / "random_week_error_curve.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nFILES CREATED")
    print("- random_week_turn_estimates.csv")
    print("- random_week_simulation_results.csv")
    print("- random_week_error_summary.csv")
    print("- random_week_error_distribution.png")
    print("- random_week_error_curve.png")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as error:
        print("\nERROR:", error)
        print(
            "\nCheck that the input data is beside this script, "
            "the selected month has a complete record for every day, "
            "and pandas/matplotlib are installed."
        )
    input("\nPress Enter to close...")
