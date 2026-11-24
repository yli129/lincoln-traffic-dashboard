
import csv
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILES AND SETTINGS
# ============================================================

BASE_FOLDER = Path(__file__).resolve().parent

INTERSECTIONS = {
    "40th Street & Normal Boulevard":
        "40th_Street_and_Normal_Boulevard_Binned_20261102145726.csv",

    "South Street & Normal Boulevard":
        "South_Street_and_Normal_Boulevard_Binned_20260909204441.csv",
}

MOVEMENTS = ["Right", "Through", "Left", "U-Turn"]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize(value):
    """Normalize column names for matching."""
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def identify_movement(column):
    """Identify a turning movement from a column name."""
    name = normalize(column)

    if "u turn" in name or "uturn" in name:
        return "U-Turn"

    if "right" in name:
        return "Right"

    if "thru" in name or "through" in name:
        return "Through"

    if "left" in name:
        return "Left"

    return None


def parse_number(value):
    """Convert a CSV value into a number."""
    try:
        value = str(value).strip().replace(",", "")
        if value == "" or value.lower() in ("nan", "none"):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def parse_datetime(value):
    """Parse a timestamp while preserving its local calendar date."""
    value = str(value).strip()

    if not value:
        return None

    try:
        return pd.to_datetime(value, errors="raise").tz_localize(None)
    except (ValueError, TypeError, AttributeError):
        try:
            return pd.to_datetime(value[:19], errors="raise")
        except Exception:
            return None


# ============================================================
# CSV READER
# Supports both Time/Section files and section-based exports.
# ============================================================

def read_csv_rows(path):
    """Read CSV rows without assuming every row has equal length."""

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(10000)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample, delimiters=",;\t"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(file, dialect)

        return [
            [cell.strip() for cell in row]
            for row in reader
            if row and any(cell.strip() for cell in row)
        ]


def extract_long_format(rows):
    """
    Read files with a Time,Section header.
    """

    header_index = None

    for i, row in enumerate(rows):
        if len(row) >= 2:
            first = normalize(row[0])
            second = normalize(row[1])

            if first in ("time", "start time") and second == "section":
                header_index = i
                break

    if header_index is None:
        return None

    header = rows[header_index]
    records = []

    for row in rows[header_index + 1:]:
        if len(row) != len(header):
            continue

        record = dict(zip(header, row))

        section_col = next(
            (c for c in header if normalize(c) == "section"),
            None
        )

        if section_col and normalize(record.get(section_col, "")) != "lights":
            continue

        records.append(record)

    if not records:
        return None

    return pd.DataFrame(records)


def extract_section_format(rows):
    """
    Read section-based exports.

    Looks for the Lights section, then locates the actual
    timestamp and movement headers inside that section.
    """

    records = []
    in_lights = False
    header = None
    movement_indices = {}
    time_index = None

    for row in rows:

        first = normalize(row[0]) if row else ""

        # Detect traffic section boundaries.
        section_names = {
            "lights",
            "single unit trucks",
            "articulated trucks",
            "buses",
            "motorized vehicles",
            "bicycles",
            "pedestrians"
        }

        if first in section_names and len(row) <= 3:
            in_lights = first == "lights"
            header = None
            continue

        if not in_lights:
            continue

        # Locate a header containing a timestamp column.
        possible_time = None

        for i, cell in enumerate(row):
            if normalize(cell) in ("start time", "time", "timestamp"):
                possible_time = i
                break

        if possible_time is not None:

            found_movements = defaultdict(list)

            for i, cell in enumerate(row):
                movement = identify_movement(cell)

                if movement:
                    found_movements[movement].append(i)

            if found_movements:
                header = row
                time_index = possible_time
                movement_indices = dict(found_movements)
                continue

        # Ignore metadata and rows before the table header.
        if header is None:
            continue

        if len(row) <= time_index:
            continue

        timestamp = parse_datetime(row[time_index])

        if timestamp is None:
            continue

        record = {
            "Time": timestamp
        }

        for movement in MOVEMENTS:
            total = 0.0

            for index in movement_indices.get(movement, []):
                if index < len(row):
                    total += parse_number(row[index])

            record[movement] = total

        records.append(record)

    if not records:
        return None

    return pd.DataFrame(records)


# ============================================================
# LOAD AND CLEAN INTERSECTION DATA
# ============================================================

def load_intersection(name, filename):

    path = BASE_FOLDER / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find file:\n{path}"
        )

    print(f"\nReading: {filename}")

    rows = read_csv_rows(path)

    # First try the older Time/Section format.
    data = extract_long_format(rows)

    if data is not None:

        time_col = next(
            c for c in data.columns
            if normalize(c) in ("time", "start time")
        )

        data["Time"] = data[time_col].apply(parse_datetime)

        for movement in MOVEMENTS:
            matching_columns = [
                c for c in data.columns
                if identify_movement(c) == movement
            ]

            if matching_columns:
                data[movement] = sum(
                    data[c].apply(parse_number)
                    for c in matching_columns
                )
            else:
                data[movement] = 0.0

    else:
        # Try the newer section-based format.
        data = extract_section_format(rows)

    if data is None or data.empty:
        raise ValueError(
            "Could not identify the Lights section and its "
            "timestamp/movement columns. Check the CSV layout."
        )

    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data = data.dropna(subset=["Time"]).copy()

    for movement in MOVEMENTS:
        data[movement] = pd.to_numeric(
            data[movement], errors="coerce"
        ).fillna(0)

    data["Total"] = data[MOVEMENTS].sum(axis=1)

    # Aggregate all 15-minute observations into daily totals.
    data["Date"] = data["Time"].dt.normalize()

    daily = (
        data.groupby("Date")[MOVEMENTS + ["Total"]]
        .sum()
        .sort_index()
    )

    daily.index = pd.to_datetime(daily.index)

    print(f"Loaded {len(data):,} traffic observations.")
    print(f"Daily records: {len(daily):,}")
    print(f"Date range: {daily.index.min().date()} "
          f"through {daily.index.max().date()}")

    return daily


# ============================================================
# FIND COMPLETE MONDAY-SUNDAY WEEKS
# ============================================================

def get_complete_weeks(daily):

    complete_weeks = []

    for date in daily.index:

        if date.weekday() != 0:
            continue

        week_dates = pd.date_range(
            start=date,
            periods=7,
            freq="D"
        )

        if all(day in daily.index for day in week_dates):
            complete_weeks.append(week_dates)

    return complete_weeks


# ============================================================
# ONE-DAY-TO-ONE-WEEK ERROR ANALYSIS
# ============================================================

def analyze_weekday_estimates(daily):

    weeks = get_complete_weeks(daily)

    if not weeks:
        raise ValueError(
            "No complete Monday-Sunday weeks were found."
        )

    print(f"\nComplete weeks available: {len(weeks)}")

    results = []

    for week_number, week_dates in enumerate(weeks, start=1):

        week_data = daily.loc[week_dates]

        actual_week = week_data.sum()

        for weekday_number, weekday in enumerate(WEEKDAYS):

            selected_date = week_dates[weekday_number]
            selected_day = daily.loc[selected_date]

            for movement in MOVEMENTS + ["Total"]:

                actual = float(actual_week[movement])
                sample_day = float(selected_day[movement])

                # Estimate weekly traffic using one day's count.
                estimated = sample_day * 7

                error = estimated - actual

                if actual != 0:
                    percent_error = (error / actual) * 100
                    absolute_percent_error = abs(percent_error)
                else:
                    percent_error = float("nan")
                    absolute_percent_error = float("nan")

                results.append({
                    "Week": week_number,
                    "Week Starting": week_dates[0],
                    "Weekday": weekday,
                    "Movement": movement,
                    "Sample Day Count": sample_day,
                    "Actual Weekly Total": actual,
                    "Estimated Weekly Total": estimated,
                    "Error (Vehicles)": error,
                    "Percent Error": percent_error,
                    "Absolute Percent Error": absolute_percent_error
                })

    return pd.DataFrame(results)


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def summarize_results(results):

    summary = (
        results.groupby(["Movement", "Weekday"])
        .agg(
            Mean_Actual_Weekly=("Actual Weekly Total", "mean"),
            Mean_Estimated_Weekly=("Estimated Weekly Total", "mean"),
            Mean_Error_Vehicles=("Error (Vehicles)", "mean"),
            Mean_Percent_Error=("Percent Error", "mean"),
            Mean_Absolute_Percent_Error=(
                "Absolute Percent Error", "mean"
            )
        )
        .reset_index()
    )

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(name, results, summary):

    print("\n" + "=" * 85)
    print(name.upper())
    print("=" * 85)

    print("\nAVERAGE ABSOLUTE PERCENTAGE ERROR")
    print("Lower values mean better weekly estimates.\n")

    pivot = summary.pivot(
        index="Movement",
        columns="Weekday",
        values="Mean_Absolute_Percent_Error"
    )

    pivot = pivot.reindex(
        index=MOVEMENTS + ["Total"],
        columns=WEEKDAYS
    )

    print(pivot.round(2).to_string())

    print("\nAVERAGE SIGNED PERCENTAGE ERROR")
    print("Positive = overestimate | Negative = underestimate\n")

    signed = summary.pivot(
        index="Movement",
        columns="Weekday",
        values="Mean_Percent_Error"
    )

    signed = signed.reindex(
        index=MOVEMENTS + ["Total"],
        columns=WEEKDAYS
    )

    print(signed.round(2).to_string())

    print("\nAVERAGE ERROR IN VEHICLES\n")

    vehicles = summary.pivot(
        index="Movement",
        columns="Weekday",
        values="Mean_Error_Vehicles"
    )

    vehicles = vehicles.reindex(
        index=MOVEMENTS + ["Total"],
        columns=WEEKDAYS
    )

    print(vehicles.round(1).to_string())


# ============================================================
# GRAPHS
# ============================================================

def create_graphs(name, daily, summary):

    # --------------------------------------------------------
    # GRAPH 1: Estimation error by weekday
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 7))

    for movement in MOVEMENTS + ["Total"]:

        subset = summary[
            summary["Movement"] == movement
        ].set_index("Weekday").reindex(WEEKDAYS)

        ax.plot(
            WEEKDAYS,
            subset["Mean_Absolute_Percent_Error"],
            marker="o",
            linewidth=2,
            label=movement
        )

    ax.set_title(
        f"{name}\nOne-Day-to-One-Week Absolute Percentage Error"
    )

    ax.set_xlabel("Day Used to Estimate the Week")
    ax.set_ylabel("Mean Absolute Percentage Error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Movement")
    plt.xticks(rotation=25)
    plt.tight_layout()

    # --------------------------------------------------------
    # GRAPH 2: Total traffic estimation error
    # --------------------------------------------------------

    total_summary = summary[
        summary["Movement"] == "Total"
    ].set_index("Weekday").reindex(WEEKDAYS)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        WEEKDAYS,
        total_summary["Mean_Absolute_Percent_Error"]
    )

    ax.set_title(
        f"{name}\nWeekly Total Estimation Error by Sample Day"
    )

    ax.set_xlabel("Sample Day")
    ax.set_ylabel("Mean Absolute Percentage Error (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25)
    plt.tight_layout()

    # --------------------------------------------------------
    # GRAPH 3: Movement composition pie chart
    # --------------------------------------------------------

    movement_totals = daily[MOVEMENTS].sum()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        movement_totals,
        labels=MOVEMENTS,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        f"{name}\nVehicle Movement Composition"
    )

    plt.tight_layout()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("ONE-DAY-TO-ONE-WEEK TRAFFIC ANALYSIS")
    print("=" * 75)

    successful_intersections = 0

    for name, filename in INTERSECTIONS.items():

        print("\n" + "=" * 75)
        print(f"LOADING: {name}")
        print("=" * 75)

        try:

            daily = load_intersection(name, filename)

            results = analyze_weekday_estimates(daily)

            summary = summarize_results(results)

            print_results(name, results, summary)

            create_graphs(name, daily, summary)

            successful_intersections += 1

        except Exception as error:

            print(f"\nERROR analyzing {name}:")
            print(error)

    print("\n" + "=" * 75)
    print("TRAFFIC ANALYSIS FINISHED")
    print("=" * 75)

    if successful_intersections == 0:
        print("No intersections could be analyzed.")
    else:
        print(
            f"Successfully analyzed "
            f"{successful_intersections} intersection(s)."
        )

    print("\nAll results are displayed in Python and Matplotlib.")
    print("No output files were created.")

    plt.show()


if __name__ == "__main__":
    main()
