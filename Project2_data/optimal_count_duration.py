from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_utils import load_intersection


DATA_FOLDER = Path(__file__).resolve().parent
CLEANED_FOLDER = DATA_FOLDER / "cleaned"

YEAR = 2025
TIMEZONE = "America/Chicago"
ERROR_THRESHOLD = 5.0

MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles"
]

def get_intersection_files():
    """Find and group cleaned CSV files by intersection."""

    files = sorted(CLEANED_FOLDER.glob("*_Binned_*.csv"))

    intersection_files = {}

    for file in files:
        name = file.name.split("_Binned_")[0]

        if name not in intersection_files:
            intersection_files[name] = []

        intersection_files[name].append(file)

    return intersection_files


def prepare_data(a_files, b_files):
    """Load, aggregate, and align two intersections."""

    def aggregate(files, name):
        df = load_intersection(files, MOTOR_SECTIONS)

        # Require all five valid sections at each timestamp
        counts = df.groupby("Time").agg(
            section_count=("Section", "nunique"),
            row_count=("Section", "size"),
            total=("Total_Vehicles", "sum")
        )

        valid = counts[
            (counts["section_count"] == len(MOTOR_SECTIONS))
            & (counts["row_count"] == len(MOTOR_SECTIONS))
        ]

        return valid[["total"]].rename(
            columns={"total": name}
        )

    a = aggregate(a_files, "A")
    b = aggregate(b_files, "B")

    # Match identical timestamps
    combined = a.join(b, how="inner")
    combined = combined.sort_index().reset_index()

    combined["Time"] = pd.to_datetime(
        combined["Time"],
        utc=True
    )

    # Use Chicago local dates for annual analysis
    local_time = combined["Time"].dt.tz_convert(TIMEZONE)

    combined = combined[
        local_time.dt.year == YEAR
    ].copy()

    combined["Date"] = (
        combined["Time"]
        .dt.tz_convert(TIMEZONE)
        .dt.date
    )

    # Require complete days for accurate daily totals
    def expected_intervals(date):
        start = pd.Timestamp(date).tz_localize(TIMEZONE)
        end = (
            pd.Timestamp(date) + pd.Timedelta(days=1)
        ).tz_localize(TIMEZONE)

        return int(
            (end - start) / pd.Timedelta(minutes=15)
        )

    daily_counts = combined.groupby("Date").size()

    complete_dates = [
        date for date, count in daily_counts.items()
        if count == expected_intervals(date)
    ]

    combined = combined[
        combined["Date"].isin(complete_dates)
    ].copy()

    combined = combined.sort_values("Time").reset_index(drop=True)

    if combined.empty:
        raise ValueError("No complete shared traffic days found.")

    return combined


def calculate_error_curve(combined):
    """Calculate estimation errors for N = 1 to Nmax."""

    a = combined["A"].to_numpy(dtype=float)
    b = combined["B"].to_numpy(dtype=float)

    n_max = len(combined)
    N = np.arange(1, n_max + 1)

    # Cumulative traffic counts
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)

    # Calculate scale factors for every N
    scale_factors = np.divide(
        b_cumsum,
        a_cumsum,
        out=np.full(n_max, np.nan),
        where=a_cumsum > 0
    )

    # Actual daily traffic
    daily = combined.groupby("Date")[["A", "B"]].sum()

    a_daily = daily["A"].to_numpy(dtype=float)
    b_daily = daily["B"].to_numpy(dtype=float)

    actual_aadt = b_daily.mean()

    # Estimate annual average daily traffic
    estimated_aadt = scale_factors * a_daily.mean()

    aadt_error = (
        np.abs(estimated_aadt - actual_aadt)
        / actual_aadt * 100
    )

    # Calculate daily MAE in batches to save memory
    daily_mae = np.full(n_max, np.nan)
    batch_size = 1024

    for start in range(0, n_max, batch_size):
        end = min(start + batch_size, n_max)

        factors = scale_factors[start:end]

        predicted = factors[:, None] * a_daily[None, :]

        errors = np.abs(predicted - b_daily[None, :])

        daily_mae[start:end] = errors.mean(axis=1)

    results = pd.DataFrame({
        "N": N,
        "Equivalent_Days": N / 96,
        "Scale_Factor": scale_factors,
        "Estimated_AADT": estimated_aadt,
        "AADT_Error_Pct": aadt_error,
        "Daily_MAE": daily_mae
    })

    return results


def main():
    # 1. Find intersections
    intersection_files = get_intersection_files()
    names = list(intersection_files.keys())

    if len(names) < 2:
        print("At least two intersections are required.")
        return

    print("\nAvailable intersections:")

    for i, name in enumerate(names, start=1):
        print(f"{i}. {name.replace('_', ' ')}")

    # 2. Let user select reference and target
    try:
        a_index = int(input("\nSelect reference (A): ")) - 1
        b_index = int(input("Select target (B): ")) - 1
    except ValueError:
        print("Please enter a number.")
        return

    if not (
        0 <= a_index < len(names)
        and 0 <= b_index < len(names)
    ):
        print("Invalid selection.")
        return

    if a_index == b_index:
        print("Please select two different intersections.")
        return

    a_name = names[a_index]
    b_name = names[b_index]

    a_files = intersection_files[a_name]
    b_files = intersection_files[b_name]

    print(f"\nReference: {a_name}")
    print(f"Target: {b_name}")

    # 3. Prepare data
    combined = prepare_data(a_files, b_files)

    print("Shared observations:", len(combined))
    print("Complete days:", combined["Date"].nunique())

    # 4. Calculate errors
    results = calculate_error_curve(combined)

    # 5. Find the first N meeting the threshold
    acceptable = results[
        results["AADT_Error_Pct"] <= ERROR_THRESHOLD
    ]

    if not acceptable.empty:
        first = acceptable.iloc[0]

        print(f"\nFirst result below {ERROR_THRESHOLD}%:")
        print(f"N = {int(first['N'])}")
        print(f"AADT error = {first['AADT_Error_Pct']:.2f}%")
        print(f"Equivalent days = {first['Equivalent_Days']:.2f}")
    else:
        print("\nNo result meets the error threshold.")

    # 6. Save results
    safe_a = re.sub(r"\W+", "_", a_name)
    safe_b = re.sub(r"\W+", "_", b_name)
    output_name = f"error_curve_{safe_a}_to_{safe_b}"

    results.to_csv(
        DATA_FOLDER / f"{output_name}.csv",
        index=False
    )

    # 7. Plot AADT Error Curve
    plt.figure(figsize=(11, 5))

    plt.plot(
        results["N"],
        results["AADT_Error_Pct"],
        linewidth=1,
        label="AADT Error"
    )

    plt.axhline(
        ERROR_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"{ERROR_THRESHOLD}% threshold"
    )

    plt.xlabel("Number of Observations (N)")
    plt.ylabel("AADT Percent Error (%)")
    plt.title("AADT Error vs. Number of Observations")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        DATA_FOLDER / f"{output_name}_aadt.png",
        dpi=200
    )

    # 8. Plot Daily MAE Curve
    plt.figure(figsize=(11, 5))

    plt.plot(
        results["N"],
        results["Daily_MAE"],
        linewidth=1
    )

    plt.xlabel("Number of Observations (N)")
    plt.ylabel("Daily MAE (vehicles/day)")
    plt.title("Daily MAE vs. Number of Observations")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        DATA_FOLDER / f"{output_name}_mae.png",
        dpi=200
    )

    plt.show()

    print("\nResults and charts saved successfully.")


if __name__ == "__main__":
    main()
