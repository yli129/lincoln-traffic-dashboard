"""Estimate annual traffic at Location B from Location A."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FOLDER = Path(__file__).parent
CLEANED_FOLDER = DATA_FOLDER / "cleaned"

LOCATION_A = "40th Street and South Street"
LOCATION_B = "South Street and Normal Boulevard"

# Five motor-vehicle sections. Bicycles and pedestrians are excluded.
VEHICLE_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]

# Right, Thru, Left, and U-Turn columns for four approaches.
VEHICLE_COLUMNS = [
    "1", "2", "3", "4",
    "7", "8", "9", "10",
    "13", "14", "15", "16",
    "19", "20", "21", "22",
]

# Number of 15-minute observations used from Location B.
DURATIONS = {
    "15 minutes": 1,
    "1 hour": 4,
    "1 day": 96,
    "2 days": 96 * 2,
    "7 days": 96 * 7,
    "14 days": 96 * 14,
    "30 days": 96 * 30,
}


def load_traffic():
    """Read all cleaned CSVs and calculate 15-minute motor traffic."""

    all_intervals = []

    for file in sorted(CLEANED_FOLDER.glob("*_Binned_*.csv")):
        data = pd.read_csv(file)

        location = (
            file.name
            .split("_Binned_", 1)[0]
            .replace("_", " ")
        )

        # Keep valid rows from the five motor-vehicle sections.
        data = data.loc[
            data["Section"].isin(VEHICLE_SECTIONS)
            & (data["Quality_Flag"] == "Valid")
        ].copy()

        # Change count columns into numbers.
        data[VEHICLE_COLUMNS] = data[VEHICLE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Total movement count for one section and one time point.
        data["Section_Total"] = data[VEHICLE_COLUMNS].sum(
            axis=1,
            min_count=1,
        )

        # Combine the five motor-vehicle sections.
        interval = (
            data.groupby("Time")
            .agg(
                Traffic=("Section_Total", "sum"),
                Sections=("Section", "nunique"),
            )
            .reset_index()
        )

        # Do not use an interval unless all five sections are present.
        interval.loc[
            interval["Sections"] != len(VEHICLE_SECTIONS),
            "Traffic",
        ] = np.nan

        interval["Location"] = location
        all_intervals.append(
            interval[["Location", "Time", "Traffic"]]
        )

    if not all_intervals:
        raise FileNotFoundError(
            "No cleaned Binned CSV files were found."
        )

    traffic = pd.concat(all_intervals, ignore_index=True)

    # Convert times to UTC so daylight saving time is handled correctly.
    traffic["Time"] = pd.to_datetime(
        traffic["Time"],
        errors="coerce",
        utc=True,
    )

    # Put the two locations into separate columns.
    traffic = traffic.pivot(
        index="Time",
        columns="Location",
        values="Traffic",
    ).sort_index()

    return traffic


def make_daily(series):
    """Add 15-minute values into local daily traffic totals."""

    local_dates = series.index.tz_convert(
        "America/Chicago"
    ).date

    return series.groupby(local_dates).sum()


def estimate_b(location_a, location_b, number_of_points):
    """Estimate Location B using a short common observation period."""

    paired = pd.concat(
        [location_a.rename("A"), location_b.rename("B")],
        axis=1,
        join="inner",
    ).dropna()

    short_period = paired.iloc[:number_of_points]

    if len(short_period) < number_of_points:
        raise ValueError("Not enough shared observations.")

    # c = B traffic divided by A traffic during the short period.
    scale = short_period["B"].sum() / short_period["A"].sum()

    # Use all known A values to estimate B.
    predicted_b = location_a * scale

    return scale, predicted_b


def main():
    traffic = load_traffic()

    location_a = traffic[LOCATION_A].dropna()
    location_b = traffic[LOCATION_B].dropna()

    actual_daily_b = make_daily(location_b)
    actual_aadt = actual_daily_b.mean()

    results = []
    all_daily_predictions = []

    for duration, number_of_points in DURATIONS.items():
        scale, predicted_b = estimate_b(
            location_a,
            location_b,
            number_of_points,
        )

        predicted_daily_b = make_daily(predicted_b)

        # Compare actual and predicted B on dates with valid B data.
        daily = pd.concat(
            [
                actual_daily_b.rename("Actual_B"),
                predicted_daily_b.rename("Predicted_B"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        estimated_aadt = predicted_daily_b.mean()
        percent_error = (
            abs(estimated_aadt - actual_aadt)
            / actual_aadt
            * 100
        )
        daily_mae = (
            daily["Actual_B"] - daily["Predicted_B"]
        ).abs().mean()

        results.append({
            "Duration": duration,
            "15_Minute_Points": number_of_points,
            "Scale_Factor": scale,
            "Actual_AADT": actual_aadt,
            "Estimated_AADT": estimated_aadt,
            "AADT_Percent_Error": percent_error,
            "Daily_MAE": daily_mae,
        })

        daily = daily.reset_index(names="Date")
        daily["Duration"] = duration
        all_daily_predictions.append(daily)

    results = pd.DataFrame(results)
    daily_predictions = pd.concat(
        all_daily_predictions,
        ignore_index=True,
    )

    # Save results for the dashboard.
    results.to_csv(
        DATA_FOLDER / "annual_estimation_results.csv",
        index=False,
    )
    daily_predictions.to_csv(
        DATA_FOLDER / "annual_daily_predictions.csv",
        index=False,
    )

    # Graph 1: amount of B data versus AADT percent error.
    plt.figure(figsize=(9, 5))
    plt.plot(
        results["Duration"],
        results["AADT_Percent_Error"],
        marker="o",
    )
    plt.xlabel("Amount of Location B Data")
    plt.ylabel("AADT Percent Error (%)")
    plt.title("How Much Location B Data Is Needed?")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(
        DATA_FOLDER / "annual_estimation_error.png",
        dpi=300,
    )
    plt.close()

    # Graph 2: actual and predicted daily traffic using 14 days.
    example = daily_predictions.loc[
        daily_predictions["Duration"] == "14 days"
    ].copy()
    example["Date"] = pd.to_datetime(example["Date"])

    plt.figure(figsize=(12, 5))
    plt.plot(
        example["Date"],
        example["Actual_B"],
        label="Actual B",
        linewidth=1,
    )
    plt.plot(
        example["Date"],
        example["Predicted_B"],
        label="Predicted B",
        linewidth=1,
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Motor Vehicle Volume")
    plt.title("Actual and Predicted Location B Using 14 Days")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        DATA_FOLDER / "annual_actual_vs_predicted.png",
        dpi=300,
    )
    plt.close()

    print("\nANNUAL TRAFFIC ESTIMATION RESULTS")
    print(
        results[
            [
                "Duration",
                "Scale_Factor",
                "Actual_AADT",
                "Estimated_AADT",
                "AADT_Percent_Error",
                "Daily_MAE",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nSaved files:")
    print("annual_estimation_results.csv")
    print("annual_daily_predictions.csv")
    print("annual_estimation_error.png")
    print("annual_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
