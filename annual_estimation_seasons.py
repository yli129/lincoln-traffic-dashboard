"""Repeat the annual traffic experiment in four seasons."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from annual_estimation import (
    DURATIONS,
    LOCATION_A,
    LOCATION_B,
    load_traffic,
    make_daily,
)


DATA_FOLDER = Path(__file__).parent

# Four Monday starting dates that avoid known bad data.
START_DATES = {
    "Winter": "2025-01-06",
    "Spring": "2025-04-07",
    "Summer": "2025-06-02",
    "Fall": "2025-09-15",
}


def main():
    # Read the cleaned 15-minute traffic data.
    traffic = load_traffic()

    location_a = traffic[LOCATION_A].dropna()
    location_b = traffic[LOCATION_B].dropna()

    # Use all valid Location B days as the benchmark.
    actual_daily_b = make_daily(location_b)
    actual_aadt = actual_daily_b.mean()

    # Keep only timestamps available at both locations.
    paired = pd.concat(
        [
            location_a.rename("A"),
            location_b.rename("B"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    results = []

    for season, start_date in START_DATES.items():
        # Convert local midnight into UTC.
        start_time = pd.Timestamp(
            start_date,
            tz="America/Chicago",
        ).tz_convert("UTC")

        # Start the experiment on this season's date.
        available = paired.loc[
            paired.index >= start_time
        ]

        for duration, number_of_points in DURATIONS.items():
            short_period = available.iloc[:number_of_points]

            if len(short_period) < number_of_points:
                continue

            # Calculate c from the short common period.
            scale = (
                short_period["B"].sum()
                / short_period["A"].sum()
            )

            # Estimate all of Location B from Location A.
            predicted_b = location_a * scale
            predicted_daily_b = make_daily(predicted_b)
            estimated_aadt = predicted_daily_b.mean()

            percent_error = (
                abs(estimated_aadt - actual_aadt)
                / actual_aadt
                * 100
            )

            # Compare daily totals on valid B dates.
            daily = pd.concat(
                [
                    actual_daily_b.rename("Actual_B"),
                    predicted_daily_b.rename("Predicted_B"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            daily_mae = (
                daily["Actual_B"]
                - daily["Predicted_B"]
            ).abs().mean()

            results.append({
                "Season": season,
                "Start_Date": start_date,
                "Duration": duration,
                "Scale_Factor": scale,
                "Actual_AADT": actual_aadt,
                "Estimated_AADT": estimated_aadt,
                "AADT_Percent_Error": percent_error,
                "Daily_MAE": daily_mae,
            })

    results = pd.DataFrame(results)

    # Keep durations in the intended order.
    duration_order = list(DURATIONS)
    results["Duration"] = pd.Categorical(
        results["Duration"],
        categories=duration_order,
        ordered=True,
    )
    results = results.sort_values(
        ["Season", "Duration"]
    )

    # Calculate a summary across the four seasons.
    summary = (
        results.groupby(
            "Duration",
            observed=True,
        )["AADT_Percent_Error"]
        .agg(["mean", "median", "min", "max"])
        .reindex(duration_order)
        .reset_index()
    )

    summary = summary.rename(columns={
        "mean": "Mean_Error",
        "median": "Median_Error",
        "min": "Minimum_Error",
        "max": "Maximum_Error",
    })

    # Save the detailed and summary tables.
    results.to_csv(
        DATA_FOLDER / "seasonal_estimation_results.csv",
        index=False,
    )
    summary.to_csv(
        DATA_FOLDER / "seasonal_error_summary.csv",
        index=False,
    )

    # Plot one error line for each season.
    plt.figure(figsize=(10, 6))

    for season in START_DATES:
        group = results.loc[
            results["Season"] == season
        ].sort_values("Duration")

        plt.plot(
            group["Duration"].astype(str),
            group["AADT_Percent_Error"],
            marker="o",
            label=season,
            alpha=0.75,
        )

    # Add the average error as a thicker black line.
    plt.plot(
        summary["Duration"].astype(str),
        summary["Mean_Error"],
        marker="o",
        color="black",
        linewidth=3,
        label="Four-season average",
    )

    plt.xlabel("Amount of Location B Data")
    plt.ylabel("AADT Percent Error (%)")
    plt.title("Prediction Error from Four Seasonal Starting Dates")
    plt.xticks(rotation=25)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        DATA_FOLDER / "seasonal_error_comparison.png",
        dpi=300,
    )
    plt.close()

    print("\nAADT PERCENT ERROR BY SEASON")

    error_table = results.pivot(
        index="Duration",
        columns="Season",
        values="AADT_Percent_Error",
    ).reindex(duration_order)

    print(error_table.round(2).to_string())

    print("\nFOUR-SEASON SUMMARY")
    print(summary.round(2).to_string(index=False))

    best = summary.loc[summary["Mean_Error"].idxmin()]

    print(
        f"\nLowest average error: {best['Duration']} "
        f"({best['Mean_Error']:.2f}%)"
    )

    print("\nSaved files:")
    print("seasonal_estimation_results.csv")
    print("seasonal_error_summary.csv")
    print("seasonal_error_comparison.png")


if __name__ == "__main__":
    main()
