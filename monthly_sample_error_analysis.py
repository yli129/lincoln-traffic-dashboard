"""
Monthly partial-count accuracy test
===================================

Question answered:
"If I collect the first 7, 14, 21, or 28 calendar days of a month,
how accurately can I estimate the traffic for the rest of that month?"

Input:
    annual_daily_predictions.csv

Required columns:
    Date, Actual_B

Important:
- The input has seven rows per date (one per Duration).
- Actual_B is identical across those seven rows, so this script keeps
  one Actual_B value per date. It does NOT count a date seven times.
- The analysis uses actual observed daily traffic (Actual_B), not
  Predicted_B, to test the sampling method.
- Missing dates are excluded from both the observed sample and the
  actual remainder. The script reports the number of available days.

Outputs:
    weekly_sample_error_results.csv
    weekly_sample_error_summary.csv
    weekly_sample_error_curve.png
    weekly_sample_error_by_month.png

Run:
    python monthly_sample_error_analysis.py
"""

from pathlib import Path
import calendar
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).resolve().parent / "annual_daily_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

DATE_COLUMN = "Date"
TRAFFIC_COLUMN = "Actual_B"

# First 1, 2, 3, or 4 weeks of the month (cumulative).
CUTOFF_DAYS = [7, 14, 21, 28]


def load_daily_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.name}.\n"
            "Place this script in the same folder as annual_daily_predictions.csv."
        )

    df = pd.read_csv(path)

    missing = {DATE_COLUMN, TRAFFIC_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[TRAFFIC_COLUMN] = pd.to_numeric(df[TRAFFIC_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN, TRAFFIC_COLUMN]).copy()
    df[DATE_COLUMN] = df[DATE_COLUMN].dt.normalize()

    # Confirm Actual_B is consistent across the seven Duration rows.
    consistency = df.groupby(DATE_COLUMN)[TRAFFIC_COLUMN].nunique()
    inconsistent = consistency[consistency > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Actual_B is not identical across all rows for these dates: "
            + ", ".join(str(x.date()) for x in inconsistent.index[:10])
        )

    # One actual daily count per calendar date.
    daily = (
        df.groupby(DATE_COLUMN, as_index=False)[TRAFFIC_COLUMN]
        .first()
        .rename(columns={DATE_COLUMN: "Date", TRAFFIC_COLUMN: "Actual_Traffic"})
        .sort_values("Date")
    )
    daily["Year"] = daily["Date"].dt.year
    daily["Month_Number"] = daily["Date"].dt.month
    daily["Month"] = daily["Date"].dt.month_name()
    daily["Day_of_Month"] = daily["Date"].dt.day
    return daily.reset_index(drop=True)


def safe_percent_error(estimate, actual):
    """Signed percentage error; positive means overestimate."""
    if actual == 0:
        return float("nan")
    return (estimate - actual) / actual * 100.0


def run_backtest(daily):
    """
    For each month and cutoff:
      1. Use the first N calendar days as the observed sample.
      2. Calculate average daily traffic in that sample.
      3. Predict remaining-month total = sample average * number of
         available dates after the cutoff.
      4. Compare with actual traffic on those remaining available dates.
      5. Also calculate estimated full-month total and its error.
    """
    rows = []

    for (year, month_num), month_data in daily.groupby(["Year", "Month_Number"]):
        month_data = month_data.sort_values("Date").copy()
        month_name = calendar.month_name[int(month_num)]
        days_in_month = calendar.monthrange(int(year), int(month_num))[1]

        for cutoff in CUTOFF_DAYS:
            sample = month_data[month_data["Day_of_Month"] <= cutoff]
            remainder = month_data[month_data["Day_of_Month"] > cutoff]

            # Need at least one observed date and one remaining date.
            if sample.empty or remainder.empty:
                continue

            sample_days = len(sample)
            remaining_days = len(remainder)
            sample_average = sample["Actual_Traffic"].mean()
            sample_total = sample["Actual_Traffic"].sum()
            actual_remaining_total = remainder["Actual_Traffic"].sum()
            actual_month_total = month_data["Actual_Traffic"].sum()

            estimated_remaining_total = sample_average * remaining_days
            estimated_month_total = sample_total + estimated_remaining_total

            remaining_error = safe_percent_error(
                estimated_remaining_total, actual_remaining_total
            )
            month_error = safe_percent_error(
                estimated_month_total, actual_month_total
            )

            rows.append({
                "Year": int(year),
                "Month_Number": int(month_num),
                "Month": month_name,
                "Sample_Cutoff_Day": cutoff,
                "Sample_Description": f"First {cutoff} calendar days",
                "Available_Sample_Days": sample_days,
                "Available_Remaining_Days": remaining_days,
                "Sample_Average_Vehicles_Per_Day": sample_average,
                "Actual_Sample_Total": sample_total,
                "Estimated_Remaining_Month_Total": estimated_remaining_total,
                "Actual_Remaining_Month_Total": actual_remaining_total,
                "Remaining_Month_Signed_Error_Percent": remaining_error,
                "Remaining_Month_Absolute_Error_Percent": abs(remaining_error),
                "Estimated_Full_Month_Total": estimated_month_total,
                "Actual_Full_Month_Total": actual_month_total,
                "Full_Month_Signed_Error_Percent": month_error,
                "Full_Month_Absolute_Error_Percent": abs(month_error),
            })

    return pd.DataFrame(rows)


def summarize_results(results):
    summary = (
        results.groupby(["Sample_Cutoff_Day", "Sample_Description"])
        .agg(
            Months_Tested=("Month", "count"),
            Mean_Absolute_Error_Remaining_Percent=(
                "Remaining_Month_Absolute_Error_Percent", "mean"
            ),
            Median_Absolute_Error_Remaining_Percent=(
                "Remaining_Month_Absolute_Error_Percent", "median"
            ),
            Mean_Signed_Error_Remaining_Percent=(
                "Remaining_Month_Signed_Error_Percent", "mean"
            ),
            Mean_Absolute_Error_Full_Month_Percent=(
                "Full_Month_Absolute_Error_Percent", "mean"
            ),
            Mean_Signed_Error_Full_Month_Percent=(
                "Full_Month_Signed_Error_Percent", "mean"
            ),
        )
        .reset_index()
        .sort_values("Sample_Cutoff_Day")
    )
    return summary


def make_error_curve(summary):
    plt.figure(figsize=(10, 6))
    x = summary["Sample_Cutoff_Day"]

    plt.plot(
        x,
        summary["Mean_Absolute_Error_Remaining_Percent"],
        marker="o",
        linewidth=2,
        label="Predicting the remaining days",
    )
    plt.plot(
        x,
        summary["Mean_Absolute_Error_Full_Month_Percent"],
        marker="s",
        linewidth=2,
        label="Estimating the full month's total",
    )

    plt.xticks(CUTOFF_DAYS, ["Week 1\n(7 days)", "Week 2\n(14 days)",
                             "Week 3\n(21 days)", "Week 4\n(28 days)"])
    plt.xlabel("Amount of data collected at the start of the month")
    plt.ylabel("Mean absolute percentage error (%)")
    plt.title("Monthly Traffic Estimate Error vs. Amount of Data Collected")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weekly_sample_error_curve.png", dpi=200)
    plt.close()


def make_month_chart(results):
    pivot = results.pivot(
        index="Month",
        columns="Sample_Cutoff_Day",
        values="Remaining_Month_Absolute_Error_Percent",
    )
    month_order = list(calendar.month_name)[1:]
    pivot = pivot.reindex([m for m in month_order if m in pivot.index])

    ax = pivot.plot(kind="bar", figsize=(13, 7), width=0.82)
    ax.set_title("Remaining-Month Prediction Error by Month and Sample Week")
    ax.set_xlabel("Month")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.legend(
        title="Data collected through day",
        labels=[f"First {c} days" for c in pivot.columns],
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weekly_sample_error_by_month.png", dpi=200)
    plt.close()


def main():
    daily = load_daily_data(DATA_FILE)
    results = run_backtest(daily)

    if results.empty:
        raise ValueError(
            "No month had both sample days and remaining days. "
            "Check the date coverage in the input file."
        )

    summary = summarize_results(results)

    results.to_csv(OUTPUT_DIR / "weekly_sample_error_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "weekly_sample_error_summary.csv", index=False)

    make_error_curve(summary)
    make_month_chart(results)

    print("\nMONTHLY PARTIAL-COUNT ACCURACY TEST")
    print("=" * 48)
    print(f"Daily dates available: {len(daily)}")
    print(f"Date range: {daily['Date'].min().date()} to {daily['Date'].max().date()}")
    print("\nMean absolute percentage error (MAPE):")
    print(summary[[
        "Sample_Description",
        "Months_Tested",
        "Mean_Absolute_Error_Remaining_Percent",
        "Mean_Absolute_Error_Full_Month_Percent",
    ]].to_string(index=False, formatters={
        "Mean_Absolute_Error_Remaining_Percent": "{:.2f}%".format,
        "Mean_Absolute_Error_Full_Month_Percent": "{:.2f}%".format,
    }))

    print("\nInterpretation:")
    print("- Lower absolute percentage error means a more accurate estimate.")
    print("- Signed error above zero means the method overestimated traffic.")
    print("- Signed error below zero means the method underestimated traffic.")
    print("- Week 2 means the first 14 calendar days cumulatively, not just days 8-14.")
    print("- Months with no remaining days after a cutoff are excluded for that cutoff.")

    print("\nFiles created:")
    print("  weekly_sample_error_results.csv")
    print("  weekly_sample_error_summary.csv")
    print("  weekly_sample_error_curve.png")
    print("  weekly_sample_error_by_month.png")


if __name__ == "__main__":
    main()
