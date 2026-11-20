"""
Monthly and Day-of-Week Traffic Adjustment Factors
--------------------------------------------------
Input: annual_daily_predictions.csv

This input contains one row per DATE and estimation DURATION, not one row
per 15-minute traffic observation. Therefore, this program uses Actual_B
once per date (it does NOT sum the seven duration rows together).

Outputs:
  monthly_adjustment_factors.csv
  weekday_adjustment_factors.csv
  daily_adjusted_estimates.csv
  adjustment_factor_summary.txt

The adjustment factor is:
    period factor = average daily traffic in period / overall average daily traffic

To convert a count from that period to an annual-average daily estimate:
    adjusted AADT = observed count / period factor

Example: if January's factor is 0.90, divide a January count by 0.90.
"""

from pathlib import Path
import pandas as pd

# Keep this script in the same folder as annual_daily_predictions.csv.
DATA_FILE = Path(__file__).resolve().parent / "annual_daily_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

DATE_COLUMN = "Date"
ACTUAL_COLUMN = "Actual_B"

# Minimum number of observed days required to publish a period factor.
MIN_DAYS_PER_PERIOD = 2


def load_daily_data(file_path):
    """Load the file and reduce its multiple duration rows to one value per day."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}.\n"
            "Put this Python script and annual_daily_predictions.csv "
            "in the same folder."
        )

    data = pd.read_csv(file_path)

    required = {DATE_COLUMN, ACTUAL_COLUMN}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[ACTUAL_COLUMN] = pd.to_numeric(data[ACTUAL_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN, ACTUAL_COLUMN]).copy()

    # Actual_B is repeated for each estimation duration on a given date.
    # Taking the daily mean avoids counting the same day's traffic 7 times.
    data["_Date_Only"] = data[DATE_COLUMN].dt.normalize()
    daily = (
        data.groupby("_Date_Only", as_index=False)[ACTUAL_COLUMN]
        .mean()
        .rename(columns={"_Date_Only": "Date", ACTUAL_COLUMN: "Daily_Traffic"})
    )

    daily["Month_Number"] = daily["Date"].dt.month
    daily["Month"] = daily["Date"].dt.month_name()
    daily["Day_Number"] = daily["Date"].dt.dayofweek
    daily["Day_of_Week"] = daily["Date"].dt.day_name()

    return daily.sort_values("Date").reset_index(drop=True)


def calculate_factors(daily):
    """Calculate period averages and adjustment factors relative to overall mean."""
    overall_average = daily["Daily_Traffic"].mean()
    if pd.isna(overall_average) or overall_average <= 0:
        raise ValueError("The overall average daily traffic must be positive.")

    monthly = (
        daily.groupby(["Month_Number", "Month"])["Daily_Traffic"]
        .agg(Average_Daily_Traffic="mean", Days_Observed="count")
        .reset_index()
        .sort_values("Month_Number")
    )
    monthly["Adjustment_Factor"] = (
        monthly["Average_Daily_Traffic"] / overall_average
    )
    monthly["AADT_Correction_Multiplier"] = 1 / monthly["Adjustment_Factor"]
    monthly["Factor_Status"] = monthly["Days_Observed"].apply(
        lambda n: "OK" if n >= MIN_DAYS_PER_PERIOD else "LOW SAMPLE"
    )

    weekdays = (
        daily.groupby(["Day_Number", "Day_of_Week"])["Daily_Traffic"]
        .agg(Average_Daily_Traffic="mean", Days_Observed="count")
        .reset_index()
        .sort_values("Day_Number")
    )
    weekdays["Adjustment_Factor"] = (
        weekdays["Average_Daily_Traffic"] / overall_average
    )
    weekdays["AADT_Correction_Multiplier"] = 1 / weekdays["Adjustment_Factor"]
    weekdays["Factor_Status"] = weekdays["Days_Observed"].apply(
        lambda n: "OK" if n >= MIN_DAYS_PER_PERIOD else "LOW SAMPLE"
    )

    return overall_average, monthly, weekdays


def create_daily_adjusted_estimates(daily, monthly, weekdays, overall_average):
    """
    Demonstrate adjustment of each day's observed count to an AADT estimate.

    Monthly-only and weekday-only estimates are shown separately. They are
    not multiplied together because doing so can double-count variation.
    """
    result = daily.copy()

    month_map = monthly.set_index("Month_Number")["Adjustment_Factor"]
    weekday_map = weekdays.set_index("Day_Number")["Adjustment_Factor"]

    result["Monthly_Factor"] = result["Month_Number"].map(month_map)
    result["Weekday_Factor"] = result["Day_Number"].map(weekday_map)

    result["AADT_Estimate_Month_Adjusted"] = (
        result["Daily_Traffic"] / result["Monthly_Factor"]
    )
    result["AADT_Estimate_Weekday_Adjusted"] = (
        result["Daily_Traffic"] / result["Weekday_Factor"]
    )

    # A simple benchmark: the unadjusted observed daily count.
    result["Unadjusted_Daily_Count"] = result["Daily_Traffic"]
    result["Overall_AADT_Reference"] = overall_average
    return result


def main():
    daily = load_daily_data(DATA_FILE)
    overall_average, monthly, weekdays = calculate_factors(daily)

    adjusted = create_daily_adjusted_estimates(
        daily, monthly, weekdays, overall_average
    )

    monthly.to_csv(OUTPUT_DIR / "monthly_adjustment_factors.csv", index=False)
    weekdays.to_csv(OUTPUT_DIR / "weekday_adjustment_factors.csv", index=False)
    adjusted.to_csv(OUTPUT_DIR / "daily_adjusted_estimates.csv", index=False)

    missing_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    missing_dates = missing_dates.difference(pd.DatetimeIndex(daily["Date"]))

    lines = [
        "MONTHLY AND DAY-OF-WEEK ADJUSTMENT FACTORS",
        "=" * 48,
        f"Dates included: {daily['Date'].min().date()} through {daily['Date'].max().date()}",
        f"Unique dates with data: {len(daily)}",
        f"Overall average daily traffic (reference AADT): {overall_average:,.2f}",
        f"Missing calendar dates within range: {len(missing_dates)}",
        "",
        "MONTHLY FACTORS",
        "Factor < 1.00: month is below the overall daily average.",
        "Factor > 1.00: month is above the overall daily average.",
        "AADT correction multiplier = 1 / adjustment factor.",
        monthly.to_string(index=False, formatters={
            "Average_Daily_Traffic": "{:,.2f}".format,
            "Adjustment_Factor": "{:.4f}".format,
            "AADT_Correction_Multiplier": "{:.4f}".format,
        }),
        "",
        "DAY-OF-WEEK FACTORS",
        weekdays.to_string(index=False, formatters={
            "Average_Daily_Traffic": "{:,.2f}".format,
            "Adjustment_Factor": "{:.4f}".format,
            "AADT_Correction_Multiplier": "{:.4f}".format,
        }),
        "",
        "HOW TO USE",
        "For a count taken in a known month: estimated AADT = count / that month's factor.",
        "For a count taken on a known weekday: estimated AADT = count / that weekday's factor.",
        "If both month and weekday are known, do not multiply these two factors blindly.",
        "Use a separately validated month-by-weekday model before combining them.",
        "",
        "IMPORTANT LIMITATION",
        "These factors are estimated from the dates in this input file. The file has",
        "one Actual_B value repeated across several estimation durations per date;",
        "the program uses one daily value per date. Factors from one year/location",
        "should be tested on other years/locations before being used citywide.",
    ]

    summary_path = OUTPUT_DIR / "adjustment_factor_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:8]))
    print("\nSaved files:")
    print("  monthly_adjustment_factors.csv")
    print("  weekday_adjustment_factors.csv")
    print("  daily_adjusted_estimates.csv")
    print("  adjustment_factor_summary.txt")


if __name__ == "__main__":
    main()
