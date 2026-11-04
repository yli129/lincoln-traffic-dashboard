"""Simple traffic overview for two intersections.

Reads the traffic-counter CSV export's Lights section. Put this script in the
same folder as the CSV files, then run: python intersection_overview.py
Requires: pandas, numpy, matplotlib
"""
from pathlib import Path
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATTERN = "*Binned_*.csv"
TIME_ZONE = "America/Chicago"
MOVEMENTS = ("_Right", "_Thru", "_Left", "_U-Turn")


def read_counter_csv(filename):
    with open(filename, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    location = Path(filename).stem
    for row in rows:
        if len(row) > 1 and row[0].strip() == "Location":
            location = row[1].strip()
            break

    section = next((i for i, row in enumerate(rows)
                    if row and row[0].strip() == "Lights"), None)
    if section is None:
        raise ValueError(f"{filename}: couldn't find the Lights section")

    if len(rows) <= section + 3:
        raise ValueError(f"{filename}: incomplete Lights section")

    direction_row = rows[section + 2]
    movement_row = rows[section + 3]
    columns = ["Time"]
    direction = ""
    for j in range(1, len(movement_row)):
        if j < len(direction_row) and direction_row[j].strip():
            direction = direction_row[j].strip()
        movement = movement_row[j].strip()
        columns.append(f"{direction}_{movement}".strip("_"))

    records = []
    for row in rows[section + 4:]:
        if not row or not row[0].strip().startswith("20"):
            if records:
                break
            continue
        row = row[:len(columns)] + [""] * max(0, len(columns) - len(row))
        records.append(row)

    if not records:
        raise ValueError(f"{filename}: no timestamped traffic rows found")

    frame = pd.DataFrame(records, columns=columns)
    frame["Time"] = pd.to_datetime(
        frame["Time"], format="%Y-%m-%d %H:%M:%S %z",
        errors="coerce", utc=True
    ).dt.tz_convert(TIME_ZONE)
    frame = frame.dropna(subset=["Time"])

    count_columns = [c for c in frame.columns if c.endswith(MOVEMENTS)]
    if not count_columns:
        raise ValueError(f"{filename}: couldn't identify vehicle movement columns")
    for col in count_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    frame["Vehicles"] = frame[count_columns].sum(axis=1)
    return location, frame[["Time", "Vehicles"]]


def load_all():
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No files matching {FILE_PATTERN!r} found in {Path.cwd()}"
        )
    by_location = {}
    for filename in files:
        location, frame = read_counter_csv(filename)
        print(f"Read {Path(filename).name} — {location}: {len(frame):,} intervals")
        by_location.setdefault(location, []).append(frame)

    result = {}
    for location, frames in by_location.items():
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values("Time").drop_duplicates("Time", keep="last")
        data = data.set_index("Time").sort_index()
        data["Date"] = data.index.date
        data["Day"] = data.index.day_name()
        data["DayNumber"] = data.index.dayofweek
        data["Hour"] = data.index.hour
        result[location] = data
    return result


def main():
    intersections = load_all()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    daily_tables = {}

    print("\n" + "=" * 72)
    print("TRAFFIC OVERVIEW — TWO INTERSECTIONS")
    print("=" * 72)
    print("Counts include right turns, through traffic, left turns, and U-turns.")
    print("Each daily total is the sum of the recorded 15-minute intervals.")

    for location, data in intersections.items():
        daily = data.groupby("Date")["Vehicles"].sum().rename("Vehicles")
        daily.index = pd.to_datetime(daily.index)
        daily_tables[location] = daily
        by_day = daily.groupby(daily.index.day_name()).agg(["mean", "median", "std", "count"])
        by_day = by_day.reindex(day_order)
        weekday = daily[daily.index.dayofweek < 5]
        weekend = daily[daily.index.dayofweek >= 5]
        busiest_day = by_day["mean"].idxmax()
        busiest_date = daily.idxmax()
        weekday_avg = weekday.mean() if len(weekday) else float("nan")
        weekend_avg = weekend.mean() if len(weekend) else float("nan")
        pct = ((weekend_avg - weekday_avg) / weekday_avg * 100
               if weekday_avg else float("nan"))

        print(f"\n--- {location} ---")
        print(f"Data coverage: {daily.index.min().date()} through {daily.index.max().date()}")
        print(f"Days recorded: {len(daily)}")
        print(f"Average vehicles per day: {daily.mean():,.0f}")
        print(f"Median vehicles per day:  {daily.median():,.0f}")
        print(f"Busiest average day: {busiest_day} ({by_day.loc[busiest_day, 'mean']:,.0f} vehicles/day)")
        print(f"Highest single day: {busiest_date.date()} ({daily.max():,.0f} vehicles)")
        print(f"Weekday average: {weekday_avg:,.0f} vehicles/day")
        print(f"Weekend average: {weekend_avg:,.0f} vehicles/day")
        print(f"Weekend vs weekday: {pct:+.1f}%")
        print("\nAverage traffic by day of week:")
        for day in day_order:
            if day in by_day.index and pd.notna(by_day.loc[day, "mean"]):
                print(f"  {day:<10} {by_day.loc[day, 'mean']:>12,.0f} vehicles/day "
                      f"(based on {int(by_day.loc[day, 'count'])} days)")

        hourly_by_date = data.groupby([data.index.date, data.index.hour])["Vehicles"].sum()
        hourly_average = hourly_by_date.groupby(level=1).mean()
        peak_hour = int(hourly_average.idxmax())
        print(f"\nBusiest average hour: {peak_hour:02d}:00–{peak_hour+1:02d}:00 "
              f"({hourly_average.max():,.0f} vehicles/hour on average)")

    # Compare only shared calendar dates to make the intersection comparison fair.
    if len(daily_tables) >= 2:
        common = pd.concat(daily_tables, axis=1, join="inner").dropna()
        print("\n" + "=" * 72)
        print("FAIR INTERSECTION COMPARISON (SHARED DATES ONLY)")
        print("=" * 72)
        if not common.empty:
            print(f"Shared date range: {common.index.min().date()} through {common.index.max().date()}")
            for location in common.columns:
                print(f"{location}: {common[location].mean():,.0f} vehicles/day on shared dates")
            print("\nAverage daily volume ranking:")
            for rank, (name, value) in enumerate(common.mean().sort_values(ascending=False).items(), 1):
                print(f"  {rank}. {name}: {value:,.0f} vehicles/day")
        else:
            print("No shared calendar dates were found; cannot make a same-date comparison.")

    # Save daily totals and day-of-week averages for later project use.
    daily_output = pd.concat(daily_tables, axis=1)
    daily_output.columns.name = "Intersection"
    daily_output.to_csv("intersection_daily_totals.csv", index_label="Date")

    average_rows = []
    for location, daily in daily_tables.items():
        grouped = daily.groupby(daily.index.day_name()).agg(["mean", "median", "std", "count"]).reindex(day_order)
        for day, row in grouped.iterrows():
            average_rows.append({"Intersection": location, "Day": day,
                                 "Average vehicles/day": row["mean"],
                                 "Median vehicles/day": row["median"],
                                 "Std dev": row["std"], "Days observed": row["count"]})
    pd.DataFrame(average_rows).to_csv("intersection_day_of_week_summary.csv", index=False)

    # Chart 1: average traffic for each weekday at each intersection.
    plot_data = {}
    for location, daily in daily_tables.items():
        plot_data[location] = daily.groupby(daily.index.day_name()).mean().reindex(day_order)
    ax = pd.DataFrame(plot_data).plot(kind="bar", figsize=(12, 6))
    ax.set_title("Average Daily Traffic by Day of Week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average vehicles per day")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("intersection_day_of_week_comparison.png", dpi=160)
    plt.show()

    # Chart 2: average 15-minute traffic by hour of day.
    plt.figure(figsize=(12, 6))
    for location, data in intersections.items():
        hourly_profile = data.groupby([data.index.date, data.index.hour])["Vehicles"].sum().groupby(level=1).mean()
        plt.plot(hourly_profile.index, hourly_profile.values, marker="o", label=location)
    plt.title("Average Traffic Pattern Throughout the Day")
    plt.xlabel("Hour of day (local time)")
    plt.ylabel("Average vehicles per hour")
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("intersection_hourly_traffic_comparison.png", dpi=160)
    plt.show()

    print("\nSaved output files:")
    print("  intersection_daily_totals.csv")
    print("  intersection_day_of_week_summary.csv")
    print("  intersection_day_of_week_comparison.png")
    print("  intersection_hourly_traffic_comparison.png")
    print("\nWhat this data can help investigate:")
    print("  • When each intersection experiences recurring high traffic.")
    print("  • Whether weekend and weekday demand differ.")
    print("  • Which location has greater average daily traffic.")
    print("  • Potential times for signal-timing or traffic-management studies.")
    print("  • Whether traffic varies substantially from day to day.")
    print("These counts describe vehicles passing through, not queue length or delay.")

if __name__ == "__main__":
    main()
