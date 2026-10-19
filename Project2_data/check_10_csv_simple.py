from pathlib import Path

import pandas as pd


# Put this Python file next to the Project2_data folder.
data_folder = Path(__file__).parent

sections = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
    "Bicycles",
    "Pedestrians",
]


# Check one CSV file.
def check_csv(file_path):
    # Read all 25 columns. Short rows are filled with NaN.
    raw = pd.read_csv(
        file_path,
        header=None,
        names=range(25),
        dtype=str,
    )

    # Read file information from the first few rows.
    start = pd.to_datetime(raw.iloc[3, 1], utc=True)
    end = pd.to_datetime(raw.iloc[4, 1], utc=True)
    location = raw.iloc[5, 1]

    # Add the section name to every data row.
    raw["Section"] = raw[0].where(raw[0].isin(sections)).ffill()

    # Keep only rows that start with a date.
    is_data_row = raw[0].fillna("").str.match(
        r"^\d{4}-\d{2}-\d{2}"
    )
    data = raw[is_data_row].copy()

    # Change the time column into datetime.
    data["Time"] = pd.to_datetime(
        data[0],
        format="%Y-%m-%d %H:%M:%S %z",
        errors="coerce",
        utc=True,
    )

    # Change the 24 count columns into numbers.
    count_columns = list(range(1, 25))
    values = data[count_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Make the expected 15-minute timeline.
    expected_times = pd.date_range(
        start=start,
        end=end,
        freq="15min",
    )
    expected_rows = len(expected_times) * len(sections)

    # Use Section + Time because each section repeats the timeline.
    valid = data.dropna(subset=["Time"])
    duplicates = valid.duplicated(["Section", "Time"]).sum()
    on_grid = valid["Time"].isin(expected_times)

    unique_on_grid = valid.loc[
        on_grid, ["Section", "Time"]
    ].drop_duplicates()

    missing_time_bins = expected_rows - len(unique_on_grid)
    off_grid_times = (~on_grid).sum()
    invalid_times = data["Time"].isna().sum()
    missing_values = values.isna().sum().sum()
    negative_values = (values < 0).sum().sum()
    sections_found = data["Section"].nunique()

    # Check days when all light-vehicle counts are zero.
    light_rows = data["Section"] == "Lights"
    light_data = pd.DataFrame(
        {
            "Time": data.loc[light_rows, "Time"],
            "Total": values.loc[light_rows].sum(
                axis=1,
                min_count=24,
            ),
        }
    ).dropna(subset=["Time"])

    light_data["Date"] = (
        light_data["Time"]
        .dt.tz_convert("America/Chicago")
        .dt.date
    )

    daily_light_total = light_data.groupby("Date")["Total"].sum()
    zero_light_days = (daily_light_total == 0).sum()

    # Decide whether this file needs attention.
    status = "GOOD"

    if (
        sections_found != 7
        or missing_time_bins > 0
        or duplicates > 0
        or invalid_times > 0
        or off_grid_times > 0
        or missing_values > 0
        or negative_values > 0
        or zero_light_days > 0
    ):
        status = "CHECK"

    return {
        "File": file_path.name,
        "Location": location,
        "Sections": sections_found,
        "Expected_rows": expected_rows,
        "Found_rows": len(data),
        "Missing_time_bins": missing_time_bins,
        "Duplicate_times": duplicates,
        "Invalid_times": invalid_times,
        "Off_grid_times": off_grid_times,
        "Missing_values": missing_values,
        "Negative_values": negative_values,
        "Zero_Lights_Days": zero_light_days,
        "Status": status,
    }


# Find all original CSV files.
# Do not check the report CSV created by this program.
csv_files = sorted(
    file_path
    for file_path in data_folder.glob("*.csv")
    if file_path.name != "csv_data_check_report.csv"
)

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files were found.")

# Check every CSV file.
results = []

for file_path in csv_files:
    results.append(check_csv(file_path))

report = pd.DataFrame(results)

# Print each CSV result clearly.
print("\n" + "=" * 70)
print("CSV DATA CHECK REPORT")
print("=" * 70)

for number, row in report.iterrows():

    print(f"\nCSV {number + 1}: {row['File']}")
    print("-" * 70)

    print(f"Location:             {row['Location']}")
    print(f"Status:               {row['Status']}")
    print(f"Sections found:       {row['Sections']} / 7")
    print(f"Expected rows:        {row['Expected_rows']}")
    print(f"Found rows:           {row['Found_rows']}")
    print(f"Missing time bins:    {row['Missing_time_bins']}")
    print(f"Duplicate times:      {row['Duplicate_times']}")
    print(f"Invalid times:        {row['Invalid_times']}")
    print(f"Off-grid times:       {row['Off_grid_times']}")
    print(f"Missing values:       {row['Missing_values']}")
    print(f"Negative values:      {row['Negative_values']}")
    print(f"Zero-light days:      {row['Zero_Lights_Days']}")

# Print a short summary.
good_files = (report["Status"] == "GOOD").sum()
check_files = (report["Status"] == "CHECK").sum()

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Total CSV files:", len(report))
print("GOOD files:", good_files)
print("Files to check:", check_files)

# Show the names of files that need attention.
if check_files > 0:
    print("\nFiles that need attention:")

    for file_name in report.loc[
        report["Status"] == "CHECK", "File"
    ]:
        print("-", file_name)
else:
    print("\nAll CSV files passed the checks.")

# Save the results as a CSV report.
output = data_folder / "csv_data_check_report.csv"
report.to_csv(output, index=False)

print("\nReport saved as:")
print(output)
