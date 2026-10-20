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
    issues = []

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
    # Record off-grid times
    off_grid_rows = valid.index[~on_grid]

    for idx in off_grid_rows:
        issues.append({
            "File": file_path.name,
            "Section": data.loc[idx, "Section"],
            "Time": data.loc[idx, "Time"],
            "Row": idx,
            "Column": 0,
            "Issue": "Off-grid Time"
        })
    unique_on_grid = valid.loc[
        on_grid, ["Section", "Time"]
    ].drop_duplicates()
    # Record duplicate times
    duplicate_rows = valid[
        valid.duplicated(["Section", "Time"], keep=False)
    ]

    for idx, row in duplicate_rows.iterrows():
        issues.append({
            "File": file_path.name,
            "Section": row["Section"],
            "Time": row["Time"],
            "Row": idx,
            "Column": None,
            "Issue": "Duplicate Time"
        })
    missing_time_bins = expected_rows - len(unique_on_grid)
    # Record missing time bins
    for section in sections:
        section_times = unique_on_grid.loc[
            unique_on_grid["Section"] == section, "Time"
        ]

        missing_times = expected_times.difference(section_times)

        for time in missing_times:
            issues.append({
                "File": file_path.name,
                "Section": section,
                "Time": time,
                "Row": None,
                "Column": None,
                "Issue": "Missing Time Bin"
            })
    off_grid_times = (~on_grid).sum()
    invalid_times = data["Time"].isna().sum()
    # Record invalid times
    invalid_rows = data.index[data["Time"].isna()]

    for idx in invalid_rows:
        issues.append({
            "File": file_path.name,
            "Section": data.loc[idx, "Section"],
            "Time": data.loc[idx, 0],
            "Row": idx,
            "Column": 0,
            "Issue": "Invalid Time"
        })
    missing_values = values.isna().sum().sum()
    # Record missing values
    for col in count_columns:
        missing_rows = values.index[values[col].isna()]

        for idx in missing_rows:
            issues.append({
                "File": file_path.name,
                "Section": data.loc[idx, "Section"],
                "Time": data.loc[idx, "Time"],
                "Row": idx,
                "Column": col,
                "Issue": "Missing Value"
            })
    negative_values = (values < 0).sum().sum()
    # Record negative values
    for col in count_columns:
        negative_rows = values.index[values[col] < 0]

        for idx in negative_rows:
            issues.append({
                "File": file_path.name,
                "Section": data.loc[idx, "Section"],
                "Time": data.loc[idx, "Time"],
                "Row": idx,
                "Column": col,
                "Issue": "Negative Value"
            })
    sections_found = data["Section"].nunique()
    # Record missing sections
    found_sections = set(data["Section"].dropna())

    missing_sections = set(sections) - found_sections

    for section in missing_sections:
        issues.append({
            "File": file_path.name,
            "Section": section,
            "Time": None,
            "Row": None,
            "Column": None,
            "Issue": "Missing Section"
        })

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
    # Record zero-light days
    zero_dates = daily_light_total.index[daily_light_total == 0]

    for date in zero_dates:
        issues.append({
            "File": file_path.name,
            "Section": "Lights",
            "Time": date,
            "Row": None,
            "Column": None,
            "Issue": "Zero Lights Day"
        })
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

    summary = {
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

    issues_df = pd.DataFrame(
        issues,
        columns=["File", "Section", "Time", "Row", "Column", "Issue"]
    )
    return summary, issues_df


# Find all original CSV files.
# Do not check the report CSV created by this program.
csv_files = sorted(
    file_path
    for file_path in data_folder.glob("*.csv")
    if file_path.name not in [
        "csv_data_check_report.csv",
        "csv_issues_report.csv"
    ]
)

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files were found.")

# Check every CSV file.
results = []
all_issues = []

for file_path in csv_files:
    summary, issues_df = check_csv(file_path)

    results.append(summary)
    all_issues.append(issues_df)

report = pd.DataFrame(results)

issues_report = pd.concat(
    all_issues,
    ignore_index=True
)

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

# Save summary report
output = data_folder / "csv_data_check_report.csv"
report.to_csv(output, index=False)

# Save detailed issues report
issues_output = data_folder / "csv_issues_report.csv"
issues_report.to_csv(issues_output, index=False)

print("\nReports saved as:")
print(output)
print(issues_output)