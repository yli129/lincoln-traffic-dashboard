# import package
from pathlib import Path
import pandas as pd

# Put this Python file in the same folder as the CSV files.
# __file__：the location of Python file
# .parent：the location of Python file folder
data_folder = Path(__file__).parent

# The current project keeps original files and reports in separate folders.
# Fall back to the script folder so the checker also works with the old layout.
raw_data_folder = data_folder / "raw_data"
if not raw_data_folder.exists():
    raw_data_folder = data_folder

report_folder = data_folder / "data_quality_reports"
report_folder.mkdir(exist_ok=True)

# 7 sections 
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
        header=None, # skip first column summary 
        names=range(25), # column 0 is time/section/text, column 1-24 is traffic num
        dtype=str, # read data using text type 
    )
    issues = []

    # Read file information from the first few rows.
    # iloc[row,column] start from 0 
    start = pd.to_datetime(raw.iloc[3, 1], utc=True) # 4 column, 2 row 
    end = pd.to_datetime(raw.iloc[4, 1], utc=True) # 4 column, 2 row 
    location = raw.iloc[5, 1] # 6 column, 2 row 

    # Add the section name to every data row.
    # keep only 7 section names,ffill() means forward fill 
    raw["Section"] = raw[0].where(raw[0].isin(sections)).ffill()

    # Keep only rows that start with a date.
    is_data_row = raw[0].fillna("").str.match(
        r"^\d{4}-\d{2}-\d{2}" # data must follow 2025-01-01 type 
    )
    data = raw[is_data_row].copy()

    # Convert the time column to datetime.
    data["Time"] = pd.to_datetime(
    data[0],
    format="%Y-%m-%d %H:%M:%S %z",  # %z reads the UTC offset, such as -0500 or -0600.
    errors="coerce",                 # Invalid times become NaT (missing datetime values).
    utc=True,                        # Convert all times to UTC so DST changes are handled correctly.
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
    # each time will repeat in 7 sections 
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

    # Check zero light-vehicle periods.
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

    # Keep the original raw row number so partial zero intervals can be
    # labeled later in the cleaning step.
    light_data["Row"] = light_data.index
    light_data["Date"] = (
        light_data["Time"]
        .dt.tz_convert("America/Chicago")
        .dt.date
    )
    light_data["Is_Zero"] = light_data["Total"] == 0

    # A full zero day must contain exactly 96 quarter-hour observations,
    # and all 96 observations must have a total of zero.
    daily_light = light_data.groupby("Date").agg(
        Intervals=("Time", "size"),
        Zero_Intervals=("Is_Zero", "sum"),
    )
    zero_dates = daily_light.index[
        (daily_light["Intervals"] == 96)
        & (daily_light["Zero_Intervals"] == 96)
    ]
    zero_light_days = len(zero_dates)

    for date in zero_dates:
        issues.append({
            "File": file_path.name,
            "Section": "Lights",
            "Time": date,
            "Row": None,
            "Column": None,
            "Issue": "Zero Lights Day"
        })

    # Report every zero interval that is not part of a complete zero day.
    # Each issue keeps its timestamp and original row number so another
    # script can assign a separate quality label to these records later.
    partial_zero_rows = light_data.loc[
        light_data["Is_Zero"]
        & ~light_data["Date"].isin(zero_dates)
    ]
    partial_zero_intervals = len(partial_zero_rows)

    for _, row in partial_zero_rows.iterrows():
        issues.append({
            "File": file_path.name,
            "Section": "Lights",
            "Time": row["Time"],
            "Row": int(row["Row"]),
            "Column": None,
            "Issue": "Partial Zero Lights Interval"
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
        or partial_zero_intervals > 0
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
        "Partial_Zero_Lights_Intervals": partial_zero_intervals,
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
    for file_path in raw_data_folder.glob("*.csv")
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

nonempty_issues = [table for table in all_issues if not table.empty]

if nonempty_issues:
    issues_report = pd.concat(
        nonempty_issues,
        ignore_index=True,
    )
else:
    issues_report = pd.DataFrame(
        columns=["File", "Section", "Time", "Row", "Column", "Issue"]
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
    print(
        "Partial zero-light "
        f"intervals: {row['Partial_Zero_Lights_Intervals']}"
    )

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
output = report_folder / "csv_data_check_report.csv"
report.to_csv(output, index=False)

# Save detailed issues report
issues_output = report_folder / "csv_issues_report.csv"
issues_report.to_csv(issues_output, index=False)

print("\nReports saved as:")
print(output)
print(issues_output)
