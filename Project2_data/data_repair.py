from pathlib import Path
import pandas as pd

data_folder = Path(__file__).parent

# Read inspection results
issues_df = pd.read_csv(
    data_folder / "csv_issues_report.csv"
)

def add_quality_flag(cleaned, row, flag):
    current = cleaned.loc[row, "Quality_Flag"]

    if current == "Valid":
        cleaned.loc[row, "Quality_Flag"] = flag
    elif flag not in current.split("; "):
        cleaned.loc[row, "Quality_Flag"] = current + "; " + flag

    return cleaned

def get_nearby_median(data, values, row, col, file_issues):

    # Get target section
    target_section = data.loc[row, "Section"]

    # Get local time from original CSV
    local_times = pd.to_datetime(
        data[0].str.slice(0, 19),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    target_time = local_times.loc[row]

    # Same section
    same_section = data["Section"] == target_section

    # Same time of day
    same_time = (
        (local_times.dt.hour == target_time.hour) &
        (local_times.dt.minute == target_time.minute)
    )

    # Within 7 days, excluding the target day
    day_diff = (
        local_times.dt.normalize() - target_time.normalize()
    ).abs().dt.days

    nearby = (day_diff >= 1) & (day_diff <= 7)

    # Get abnormal dates
    zero_days = file_issues.loc[
        file_issues["Issue"] == "Zero Lights Day",
        "Time"
    ]

    zero_days = pd.to_datetime(zero_days).dt.date

    # Exclude abnormal dates
    valid_day = ~local_times.dt.date.isin(zero_days)

    # Select valid candidates
    candidates = values.loc[
        same_section & same_time & nearby & valid_day,
        col
    ]

    candidates = candidates[
        candidates.notna() & (candidates >= 0)
    ]

    # No valid candidates
    if candidates.empty:
        return None

    return candidates.median()

def repair_missing_value(raw, data, values, file_issues):
    cleaned = raw.copy()

    # Find missing value issues
    missing_issues = file_issues[
        file_issues["Issue"] == "Missing Value"
    ]

    for _, issue in missing_issues.iterrows():

        row = int(issue["Row"])
        col = int(issue["Column"])

        # Calculate median from nearby dates
        median_value = get_nearby_median(
            data, values, row, col, file_issues
        )

        # Replace missing value
        if median_value is not None:
            cleaned.loc[row, col] = str(round(median_value))
            
            cleaned = add_quality_flag(
                cleaned, row, "Imputed"
            )
    return cleaned

def repair_negative_value(raw, data, values, file_issues):
    cleaned = raw.copy()

    negative_issues = file_issues[
        file_issues["Issue"] == "Negative Value"
    ]
    for _, issue in negative_issues.iterrows():

        row = int(issue["Row"])
        col = int(issue["Column"])

        # Calculate median from nearby dates
        median_value = get_nearby_median(
            data, values, row, col, file_issues
        )

        # Replace negative value
        if median_value is not None:
            cleaned.loc[row, col] = str(round(median_value))

            cleaned = add_quality_flag(
                cleaned, row, "Imputed"
            )
    return cleaned

def repair_zero_lights_day(raw, data, values, file_issues):
    cleaned = raw.copy()

    zero_issues = file_issues[
        file_issues["Issue"] == "Zero Lights Day"
    ]
    
    local_dates = pd.to_datetime(
        data[0].str.slice(0, 10),
        errors="coerce"
    ).dt.date
    for _, issue in zero_issues.iterrows():

        target_date = pd.to_datetime(issue["Time"]).date()

        mask = local_dates == target_date

        affected_rows = data.index[mask]

        for row in affected_rows:
            cleaned = add_quality_flag(
                cleaned, row, "Zero Lights Day"
            )

    return cleaned

def repair_duplicate_time(raw, data, values, file_issues):
    cleaned = raw.copy()

    duplicate_issues = file_issues[
        file_issues["Issue"] == "Duplicate Time"
    ]

    rows_to_drop = set()

    for _, issue in duplicate_issues.iterrows():

        row = int(issue["Row"])

        target_time = data.loc[row, "Time"]
        target_section = data.loc[row, "Section"]

        same_records = data[
            (data["Time"] == target_time) &
            (data["Section"] == target_section)
        ]
    
        duplicate_values = values.loc[same_records.index]

        if len(duplicate_values.drop_duplicates()) == 1:

            rows_to_drop.update(same_records.index[1:])
        else:
            for row in same_records.index:
                cleaned = add_quality_flag(
                    cleaned, row, "Duplicate Conflict"
                )

    cleaned = cleaned.drop(index=list(rows_to_drop))

    return cleaned

def repair_invalid_time(raw, data, values, file_issues):
    cleaned = raw.copy()

    invalid_issues = file_issues[
        file_issues["Issue"] == "Invalid Time"
    ]

    for _, issue in invalid_issues.iterrows():
        row = int(issue["Row"])

        original_time = cleaned.loc[row, 0]

        parsed_time = pd.to_datetime(
            original_time,
            errors="coerce",
            utc=True
        )

        if pd.notna(parsed_time):
            cleaned = add_quality_flag(
                cleaned, row, "Time Needs Review"
            )
        else:
            cleaned = add_quality_flag(
                cleaned, row, "Invalid Time"
            )

    return cleaned

def repair_off_grid_time(raw, data, values, file_issues):
    cleaned = raw.copy()

    off_grid_issues = file_issues[
        file_issues["Issue"] == "Off-grid Time"
    ]

    for _, issue in off_grid_issues.iterrows():

        row = int(issue["Row"])

        cleaned = add_quality_flag(
            cleaned, row, "Off-grid Time"
        )

    return cleaned

output_folder = data_folder / "cleaned"
output_folder.mkdir(exist_ok=True)

# Repair each file with issues
csv_files = [
    p for p in data_folder.glob("*.csv")
    if p.name not in [
        "csv_issues_report.csv",
        "csv_data_check_report.csv"
    ]
]

for file_path in csv_files:
    file_name = file_path.name

    file_issues = issues_df[
        issues_df["File"] == file_name
    ]

    file_path = data_folder / file_name

    raw = pd.read_csv(
        file_path,
        header=None,
        names=range(25),
        dtype=str
    )

    # Identify sections
    sections = [
        "Lights",
        "Single-Unit Trucks",
        "Articulated Trucks",
        "Buses",
        "Motorized Vehicles",
        "Bicycles",
        "Pedestrians",
    ]

    raw["Section"] = raw[0].where(
        raw[0].isin(sections)
    ).ffill()

    # Identify data rows
    is_data_row = raw[0].fillna("").str.match(
        r"^\d{4}-\d{2}-\d{2}"
    )

    data = raw[is_data_row].copy()

    # Convert time
    data["Time"] = pd.to_datetime(
        data[0],
        format="%Y-%m-%d %H:%M:%S %z",
        errors="coerce",
        utc=True
    )

    # Convert numerical values
    count_columns = list(range(1, 25))

    values = data[count_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )
    # Initialize
    cleaned = raw.copy()
    cleaned["Quality_Flag"] = "Valid"

    # Repair numerical errors
    cleaned = repair_missing_value(
        cleaned, data, values, file_issues
    )

    cleaned = repair_negative_value(
        cleaned, data, values, file_issues
    )

    # Flag abnormal 
    cleaned = repair_zero_lights_day(
        cleaned, data, values, file_issues
    )

    cleaned = repair_invalid_time(
        cleaned, data, values, file_issues
    )

    cleaned = repair_off_grid_time(
        cleaned, data, values, file_issues
    )

    cleaned = repair_duplicate_time(
        cleaned, data, values, file_issues
    )
    # Keep only traffic data rows
    cleaned_data = cleaned.loc[
        cleaned.index.intersection(data.index)
    ].copy()

    # Rename time column
    cleaned_data = cleaned_data.rename(columns={0: "Time"})

    # Select output columns
    output_columns = (
        ["Time", "Section"]
        + list(range(1, 25))
        + ["Quality_Flag"]
    )

    cleaned_data = cleaned_data[output_columns]

    directions = [
    "Southbound",
    "Westbound",
    "Northbound",
    "Eastbound"
    ]

    movements = [
        "Right", "Thru", "Left",
        "U-Turn", "Peds CW", "Peds CCW"
    ]
    
    feature_names = [
        f"{direction}_{movement}"
        for direction in directions
        for movement in movements
    ]

    rename_map = dict(zip(range(1,25), feature_names))

    cleaned_data = cleaned_data.rename(columns = rename_map)
    
    # Save cleaned CSV
    output_path = output_folder / file_name

    cleaned_data.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
