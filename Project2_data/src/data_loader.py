"""Load cleaned Lincoln traffic files and build movement-level daily totals.

All analysis code uses ``cleaned_data`` only. A local date is accepted only
when every required motor-vehicle record is present, numeric, unique, and has
``Quality_Flag == "Valid"``.
"""

from pathlib import Path

import pandas as pd

from config import (
    CLEANED_DATA_DIR,
    DIRECTIONS,
    LOCAL_TIME_ZONE,
    MOTOR_SECTIONS,
    MOVEMENT_COLUMNS,
    TURNS,
)


def location_from_filename(file_path: Path) -> str:
    """Convert a cleaned CSV filename into a readable intersection name."""

    return (
        file_path.name.split("_Binned_", 1)[0]
        .replace("_and_", " & ")
        .replace("_", " ")
    )


def discover_cleaned_files(folder: Path = CLEANED_DATA_DIR) -> dict[str, list[Path]]:
    """Group cleaned CSV files by intersection."""

    groups: dict[str, list[Path]] = {}
    for file_path in sorted(folder.glob("*_Binned_*.csv")):
        groups.setdefault(location_from_filename(file_path), []).append(file_path)

    if not groups:
        raise FileNotFoundError(
            f"No cleaned CSV files were found in {folder}. "
            "The analysis must read Project2_data/cleaned_data/."
        )
    return groups


def expected_intervals_by_date(year: int) -> pd.Series:
    """Return the correct number of local 15-minute intervals for each date."""

    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year + 1}-01-01 00:00",
        freq="15min",
        inclusive="left",
        tz=LOCAL_TIME_ZONE,
    )
    return pd.Series(timestamps.date).value_counts().sort_index()


def add_movement_measures(interval_movements: pd.DataFrame) -> pd.DataFrame:
    """Create overall, direction, turn, and direction-by-turn measures."""

    measures = pd.DataFrame(index=interval_movements.index)
    measures["Total Motor Vehicles"] = interval_movements[MOVEMENT_COLUMNS].sum(axis=1)

    for direction in DIRECTIONS:
        columns = [f"{direction}_{turn}" for turn in TURNS]
        measures[f"{direction} Total"] = interval_movements[columns].sum(axis=1)

    for turn in TURNS:
        columns = [f"{direction}_{turn}" for direction in DIRECTIONS]
        measures[f"All {turn} Turns"] = interval_movements[columns].sum(axis=1)

    for direction in DIRECTIONS:
        for turn in TURNS:
            measures[f"{direction} {turn}"] = interval_movements[
                f"{direction}_{turn}"
            ]

    return measures


def measure_group(measure_name: str) -> str:
    """Return the dashboard group for a traffic measure."""

    if measure_name == "Total Motor Vehicles":
        return "Overall total"
    if measure_name.endswith(" Total"):
        return "Direction total"
    if measure_name.startswith("All "):
        return "Turn total"
    return "Direction and turn"


def _read_required_columns(file_path: Path) -> pd.DataFrame:
    """Read only columns required by the movement analysis."""

    required = ["Time", "Section", "Quality_Flag", *MOVEMENT_COLUMNS]
    header = pd.read_csv(file_path, nrows=0)
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path.name}: {', '.join(missing)}"
        )
    return pd.read_csv(file_path, usecols=required)


def load_location(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load one intersection and remove every incomplete or invalid local date."""

    data = pd.concat(
        [_read_required_columns(file_path) for file_path in sorted(files)],
        ignore_index=True,
    )
    data = data.loc[data["Section"].isin(MOTOR_SECTIONS)].copy()
    data["Quality_Flag"] = data["Quality_Flag"].astype("string").str.strip()

    data["Time"] = pd.to_datetime(
        data["Time"], errors="coerce", utc=True, format="mixed"
    )
    if data["Time"].isna().any():
        raise ValueError("At least one cleaned timestamp could not be read.")

    data["Local_Date"] = data["Time"].dt.tz_convert(LOCAL_TIME_ZONE).dt.date
    data[MOVEMENT_COLUMNS] = data[MOVEMENT_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    years = sorted({date.year for date in data["Local_Date"]})
    if len(years) != 1:
        raise ValueError(f"Expected one local calendar year, found {years}.")

    bad_dates: set = set()
    duplicate_rows = data.duplicated(subset=["Time", "Section"], keep=False)
    bad_dates.update(data.loc[duplicate_rows, "Local_Date"])

    invalid_quality = data["Quality_Flag"].ne("Valid")
    bad_dates.update(data.loc[invalid_quality, "Local_Date"])

    invalid_values = data[MOVEMENT_COLUMNS].isna().any(axis=1)
    bad_dates.update(data.loc[invalid_values, "Local_Date"])

    expected_intervals = expected_intervals_by_date(years[0])
    row_counts = data.groupby("Local_Date").size()
    for local_date, interval_count in expected_intervals.items():
        expected_rows = int(interval_count) * len(MOTOR_SECTIONS)
        if row_counts.get(local_date, 0) != expected_rows:
            bad_dates.add(local_date)

    section_counts = data.groupby("Time")["Section"].nunique()
    incomplete_times = section_counts.loc[section_counts.ne(len(MOTOR_SECTIONS))].index
    if len(incomplete_times):
        bad_dates.update(incomplete_times.tz_convert(LOCAL_TIME_ZONE).date)

    all_dates = set(expected_intervals.index)
    valid_dates = sorted(all_dates - bad_dates)

    # The model receives only records explicitly marked Valid.
    valid_rows = data.loc[
        data["Local_Date"].isin(valid_dates)
        & data["Quality_Flag"].eq("Valid")
    ].copy()

    interval_movements = valid_rows.groupby("Time")[MOVEMENT_COLUMNS].sum(
        min_count=len(MOTOR_SECTIONS)
    )
    interval_measures = add_movement_measures(interval_movements)

    local_dates = pd.to_datetime(
        interval_measures.index.tz_convert(LOCAL_TIME_ZONE).date
    )
    daily_measures = interval_measures.groupby(local_dates).sum()
    daily_measures.index.name = "Date"

    flag_counts = data["Quality_Flag"].value_counts()
    quality = {
        "Calendar_Year": years[0],
        "Total_Dates": len(all_dates),
        "Valid_Dates": len(valid_dates),
        "Excluded_Dates": len(bad_dates),
        "Excluded_Date_List": ", ".join(
            date.isoformat() for date in sorted(bad_dates)
        ),
        "Valid_Rows": int(flag_counts.get("Valid", 0)),
        "Partial_Zero_Rows": int(flag_counts.get("Partial Zero", 0)),
        "All_Zero_Day_Rows": int(flag_counts.get("All Zero Day", 0)),
    }
    return interval_measures, daily_measures, quality


def load_all_locations(
    folder: Path = CLEANED_DATA_DIR,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Load every cleaned intersection with the same strict validity rule."""

    locations: dict[str, dict[str, pd.DataFrame]] = {}
    quality_rows = []

    for location, files in discover_cleaned_files(folder).items():
        intervals, daily, quality = load_location(files)
        locations[location] = {"intervals": intervals, "daily": daily}
        quality.update({"Location": location, "CSV_Files": len(files)})
        quality_rows.append(quality)

    quality_report = pd.DataFrame(quality_rows)[
        [
            "Location",
            "CSV_Files",
            "Calendar_Year",
            "Total_Dates",
            "Valid_Dates",
            "Excluded_Dates",
            "Excluded_Date_List",
            "Valid_Rows",
            "Partial_Zero_Rows",
            "All_Zero_Day_Rows",
        ]
    ].sort_values("Location")
    return locations, quality_report
