#==============================================================================================================
# Traffic Data Utilities
# Shared utility functions for traffic data analysis.
#==============================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Motor vehicle sections
MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles"
]

# Individual section
INDIVIDUAL_SECTION = MOTOR_SECTIONS + [
    "Bicycles",
    "Pedestrians"
]

# Combined motor vehicle option
ALL_MOTOR_VEHICLES = "All Motor Vehicles"

# User selection options
SECTION_OPTIONS = [ALL_MOTOR_VEHICLES] + INDIVIDUAL_SECTION

DIRECTIONS = [
    "Southbound", "Westbound",
    "Northbound", "Eastbound"
]

VEHICLE_MOVEMENTS = ["Right", "Thru", "Left", "U-Turn"]

def load_intersection(files, section):
    """
    Load and combine cleaned traffic data for one intersection.

    Parameters
    ----------
    files : list
        List of CSV file paths for the same intersection.

    section : list
        Sections to include, e.g., ["Lights", "Buses"].

    Returns
    -------
    pd.DataFrame
        Combined traffic data containing only valid records,
        all original movement columns, and Total_Vehicles.

    Usage
    -----
    df = load_intersection(files, ["Lights"])
    """
    dataframes = []
    # Load all files which is belong to same intersection
    for file in files:
        df = pd.read_csv(file)

        # Select target section
        df = df[df["Section"].isin(section)].copy()

        # Keep only valid samples
        df = df[df["Quality_Flag"] == "Valid"].copy()

        # Select vehicle movements
        vehicle_columns = [
            col for col in df.columns
            if col.split("_")[-1] in VEHICLE_MOVEMENTS
        ]
        
        # Calculate total vehicles per time interval
        df["Total_Vehicles"] = df[vehicle_columns].sum(axis=1)

        # Save the processed DataFrame
        dataframes.append(df)

    # Combine all CSV files vertically
    combined = pd.concat(dataframes, ignore_index=True)

    return combined

    
def _expected_bins(date, bin_minutes=15, tz="America/Chicago"):
    """Number of bins a local calendar day should have (92/100 on DST days)."""
    start = pd.Timestamp(date).tz_localize(tz)
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((end - start) / pd.Timedelta(minutes=bin_minutes))


def aggregate_daily(df, bin_minutes=15, require_complete=True, verbose=True):
    """
    Aggregate binned traffic counts into daily totals (local time).

    Days are kept only if every expected time bin is present. Because
    load_intersection() removes non-Valid bins, an incomplete day would
    otherwise be silently under-counted and distort the factors.

    Returns
    -------
    pd.DataFrame
        One row per Section and Date with all numeric traffic columns.
    """
    df = df.copy()

    local = pd.to_datetime(df["Time"], utc=True).dt.tz_convert("America/Chicago")
    df["Date"] = local.dt.date

    # Guard against double counting if files overlap
    df = df.drop_duplicates(subset=["Section", "Time"])

    movement_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col != "N_Bins"
    ]

    daily = (
        df.groupby(["Section", "Date"])[movement_columns]
        .sum(min_count=1)
        .reset_index()
    )

    n_bins = (
        df.groupby(["Section", "Date"])["Time"].nunique()
        .rename("N_Bins").reset_index()
    )
    daily = daily.merge(n_bins, on=["Section", "Date"])

    expected = {d: _expected_bins(d, bin_minutes) for d in daily["Date"].unique()}
    daily["Expected_Bins"] = daily["Date"].map(expected)
    complete = daily["N_Bins"] == daily["Expected_Bins"]

    if verbose:
        print(f"  aggregate_daily: dropped {(~complete).sum()} incomplete "
              f"section-days out of {len(daily)}")

    if require_complete:
        daily = daily[complete]

    return daily.drop(columns=["N_Bins", "Expected_Bins"]).reset_index(drop=True)


def average_daily_traffic(daily, start_date, end_date):
    """
    Calculate average daily traffic for a selected period.

    Parameters
    ----------
    daily : pd.DataFrame
        Daily traffic data from aggregate_daily().
    start_date, end_date : str
        Start and end dates in YYYY-MM-DD format.

    Returns
    -------
    pd.Series
        Average daily traffic for each traffic column.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    selected = daily[
        (daily["Date"]>= start) &
        (daily["Date"]<= end)
    ]

    traffic_columns = selected.select_dtypes(include="number").columns

    adt = selected[traffic_columns].mean()

    return adt
# 
def calculate_factor(sample_adt, target_adt):
    return target_adt / sample_adt.replace(0, float("nan"))

# Calculate the error 
def estimate_adt(sample_adt, factor):
    return sample_adt * factor