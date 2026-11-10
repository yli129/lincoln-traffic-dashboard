from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_utils import load_intersection


DATA_FOLDER = Path(__file__).resolve().parent
CLEANED_FOLDER = DATA_FOLDER / "cleaned"

MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles"
]

def get_intersection_files():
    # Find and group cleaned CSV files by intersection
    files = sorted(CLEANED_FOLDER.glob("*_Binned_*.csv"))

    intersection_files = {}

    for file in files:
        name = file.name.split("_Binned_")[0]

        if name not in intersection_files:
            intersection_files[name] = []

        intersection_files[name].append(file)

    return intersection_files


def prepare_data(a_files, b_files, section):
    # Select sections based on user's choice
    if section == "All Motor Vehicles":
        sections = MOTOR_SECTIONS
    else:
        sections = [section]
    # Load, aggregate, and align two intersections
    a = load_intersection(a_files, sections)
    b = load_intersection(b_files, sections)
    #print(a.head())
    #print(b.head())

    a["Time"] = pd.to_datetime(a["Time"], utc=True)
    b["Time"] = pd.to_datetime(b["Time"], utc=True)

    # Aggregate traffic by time and section
    a = a.groupby(["Time", "Section"])["Total_Vehicles"].sum().reset_index()
    b = b.groupby(["Time", "Section"])["Total_Vehicles"].sum().reset_index()
    #print(a.head())
    #print(b.head())
    # Combine and 
    combined = pd.merge(a, b, on=["Time", "Section"], how="inner", suffixes=("_A", "_B"))
    #print(combined.head())
    #print(combined.shape)

    # Select different vehicle type based on user's choice
    if section == "All Motor Vehicles":
        # Count distinct sections at each timestamp
        counts = combined.groupby("Time")["Section"].nunique()

        # Find timestamps containing all five sections
        valid_times = counts[counts == len(MOTOR_SECTIONS)].index

        # Keep only complete timestamps
        combined = combined[combined["Time"].isin(valid_times)]

        selected = combined.groupby("Time")[["Total_Vehicles_A", "Total_Vehicles_B"]].sum().reset_index()
    else:
        selected = combined.copy()

    selected = selected.rename(columns={"Total_Vehicles_A": "A", "Total_Vehicles_B": "B"})

    selected = selected.sort_values("Time").reset_index(drop=True)

    return selected 

def calculate_error_curve(selected):
    a = selected["A"].to_numpy(dtype=float)
    b = selected["B"].to_numpy(dtype=float)

    n_max = len(selected["B"])
    N = np.arange(1, n_max+1)

    a_sum = np.cumsum(a)
    b_sum = np.cumsum(b)
    scale_factors = b_sum / a_sum 

    # Convert UTC timestamps to local dates
    selected = selected.copy()

    selected["Date"] = (
        selected["Time"]
        .dt.tz_convert("America/Chicago")
        .dt.date
    )

    # Calculate daily volume 
    daily = selected.groupby("Date")[["A","B"]].sum()

    # Calculate actual average annual daily traffic volume in intersection B
    actual_aadt = daily["B"].mean()

    # Calculate estimate average annual daily traffic volume in intersection B
    estimated_aadt = scale_factors * daily["A"].mean()

    aadt_error = (np.abs(estimated_aadt -  actual_aadt)) / actual_aadt * 100

    results = pd.DataFrame({
        "N": N,
        "Scale_Factor": scale_factors,
        "Estimated_AADT": estimated_aadt,
        "AADT_Error_Pct": aadt_error
    })
    return results

def main():
    # Find intersections
    intersection_files = get_intersection_files()
    names = list(intersection_files.keys())

    if len(names) < 2:
        print("At least two intersections are required.")
        return

    print("\nAvailable intersections:")

    for i, name in enumerate(names, start=1):
        print(f"{i}. {name.replace('_', ' ')}")

    # Let user select reference and target
    try:
        a_index = int(input("\nSelect reference (A): ")) - 1
        b_index = int(input("Select target (B): ")) - 1
    except ValueError:
        print("Please enter a number.")
        return

    if not (
        0 <= a_index < len(names)
        and 0 <= b_index < len(names)
    ):
        print("Invalid selection.")
        return

    if a_index == b_index:
        print("Please select two different intersections.")
        return

    a_name = names[a_index]
    b_name = names[b_index]

    a_files = intersection_files[a_name]
    b_files = intersection_files[b_name]

    section_options = ["All Motor Vehicles"] + MOTOR_SECTIONS

    for i, name in enumerate(section_options, 1):
        print(f"{i}. {name}")

    choice = int(input("Select a section: "))

    section = section_options[choice - 1]

    print(f"\nReference: {a_name}")
    print(f"Target: {b_name}")

    # Prepare data
    combined = prepare_data(a_files, b_files, section)

    # Calculate errors
    results = calculate_error_curve(combined)

    print(results.head())
    print(results.shape)

    # Save results
    results.to_csv(
        DATA_FOLDER / "error_curve_results.csv",
        index=False
    )

    # Plot Error Curve
    plt.figure(figsize=(10, 5))

    plt.plot(
        results["N"],
        results["AADT_Error_Pct"]
    )

    for threshold, color in [(10, "red"), (5, "green")]:
        acceptable = results[results["AADT_Error_Pct"] <= threshold]

        if not acceptable.empty:
            n = int(acceptable.iloc[0]["N"])

            plt.axvline(
                x=n,
                color=color,
                linestyle=":",
                label=f"{threshold}%: N={n}"
            )

    plt.legend()

    plt.xlabel("Number of Observations (N)")
    plt.ylabel("AADT Error (%)")
    plt.title("AADT Error vs. Observation Duration")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
