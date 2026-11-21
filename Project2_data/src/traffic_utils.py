#==============================================================================================================
# Traffic Data Utilities
#
# Shared functions for traffic pattern analysis and estimation scripts.
#==============================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def load_intersection(files, section):
    dataframes = []
    # Load all files which is belong to same intersection
    for file in files:
        df = pd.read_csv(file)

        # Select target section
        df = df[df["Section"].isin(section)].copy()

        # Keep only valid samples
        df = df[df["Quality_Flag"] == "Valid"].copy()

        # Select vehicle movements
        movements = ["Right", "Thru", "Left", "U-Turn"]

        vehicle_columns = [
            col for col in df.columns
            if col.split("_")[-1] in movements
        ]
        
        # Calculate total vehicles per time interval
        df["Total_Vehicles"] = df[vehicle_columns].sum(axis=1)

        # Save the processed DataFrame
        dataframes.append(df)

    # Combine all CSV files vertically
    combined = pd.concat(dataframes, ignore_index=True)

    return combined