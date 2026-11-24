"""Shared project paths and traffic-data constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LOCAL_TIME_ZONE = "America/Chicago"

MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]

DIRECTIONS = ["Southbound", "Westbound", "Northbound", "Eastbound"]
TURNS = ["Right", "Thru", "Left", "U-Turn"]

MOVEMENT_COLUMNS = [
    f"{direction}_{turn}"
    for direction in DIRECTIONS
    for turn in TURNS
]

# The error curve evaluates every whole-day calibration length in this range.
# Ninety days shows longer-term stability while remaining well below a full year.
ERROR_CURVE_MAX_DAYS = 90
