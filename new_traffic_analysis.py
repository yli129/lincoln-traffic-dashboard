"""Traffic Dashboard analysis functions."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Columns for Right, Thru, Left, and U-Turn.
# Pedestrian columns are not included.
VEHICLE_COLUMNS = [
    "1", "2", "3", "4",
    "7", "8", "9", "10",
    "13", "14", "15", "16",
    "19", "20", "21", "22",
]

# A complete day has 96 15-minute time slots.
TIME_SLOTS = pd.date_range(
    "00:00",
    "23:45",
    freq="15min",
).strftime("%H:%M")


def load_intersection(files, section):
    """Load one vehicle section from one intersection."""

    if not files:
        raise ValueError(
            "No CSV files found for the selected intersection."
        )

    dataframes = []

    # Read every CSV file for this intersection.
    for file in sorted(files):

        df = pd.read_csv(file)

        # Keep only the selected section and valid rows.
        df = df.loc[
            (df["Section"] == section)
            & (df["Quality_Flag"] == "Valid")
        ].copy()

        # Check that all vehicle count columns exist.
        missing_columns = [
            column
            for column in VEHICLE_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing count columns in {file.name}: "
                f"{missing_columns}"
            )

        # Change the vehicle count columns into numbers.
        df[VEHICLE_COLUMNS] = df[
            VEHICLE_COLUMNS
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Add all vehicle movements in each 15-minute interval.
        df["Total_Vehicles"] = df[
            VEHICLE_COLUMNS
        ].sum(
            axis=1,
            min_count=1,
        )

        dataframes.append(df)

    # Combine all CSV files from this intersection.
    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    if combined.empty:
        raise ValueError(
            f"No valid records found for section '{section}'."
        )

    return combined


def build_matrix(df):
    """Create a Date by 15-minute Time Slot matrix."""

    data = df.copy()

    # Get the local date and local time.
    data["Date"] = data["Time"].str[:10]
    data["Time_Slot"] = data["Time"].str[11:16]

    # Find repeated local times.
    # This can happen when daylight saving time ends.
    duplicates = data.duplicated(
        subset=["Date", "Time_Slot"],
        keep=False,
    )

    duplicate_dates = sorted(
        data.loc[
            duplicates,
            "Date",
        ].unique().tolist()
    )

    # Remove dates containing repeated local times.
    if duplicate_dates:
        data = data.loc[
            ~data["Date"].isin(duplicate_dates)
        ].copy()

    # Each row represents one date.
    # Each column represents one 15-minute time slot.
    matrix = data.pivot(
        index="Date",
        columns="Time_Slot",
        values="Total_Vehicles",
    )

    # Put all 96 time slots in the correct order.
    matrix = matrix.reindex(
        columns=TIME_SLOTS
    )

    # Remove incomplete days.
    matrix = matrix.dropna().sort_index()

    if matrix.empty:
        raise ValueError(
            "No complete valid days remain "
            "after quality filtering."
        )

    return matrix, duplicate_dates


def combine_matrices(matrices):
    """Combine the selected intersections by shared dates."""

    if not matrices:
        raise ValueError(
            "Select at least one intersection."
        )

    combined = pd.concat(
        matrices,
        axis=1,
        join="inner",
    )

    if combined.empty:
        raise ValueError(
            "The selected intersections share "
            "no complete valid dates."
        )

    return combined


def preprocess_matrix(combined):
    """Subtract each day's average traffic volume."""

    X = combined.to_numpy(dtype=float)

    X_processed = (
        X
        - X.mean(
            axis=1,
            keepdims=True,
        )
    )

    return X_processed


def run_svd(X_processed):
    """Perform singular value decomposition."""

    return np.linalg.svd(
        X_processed,
        full_matrices=False,
    )


def get_pca_scores(U, S, k=None):
    """Calculate the scores used by K-means."""

    if k is None:
        k = len(S)

    scores = U[:, :k] * S[:k]

    return scores


def run_clustering(scores, n_clusters):
    """Group dates with similar traffic patterns."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(scores)

    return labels


def get_cluster_profiles(combined, labels):
    """Create the cluster results and average profiles."""

    result = pd.DataFrame({
        "Date": combined.index,
        "Cluster": labels,
    })

    # Count the number of dates in each cluster.
    counts = (
        result["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Calculate the average traffic pattern
    # for each cluster.
    profiles = combined.groupby(labels).mean()

    # Add the weekday name for each date.
    result["Weekday"] = pd.to_datetime(
        result["Date"]
    ).dt.day_name()

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Count weekdays in each cluster.
    weekday_counts = pd.crosstab(
        result["Weekday"],
        result["Cluster"],
    )

    weekday_counts = weekday_counts.reindex(
        index=weekdays,
        columns=counts.index,
        fill_value=0,
    )

    return (
        result,
        counts,
        profiles,
        weekday_counts,
    )


def run_analysis(matrices, n_clusters):
    """Run the complete clustering analysis."""

    # Combine the selected intersections.
    combined = combine_matrices(matrices)

    # Make sure K is valid.
    if not 2 <= n_clusters < len(combined):
        raise ValueError(
            f"K must be between 2 and "
            f"{len(combined) - 1} for "
            f"{len(combined)} valid days."
        )

    # Center the daily traffic data.
    X_processed = preprocess_matrix(combined)

    # Perform SVD.
    U, S, Vt = run_svd(X_processed)

    # Calculate the scores.
    scores = get_pca_scores(
        U,
        S,
        k=None,
    )

    # Run K-means clustering.
    labels = run_clustering(
        scores,
        n_clusters,
    )

    # Return all results.
    return get_cluster_profiles(
        combined,
        labels,
    )