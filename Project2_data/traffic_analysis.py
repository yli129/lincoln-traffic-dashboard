"""Traffic Dashboard analysis functions, adapted from the original script."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


MOVEMENTS = ("Right", "Thru", "Left", "U-Turn")
TIME_SLOTS = pd.date_range("00:00", "23:45", freq="15min").strftime("%H:%M")


def load_intersection(files, section):
    """Load valid records for one intersection and vehicle section."""
    if not files:
        raise ValueError("No CSV files found for the selected intersection.")

    dataframes = []
    for file in sorted(files):
        df = pd.read_csv(file)
        df = df.loc[(df["Section"] == section) & (df["Quality_Flag"] == "Valid")].copy()
        vehicle_columns = [
            col for col in df.columns if col.split("_")[-1] in MOVEMENTS
        ]
        if not vehicle_columns:
            raise ValueError(f"No vehicle movement columns found in {file.name}.")
        df["Total_Vehicles"] = df[vehicle_columns].sum(axis=1)
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)
    if combined.empty:
        raise ValueError(f"No valid records found for section '{section}'.")
    return combined


def build_matrix(df):
    data = df.copy()
    data["Date"] = data["Time"].str[:10]
    data["Time_Slot"] = data["Time"].str[11:16]

    duplicates = data.duplicated(subset=["Date", "Time_Slot"], keep=False)
    duplicate_dates = sorted(data.loc[duplicates, "Date"].unique().tolist())
    if duplicate_dates:
        data = data.loc[~data["Date"].isin(duplicate_dates)].copy()

    matrix = data.pivot(index="Date", columns="Time_Slot", values="Total_Vehicles")
    matrix = matrix.reindex(columns=TIME_SLOTS).dropna().sort_index()
    if matrix.empty:
        raise ValueError("No complete valid days remain after quality filtering.")
    return matrix, duplicate_dates


def combine_matrices(matrices):
    if not matrices:
        raise ValueError("Select at least one intersection.")
    combined = pd.concat(matrices, axis=1, join="inner")
    if combined.empty:
        raise ValueError("The selected intersections share no complete valid dates.")
    return combined


def preprocess_matrix(combined):
    """Original method: center every date across its selected features."""
    X = combined.to_numpy(dtype=float)
    return X - X.mean(axis=1, keepdims=True)


def run_svd(X_processed):
    return np.linalg.svd(X_processed, full_matrices=False)


def get_pca_scores(U, S, k=None):
    if k is None:
        k = len(S)
    return U[:, :k] * S[:k]


def run_clustering(scores, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return model.fit_predict(scores)


def get_cluster_profiles(combined, labels):
    result = pd.DataFrame({"Date": combined.index, "Cluster": labels})
    counts = result["Cluster"].value_counts().sort_index()
    profiles = combined.groupby(labels).mean()
    result["Weekday"] = pd.to_datetime(result["Date"]).dt.day_name()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = pd.crosstab(result["Weekday"], result["Cluster"])
    weekday_counts = weekday_counts.reindex(index=weekdays, columns=counts.index, fill_value=0)
    return result, counts, profiles, weekday_counts


def run_analysis(matrices, n_clusters):
    combined = combine_matrices(matrices)
    if not 2 <= n_clusters < len(combined):
        raise ValueError(f"K must be between 2 and {len(combined)-1} for {len(combined)} valid days.")
    X_processed = preprocess_matrix(combined)
    U, S, Vt = run_svd(X_processed)
    scores = get_pca_scores(U, S, k=None)
    labels = run_clustering(scores, n_clusters)
    return get_cluster_profiles(combined, labels)
