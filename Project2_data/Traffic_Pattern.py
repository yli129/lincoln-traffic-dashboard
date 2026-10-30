import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans

# Folder containing cleaned CSV files
data_folder = Path(__file__).parent / "cleaned"

def load_intersection(files, section):
    dataframes = []
    # Load all files which is belong to same intersection
    for file in files:
        df = pd.read_csv(file)

        # Select target section
        df = df[df["Section"] == section].copy()

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

def build_matrix(df):
    data = df.copy()
    # Extract local date and time slots
    data["Date"] = data["Time"].str[:10]
    data["Time_Slot"] = data["Time"].str[11:16]

    # Check duplicate date/time slots
    duplicates = data.duplicated(
        subset=["Date", "Time_Slot"],
        keep=False
    )
    if duplicates.any():
        # Find dates containing duplicate time slots
        duplicate_dates = data.loc[duplicates, "Date"].unique()

        print("Duplicate dates:", duplicate_dates)

        # Exclude these dates
        data = data[~data["Date"].isin(duplicate_dates)].copy()

    # Build Date × Time matrix
    matrix = data.pivot(
        index="Date",
        columns="Time_Slot",
        values="Total_Vehicles"
    )

    # Define all 96 expected time slots
    time_slots = pd.date_range(
        "00:00",
        "23:45",
        freq="15min"
    ).strftime("%H:%M")

    # Keep a consistent column order
    matrix = matrix.reindex(columns=time_slots)

    # Keep only complete days
    matrix = matrix.dropna()

    return matrix

def combine_matrices(matrices):
    combined = pd.concat(matrices, axis=1, join="inner")
    return combined

def preprocess_matrix(combined):
    X = combined.to_numpy(dtype=float)
    X_processed = X - X.mean(axis=1, keepdims=True)
    return X_processed

def run_svd(X_processed):
    # Perform SVD
    U, S, Vt = np.linalg.svd(X_processed, full_matrices=False)

    return U, S, Vt

def get_pca_scores(U, S, k=None):
    if k is None:
        k = len(S)
    scores = U[:, :k] * S[:k]
    return scores

def run_clustering(scores,n_clusters):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(scores)
    return labels

def get_cluster_profiles(combined, labels):
    result = pd.DataFrame({
        "Date": combined.index,
        "Cluster": labels
    })

    counts = result["Cluster"].value_counts().sort_index()

    traffic = combined.copy()

    profiles = traffic.groupby(labels).mean()

    result["Weekday"] = pd.to_datetime(result["Date"]).dt.day_name()

    weekday_counts = pd.crosstab(result["Weekday"], result["Cluster"])

    return result, counts, profiles,weekday_counts

def run_analysis(matrices,n_clusters):
    combined = combine_matrices(matrices)

    X_processed = preprocess_matrix(combined)

    U, S, Vt = run_svd(X_processed)

    scores = get_pca_scores(U, S, k=None)

    labels = run_clustering(scores,n_clusters)

    result, counts, profiles,weekday_counts = get_cluster_profiles(combined, labels)

    return result, counts, profiles,weekday_counts

def plot_cluster_profiles(profiles):
    fig, ax = plt.subplots(figsize=(12, 5))

    times = profiles.columns.get_level_values(1)

    for cluster_id, row in profiles.iterrows():
        ax.plot(
            range(len(times)),
            row.values,
            label=f"Cluster {cluster_id}"
        )
    ax.set_xticks(range(0, 96, 8))
    ax.set_xticklabels(times[::8])

    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Average Traffic Volume")
    ax.legend()
    plt.show()

def main():
    data_folder = Path(__file__).parent / "cleaned"
    files_A = list(
        data_folder.glob("40th_Street_and_South_Street_Binned_*.csv")
    )

    df_A = load_intersection(files_A, "Single-Unit Trucks")

    matrix_A = build_matrix(df_A)

    # Create matrices Dictionary
    matrices = {"Intersection A": matrix_A}

    # Run analysis
    result, counts, profiles, weekday_counts = run_analysis(matrices,n_clusters=2)

    plot_cluster_profiles(profiles)

    print(counts)
    print(profiles.shape)
    print(weekday_counts)

if __name__ == "__main__":
    main()
