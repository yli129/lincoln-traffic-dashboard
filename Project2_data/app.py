"""Run with: streamlit run app.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from traffic_analysis import build_matrix, load_intersection, run_analysis


DATA_FOLDER = Path(__file__).resolve().parent / "cleaned"
VEHICLE_SECTIONS = [
    "Motorized Vehicles",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Bicycles",
]


def discover_intersections(folder):
    """Map each intersection filename prefix to its CSV files."""
    groups = {}
    for file in sorted(folder.glob("*_Binned_*.csv")):
        name = file.name.split("_Binned_", 1)[0]
        groups.setdefault(name, []).append(file)
    return groups


def readable_name(name):
    return name.replace("_and_", " & ").replace("_", " ")


def plot_profiles(profiles, intersection):
    """A separate cluster comparison chart for each selected intersection."""
    subset = profiles[intersection]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    times = subset.columns
    x = np.arange(len(times))
    for cluster_id, row in subset.iterrows():
        ax.plot(x, row.to_numpy(), label=f"Cluster {cluster_id}", linewidth=1.8)
    ax.set_xticks(x[::8], times[::8], rotation=0)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Average count per 15 minutes")
    ax.set_title(readable_name(intersection))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weekdays(weekday_counts):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    weekday_counts.plot.bar(ax=ax, width=0.8)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Number of days")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Lincoln Traffic Dashboard", page_icon="🚦", layout="wide")
st.title("Lincoln Traffic Dashboard")
st.caption("Daily traffic pattern clustering · 15-minute traffic counts")

if not DATA_FOLDER.is_dir():
    st.error(f"Missing data folder: {DATA_FOLDER}. Create a 'cleaned' folder next to app.py and place the cleaned CSVs inside.")
    st.stop()

available = discover_intersections(DATA_FOLDER)
if not available:
    st.error("No files matching '*_Binned_*.csv' were found inside the cleaned folder.")
    st.stop()

with st.sidebar:
    st.header("Analysis settings")
    selected = st.multiselect(
        "Intersections",
        options=list(available),
        default=list(available)[:1],
        format_func=readable_name,
    )
    section = st.selectbox("Vehicle section", options=VEHICLE_SECTIONS, index=1)
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=2)
    analyze = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("The current method uses row centering and all SVD components, matching your original Python implementation.")

st.info("Select one or more intersections and a vehicle section, then click **Run analysis**.")

if analyze:
    if not selected:
        st.warning("Please select at least one intersection.")
        st.stop()

    try:
        matrices = {}
        excluded_days = {}
        for name in selected:
            df = load_intersection(available[name], section)
            matrix, duplicates = build_matrix(df)
            matrices[name] = matrix
            excluded_days[name] = duplicates
        result, counts, profiles, weekday_counts = run_analysis(matrices, n_clusters=k)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to complete analysis: {exc}")
        st.stop()

    st.subheader("Cluster overview")
    a, b, c = st.columns(3)
    a.metric("Valid dates", len(result))
    b.metric("Intersections", len(selected))
    c.metric("Clusters", len(counts))
    st.dataframe(counts.rename("Number of days").to_frame(), use_container_width=True)

    st.subheader("Average daily traffic profiles")
    st.caption("Plots display original average vehicle counts, not centered SVD scores.")
    for name in selected:
        fig = plot_profiles(profiles, name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Day-of-week distribution")
    fig = plot_weekdays(weekday_counts)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(weekday_counts, use_container_width=True)

    st.subheader("Dates by cluster")
    show_cluster = st.selectbox("View cluster", options=[int(v) for v in counts.index])
    view = result.loc[result["Cluster"] == show_cluster, ["Date", "Weekday", "Cluster"]]
    st.caption(f"Cluster {show_cluster}: {len(view)} dates")
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button(
        "Download all labeled dates (CSV)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="traffic_cluster_dates.csv", mime="text/csv"
    )

    with st.expander("Data quality details"):
        for name, days in excluded_days.items():
            st.write(f"**{readable_name(name)}**: {len(days)} date(s) excluded due to duplicate local time slots")
            if days:
                st.write(", ".join(days))
        st.caption("Days missing any of the 96 time slots are also excluded; multiple intersections are aligned on common valid dates.")
        if len(selected) > 1:
            st.warning("For multiple intersections, the current row-centering step uses a single daily mean across all selected intersections. Consider independent per-intersection centering when interpreting joint patterns.")
