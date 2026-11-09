"""A presentation-ready Streamlit dashboard for annual traffic estimation."""

from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None


# This page can run by itself or from Streamlit's pages folder.
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER.parent if SCRIPT_FOLDER.name == "pages" else SCRIPT_FOLDER

RESULTS_FILE = DATA_FOLDER / "annual_estimation_results.csv"
PREDICTIONS_FILE = DATA_FOLDER / "annual_daily_predictions.csv"
SEASONAL_RESULTS_FILE = DATA_FOLDER / "seasonal_estimation_results.csv"
SEASONAL_SUMMARY_FILE = DATA_FOLDER / "seasonal_error_summary.csv"
CLEANED_FOLDER = DATA_FOLDER / "cleaned"

REFERENCE_LOCATION = "40th Street & South Street"
TARGET_LOCATION = "South Street & Normal Boulevard"
REFERENCE_DATA_NAME = "40th Street and South Street"
TARGET_DATA_NAME = "South Street and Normal Boulevard"

# Approximate intersection-center coordinates for the small study map.
REFERENCE_LAT = 40.7915
REFERENCE_LON = -96.6632
TARGET_LAT = 40.7914
TARGET_LON = -96.6653

DURATION_ORDER = [
    "15 minutes",
    "1 hour",
    "1 day",
    "2 days",
    "7 days",
    "14 days",
    "30 days",
]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]
PROFILE_SECTIONS = MOTOR_SECTIONS + ["Bicycles", "Pedestrians"]
PROFILE_CHART_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
]
PROFILE_SECTION_LABELS = {
    "Lights": "Light vehicles",
    "Single-Unit Trucks": "Single-unit trucks",
    "Articulated Trucks": "Articulated trucks",
    "Buses": "Buses",
}
VEHICLE_COLUMNS = [
    "1", "2", "3", "4",
    "7", "8", "9", "10",
    "13", "14", "15", "16",
    "19", "20", "21", "22",
]

# These descriptions keep the dashboard readable for a nontechnical audience.
DURATION_NOTES = {
    "15 minutes": "one 15-minute observation",
    "1 hour": "four 15-minute observations",
    "1 day": "96 observations from one day",
    "2 days": "192 observations from two days",
    "7 days": "one complete week",
    "14 days": "two complete weeks",
    "30 days": "approximately one month",
}

# The fixed-start experiment begins on January 1, 2025.
CALIBRATION_LENGTHS = {
    "15 minutes": pd.Timedelta(minutes=15),
    "1 hour": pd.Timedelta(hours=1),
    "1 day": pd.Timedelta(days=1),
    "2 days": pd.Timedelta(days=2),
    "7 days": pd.Timedelta(days=7),
    "14 days": pd.Timedelta(days=14),
    "30 days": pd.Timedelta(days=30),
}


@st.cache_data
def load_results():
    """Read and organize the four result files created by the model scripts."""
    required_files = [
        RESULTS_FILE,
        PREDICTIONS_FILE,
        SEASONAL_RESULTS_FILE,
        SEASONAL_SUMMARY_FILE,
    ]
    missing_files = [file.name for file in required_files if not file.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing result file(s): " + ", ".join(missing_files)
        )

    results = pd.read_csv(RESULTS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    seasonal = pd.read_csv(SEASONAL_RESULTS_FILE)
    summary = pd.read_csv(SEASONAL_SUMMARY_FILE)

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    seasonal["Start_Date"] = pd.to_datetime(seasonal["Start_Date"])

    # Categorical columns preserve the intended order in charts and tables.
    for table in (results, predictions, seasonal, summary):
        table["Duration"] = pd.Categorical(
            table["Duration"],
            categories=DURATION_ORDER,
            ordered=True,
        )
    seasonal["Season"] = pd.Categorical(
        seasonal["Season"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    return results, predictions, seasonal, summary


@st.cache_data
def load_traffic_profile():
    """Build 2025 vehicle-type and monthly summaries from the cleaned files."""
    files = sorted(CLEANED_FOLDER.glob("*_Binned_*.csv"))
    if not files:
        raise FileNotFoundError("No cleaned traffic CSV files were found.")

    all_sections = []
    needed_columns = ["Time", "Section", "Quality_Flag"] + VEHICLE_COLUMNS

    for file in files:
        data = pd.read_csv(file, usecols=needed_columns)
        data = data.loc[
            data["Section"].isin(PROFILE_SECTIONS)
            & data["Quality_Flag"].eq("Valid")
        ].copy()
        data[VEHICLE_COLUMNS] = data[VEHICLE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        data["Section_Total"] = data[VEHICLE_COLUMNS].sum(
            axis=1,
            min_count=1,
        )

        # Match the annual model: accept a timestamp only when all five
        # motor-vehicle categories are present and valid.
        section_counts = (
            data.loc[data["Section"].isin(MOTOR_SECTIONS)]
            .groupby("Time")["Section"]
            .nunique()
        )
        complete_times = section_counts.loc[
            section_counts.eq(len(MOTOR_SECTIONS))
        ].index
        data = data.loc[data["Time"].isin(complete_times)].copy()

        data["Location"] = (
            file.name.split("_Binned_", 1)[0].replace("_", " ")
        )
        all_sections.append(
            data[["Location", "Time", "Section", "Section_Total"]]
        )

    traffic = pd.concat(all_sections, ignore_index=True)
    traffic["Time"] = pd.to_datetime(
        traffic["Time"],
        errors="coerce",
        utc=True,
    )
    traffic = traffic.dropna(subset=["Time", "Section_Total"])
    traffic["Date"] = (
        traffic["Time"].dt.tz_convert("America/Chicago").dt.tz_localize(None).dt.normalize()
    )

    daily_by_section = (
        traffic.groupby(
            ["Location", "Date", "Section"],
            observed=True,
        )["Section_Total"]
        .sum()
        .reset_index(name="Daily_Count")
    )
    section_aadt = daily_by_section.pivot_table(
        index="Location",
        columns="Section",
        values="Daily_Count",
        aggfunc="mean",
    )

    motor_daily = (
        daily_by_section.loc[daily_by_section["Section"].isin(MOTOR_SECTIONS)]
        .groupby(["Location", "Date"], observed=True)["Daily_Count"]
        .sum()
        .reset_index(name="Motor_Traffic")
    )
    motor_daily["Month"] = motor_daily["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        motor_daily.groupby(["Month", "Location"], observed=True)["Motor_Traffic"]
        .mean()
        .reset_index(name="Average daily traffic")
    )

    # Normalize each vehicle category to its own annual average. This lets
    # small categories and light vehicles share one readable chart.
    monthly_by_section = daily_by_section.loc[
        daily_by_section["Section"].isin(PROFILE_CHART_SECTIONS)
    ].copy()
    monthly_by_section["Month"] = (
        monthly_by_section["Date"].dt.to_period("M").dt.to_timestamp()
    )
    monthly_by_section = (
        monthly_by_section.groupby(
            ["Location", "Month", "Section"],
            observed=True,
        )["Daily_Count"]
        .mean()
        .reset_index(name="Monthly average")
    )
    annual_by_section = (
        daily_by_section.loc[
            daily_by_section["Section"].isin(PROFILE_CHART_SECTIONS)
        ]
        .groupby(["Location", "Section"], observed=True)["Daily_Count"]
        .mean()
        .reset_index(name="Annual average")
    )
    monthly_by_section = monthly_by_section.merge(
        annual_by_section,
        on=["Location", "Section"],
        how="left",
    )
    monthly_by_section["Index"] = (
        monthly_by_section["Monthly average"]
        / monthly_by_section["Annual average"]
        * 100
    )

    return section_aadt, motor_daily, monthly, monthly_by_section


def duration_row(table, duration):
    """Return the single row for one tested duration."""
    matching = table.loc[table["Duration"] == duration]
    if matching.empty:
        raise ValueError(f"No result was found for {duration}.")
    return matching.iloc[0]


def format_result_table(data):
    """Create a clean seasonal table without changing the original data."""
    display = data[
        [
            "Season",
            "Start_Date",
            "Estimated_AADT",
            "AADT_Percent_Error",
            "Daily_MAE",
        ]
    ].copy()
    display.columns = [
        "Season",
        "Start date",
        "Estimated AADT",
        "AADT error",
        "Daily MAE",
    ]
    display["Season"] = display["Season"].astype(str)
    display["Start date"] = display["Start date"].dt.strftime("%b %d, %Y")
    display["Estimated AADT"] = display["Estimated AADT"].map(
        lambda value: f"{value:,.0f}"
    )
    display["AADT error"] = display["AADT error"].map(
        lambda value: f"{value:.2f}%"
    )
    display["Daily MAE"] = display["Daily MAE"].map(
        lambda value: f"{value:,.0f} vehicles/day"
    )
    return display


def make_duration_cards(summary):
    """Create DfT-style cards for the most useful count durations."""
    card_durations = ["1 day", "2 days", "7 days", "14 days", "30 days"]
    previous_duration = {
        "1 day": "1 hour",
        "2 days": "1 day",
        "7 days": "2 days",
        "14 days": "7 days",
        "30 days": "14 days",
    }
    cards = []

    for duration in card_durations:
        current_error = float(duration_row(summary, duration)["Mean_Error"])
        previous = previous_duration[duration]
        previous_error = float(duration_row(summary, previous)["Mean_Error"])
        change = current_error - previous_error

        if change < -0.005:
            change_class = "improved"
            change_text = (
                f"↓ {abs(change):.2f} percentage points vs {previous}"
            )
        elif change > 0.005:
            change_class = "worse"
            change_text = f"↑ {change:.2f} percentage points vs {previous}"
        else:
            change_class = "unchanged"
            change_text = f"Little change vs {previous}"

        recommended_class = " recommended" if duration == "14 days" else ""
        badge = (
            '<div class="duration-badge">Lowest average error</div>'
            if duration == "14 days"
            else '<div class="duration-badge placeholder">Tested option</div>'
        )
        # Keep the HTML on one continuous line. Indented lines after a blank
        # line can be interpreted as a Markdown code block by Streamlit.
        cards.append(
            f'<div class="duration-card{recommended_class}">'
            f"{badge}"
            f'<div class="duration-name">{duration}</div>'
            f'<div class="duration-value">{current_error:.2f}<span>%</span></div>'
            '<div class="duration-unit">four-season mean AADT error</div>'
            f'<div class="duration-change {change_class}">{change_text}</div>'
            "</div>"
        )

    return '<div class="duration-card-grid">' + "".join(cards) + "</div>"


def make_vehicle_cards(section_aadt, motor_daily):
    """Create target-site traffic cards and compare them with the reference."""
    total_by_location = (
        motor_daily.groupby("Location", observed=True)["Motor_Traffic"].mean()
    )
    card_specs = [
        ("All motor vehicles", None),
        ("Light vehicles", "Lights"),
        ("Single-unit trucks", "Single-Unit Trucks"),
        ("Articulated trucks", "Articulated Trucks"),
        ("Buses", "Buses"),
    ]
    cards = []

    for label, section in card_specs:
        if section is None:
            reference_value = float(total_by_location.loc[REFERENCE_DATA_NAME])
            target_value = float(total_by_location.loc[TARGET_DATA_NAME])
        else:
            reference_value = float(section_aadt.loc[REFERENCE_DATA_NAME, section])
            target_value = float(section_aadt.loc[TARGET_DATA_NAME, section])

        if reference_value > 0:
            difference = (target_value / reference_value - 1) * 100
            if difference >= 0:
                detail = f"Up {difference:.1f}% from the reference site"
            else:
                detail = f"Down {abs(difference):.1f}% from the reference site"
        else:
            detail = "Reference-site comparison unavailable"

        cards.append(
            '<div class="vehicle-card">'
            f'<div class="category">{label}</div>'
            f'<div class="value">{target_value:,.0f} '
            '<span class="unit">vehicles/day</span></div>'
            f'<div class="detail">{detail}</div>'
            "</div>"
        )

    return '<div class="vehicle-grid">' + "".join(cards) + "</div>"


def show_location_map():
    """Show the reference and target intersections and the A-to-B direction."""
    locations = pd.DataFrame(
        [
            {
                "label": "A",
                "name": REFERENCE_LOCATION,
                "role": "Reference site · full-year data",
                "lat": REFERENCE_LAT,
                "lon": REFERENCE_LON,
                "color": [11, 60, 93, 230],
            },
            {
                "label": "B",
                "name": TARGET_LOCATION,
                "role": "Target site · short count",
                "lat": TARGET_LAT,
                "lon": TARGET_LON,
                "color": [242, 142, 43, 230],
            },
        ]
    )

    # Streamlit normally installs PyDeck. The fallback still shows both points.
    if pdk is None:
        st.map(locations[["lat", "lon"]], zoom=16, width="stretch")
        st.caption("Blue/reference: A · Orange/target: B · Model direction: A → B")
        return

    route = pd.DataFrame(
        [
            {
                "source_lon": REFERENCE_LON,
                "source_lat": REFERENCE_LAT,
                "target_lon": TARGET_LON,
                "target_lat": TARGET_LAT,
            }
        ]
    )
    direction = pd.DataFrame(
        [
            {
                "lon": (REFERENCE_LON + TARGET_LON) / 2,
                "lat": (REFERENCE_LAT + TARGET_LAT) / 2,
                "text": "A → B",
            }
        ]
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        locations,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=30,
        radius_min_pixels=9,
        radius_max_pixels=18,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
    )
    point_labels = pdk.Layer(
        "TextLayer",
        locations,
        get_position="[lon, lat]",
        get_text="label",
        get_color=[255, 255, 255, 255],
        get_size=16,
        get_alignment_baseline="center",
        pickable=False,
    )
    route_layer = pdk.Layer(
        "ArcLayer",
        route,
        get_source_position="[source_lon, source_lat]",
        get_target_position="[target_lon, target_lat]",
        get_source_color=[11, 60, 93, 210],
        get_target_color=[242, 142, 43, 210],
        get_width=5,
        pickable=False,
    )
    direction_label = pdk.Layer(
        "TextLayer",
        direction,
        get_position="[lon, lat]",
        get_text="text",
        get_color=[20, 43, 58, 255],
        get_size=15,
        get_pixel_offset=[0, -24],
        get_alignment_baseline="center",
        pickable=False,
    )

    view_state = pdk.ViewState(
        latitude=(REFERENCE_LAT + TARGET_LAT) / 2,
        longitude=(REFERENCE_LON + TARGET_LON) / 2,
        zoom=16.7,
        pitch=0,
    )
    deck = pdk.Deck(
        layers=[route_layer, point_layer, point_labels, direction_label],
        initial_view_state=view_state,
        tooltip={"html": "<b>{label} · {name}</b><br>{role}"},
    )
    st.pydeck_chart(deck, height=300, width="stretch")


def make_traffic_chart(chart_data, selected_duration):
    """Layer the calibration window behind the actual and predicted lines."""
    calibration_start = chart_data.index.min().normalize()
    calibration_end = calibration_start + CALIBRATION_LENGTHS[selected_duration]

    long_data = (
        chart_data.reset_index()
        .rename(columns={chart_data.index.name or "index": "Date"})
        .melt(
            id_vars="Date",
            var_name="Series",
            value_name="Vehicles per day",
        )
    )
    calibration = pd.DataFrame(
        {
            "Start": [calibration_start],
            "End": [calibration_end],
            "Window": [f"Known target data: {selected_duration}"],
        }
    )

    shaded_window = (
        alt.Chart(calibration)
        .mark_rect(color="#f4c95d", opacity=0.22)
        .encode(
            x=alt.X("Start:T", title="Date"),
            x2="End:T",
            tooltip=[
                alt.Tooltip("Window:N", title="Calibration"),
                alt.Tooltip("Start:T", title="Starts", format="%b %d, %Y %H:%M"),
                alt.Tooltip("End:T", title="Ends", format="%b %d, %Y %H:%M"),
            ],
        )
    )
    traffic_lines = (
        alt.Chart(long_data)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(
                "Vehicles per day:Q",
                title="Vehicles per day",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Actual traffic", "Predicted traffic"],
                    range=["#0b3c5d", "#f28e2b"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip(
                    "Vehicles per day:Q",
                    title="Traffic",
                    format=",.0f",
                ),
            ],
        )
    )

    return (
        (shaded_window + traffic_lines)
        .properties(height=430)
        .interactive(bind_y=False)
    ), calibration_start, calibration_end


def make_error_chart(summary, shown_durations):
    """Show mean, best, and worst seasonal errors with a highlighted result."""
    error_data = summary.loc[
        summary["Duration"].isin(shown_durations),
        ["Duration", "Mean_Error", "Minimum_Error", "Maximum_Error"],
    ].copy()
    error_data["Duration"] = error_data["Duration"].astype(str)
    error_data = error_data.rename(
        columns={
            "Mean_Error": "Four-season mean",
            "Minimum_Error": "Best seasonal result",
            "Maximum_Error": "Worst seasonal result",
        }
    )
    long_error = error_data.melt(
        id_vars="Duration",
        var_name="Series",
        value_name="AADT error (%)",
    )

    base = alt.Chart(long_error).encode(
        x=alt.X(
            "Duration:N",
            sort=shown_durations,
            title="Amount of target-site data",
        ),
        y=alt.Y(
            "AADT error (%):Q",
            title="AADT percent error",
            scale=alt.Scale(zero=True),
        ),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=[
                    "Four-season mean",
                    "Best seasonal result",
                    "Worst seasonal result",
                ],
                range=["#111827", "#2ca02c", "#d95f02"],
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("Duration:N", title="Target data"),
            alt.Tooltip("Series:N", title="Measure"),
            alt.Tooltip("AADT error (%):Q", title="Error", format=".2f"),
        ],
    )
    lines = base.mark_line(point=True, strokeWidth=2.5)

    highlighted = long_error.loc[
        (long_error["Duration"] == "14 days")
        & (long_error["Series"] == "Four-season mean")
    ]
    highlight_point = (
        alt.Chart(highlighted)
        .mark_point(
            size=230,
            filled=True,
            color="#111827",
            stroke="white",
            strokeWidth=3,
        )
        .encode(
            x=alt.X("Duration:N", sort=shown_durations),
            y="AADT error (%):Q",
            tooltip=[
                alt.Tooltip("Duration:N", title="Recommended"),
                alt.Tooltip("AADT error (%):Q", title="Mean error", format=".2f"),
            ],
        )
    )
    highlight_label = (
        alt.Chart(highlighted)
        .mark_text(
            align="center",
            dy=-18,
            fontWeight="bold",
            color="#111827",
        )
        .encode(
            x=alt.X("Duration:N", sort=shown_durations),
            y="AADT error (%):Q",
            text=alt.value("Lowest mean"),
        )
    )

    return (lines + highlight_point + highlight_label).properties(height=430)


st.set_page_config(
    page_title="How Many Traffic Count Days Are Enough?",
    page_icon="🚦",
    layout="wide",
)

# A small amount of CSS creates hierarchy without changing Streamlit's behavior.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }
        .traffic-hero {
            background: linear-gradient(120deg, #0b3c5d 0%, #176b87 100%);
            border-radius: 20px;
            color: white;
            padding: 1.7rem 2rem 1.8rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(11, 60, 93, 0.14);
        }
        .traffic-hero .eyebrow {
            color: #a9d7e8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }
        .traffic-hero h1 {
            color: white;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.05;
            margin: 0;
        }
        .traffic-hero p {
            color: #e8f4f8;
            font-size: 1.03rem;
            line-height: 1.55;
            margin: 0.85rem 0 0 0;
            max-width: 980px;
        }
        .location-strip {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin: 0.25rem 0 1.15rem 0;
        }
        .location-card {
            background: #f7fafc;
            border: 1px solid #d9e3ea;
            border-radius: 13px;
            padding: 0.85rem 1rem;
        }
        .location-card .label {
            color: #587184;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .location-card .name {
            color: #142b3a;
            font-size: 1.02rem;
            font-weight: 650;
            margin-top: 0.2rem;
        }
        .vehicle-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.65rem 0 1rem 0;
        }
        .vehicle-card {
            background: #f3f4f6;
            border-bottom: 3px solid #d1d5db;
            border-radius: 4px 4px 0 0;
            min-height: 155px;
            padding: 1rem 1.1rem;
        }
        .vehicle-card .category {
            color: #111827;
            font-size: 1rem;
            font-weight: 750;
        }
        .vehicle-card .value {
            color: #0f172a;
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0.55rem 0 0.35rem 0;
        }
        .vehicle-card .unit {
            color: #475569;
            font-size: 0.92rem;
            font-weight: 500;
        }
        .vehicle-card .detail {
            color: #4b5563;
            font-size: 0.89rem;
            line-height: 1.4;
        }
        .duration-card-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.55rem 0 1.25rem 0;
        }
        .duration-card {
            background: #f3f4f6;
            border-bottom: 3px solid #d1d5db;
            border-radius: 5px 5px 0 0;
            min-height: 185px;
            padding: 0.9rem 0.95rem 1rem 0.95rem;
        }
        .duration-card.recommended {
            background: #eaf3f8;
            border: 2px solid #176b87;
            border-bottom-width: 5px;
            border-radius: 8px;
        }
        .duration-badge {
            color: #176b87;
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.035em;
            min-height: 1.1rem;
            text-transform: uppercase;
        }
        .duration-badge.placeholder { color: transparent; }
        .duration-name {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 750;
            margin-top: 0.22rem;
        }
        .duration-value {
            color: #111827;
            font-size: 2.25rem;
            font-weight: 780;
            line-height: 1.05;
            margin-top: 0.55rem;
        }
        .duration-value span {
            font-size: 1rem;
            font-weight: 600;
            margin-left: 0.15rem;
        }
        .duration-unit {
            color: #5d6672;
            font-size: 0.76rem;
            line-height: 1.25;
            margin-top: 0.32rem;
        }
        .duration-change {
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1.3;
            margin-top: 0.68rem;
        }
        .duration-change.improved { color: #18753c; }
        .duration-change.worse { color: #b42318; }
        .duration-change.unchanged { color: #5d6672; }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #dce5eb;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 16px rgba(20, 43, 58, 0.04);
        }
        @media (max-width: 760px) {
            .traffic-hero { padding: 1.35rem; }
            .location-strip { grid-template-columns: 1fr; }
            .vehicle-grid { grid-template-columns: 1fr; }
            .duration-card-grid { grid-template-columns: 1fr; }
            .duration-card { min-height: 0; }
        }
        @media (min-width: 761px) and (max-width: 1100px) {
            .duration-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    results, predictions, seasonal, summary = load_results()
except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Run annual_estimation.py and annual_estimation_seasons.py first, "
        "then reload this page."
    )
    st.stop()


# Global facts used throughout the dashboard.
best_summary = summary.loc[summary["Mean_Error"].idxmin()]
best_duration = str(best_summary["Duration"])
actual_aadt = float(results["Actual_AADT"].iloc[0])
valid_target_days = int(
    predictions.loc[predictions["Duration"] == DURATION_ORDER[0], "Date"].nunique()
)
missing_target_days = 365 - valid_target_days


# Controls live in the sidebar so the report itself stays focused on the story.
with st.sidebar:
    st.header("Explore the experiment")
    selected_duration = st.selectbox(
        f"Amount of {TARGET_LOCATION} data",
        DURATION_ORDER,
        index=DURATION_ORDER.index("14 days"),
    )
    chart_view = st.radio(
        "Daily traffic chart",
        ["7-day average", "Daily values"],
        help="The 7-day average reduces day-to-day noise and makes the yearly pattern easier to see.",
    )
    include_short_tests = st.checkbox(
        "Include 15-minute and 1-hour tests",
        value=False,
        help="These two very short tests have large errors and compress the main seasonal chart.",
    )
    st.divider()
    st.caption(
        "Lower AADT percent error is better. Use the tabs to move from the "
        "main finding to robustness, method, and data quality."
    )


st.markdown(
    f"""
    <section class="traffic-hero">
        <div class="eyebrow">Math 435 · Lincoln traffic study</div>
        <h1>How Many Traffic Count Days Are Enough?</h1>
        <p>
            We estimate a full year of traffic at {escape(TARGET_LOCATION)} by
            combining year-round data from {escape(REFERENCE_LOCATION)} with
            only a short count at the target intersection.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="location-strip">
        <div class="location-card">
            <div class="label">Reference site · year-round data known</div>
            <div class="name">A · {escape(REFERENCE_LOCATION)}</div>
        </div>
        <div class="location-card">
            <div class="label">Target site · short count used</div>
            <div class="name">B · {escape(TARGET_LOCATION)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

headline1, headline2, headline3, headline4 = st.columns(4)
headline1.metric("Recommended length", best_duration)
headline2.metric("Four-season mean error", f"{best_summary['Mean_Error']:.2f}%")
headline3.metric(
    "Seasonal error range",
    f"{best_summary['Minimum_Error']:.1f}–{best_summary['Maximum_Error']:.1f}%",
)
headline4.metric(
    "Actual AADT",
    f"{actual_aadt:,.0f}",
    help=f"Actual annual average daily traffic at {TARGET_LOCATION}, in vehicles/day.",
)

st.success(
    f"Preliminary finding: {best_duration} produced the lowest average AADT "
    f"error across the four tested seasonal starting dates. It is the best "
    f"tested option—not a universal rule for every intersection."
)


overview_tab, profile_tab, seasonal_tab, method_tab, quality_tab = st.tabs(
    [
        "📊 Overview",
        "🚗 Traffic profile",
        "🌦️ Seasonal robustness",
        "∑ Method",
        "✓ Data quality",
    ]
)


with overview_tab:
    st.subheader("Study locations")
    map_column, map_story = st.columns([2.1, 1])
    with map_column:
        show_location_map()
    with map_story:
        st.markdown(
            f"""
**A · Reference location**  
{REFERENCE_LOCATION}  
Full-year traffic is treated as known.

**B · Estimated location**  
{TARGET_LOCATION}  
Only a short calibration count is treated as known.

**Model direction:** A → B
            """
        )
        st.info(
            "The map explains the model relationship; it does not mean that "
            "vehicles physically travel from A to B."
        )

    st.divider()
    st.subheader("Fixed-start experiment")
    st.write(
        "This view starts the short count on January 1, 2025. It is one "
        "example start date; the next tab tests whether the conclusion changes "
        "when the count begins in another season."
    )

    selected_result = duration_row(results, selected_duration)

    result1, result2, result3, result4 = st.columns(4)
    result1.metric("Target data used", selected_duration)
    result2.metric("Estimated AADT", f"{selected_result['Estimated_AADT']:,.0f}")
    result3.metric(
        "AADT percent error",
        f"{selected_result['AADT_Percent_Error']:.2f}%",
    )
    result4.metric("Daily MAE", f"{selected_result['Daily_MAE']:,.0f} vehicles/day")

    st.caption(
        f"Calibration uses {DURATION_NOTES[selected_duration]}. "
        f"The resulting scale factor is c = {selected_result['Scale_Factor']:.4f}."
    )

    selected_predictions = predictions.loc[
        predictions["Duration"] == selected_duration,
        ["Date", "Actual_B", "Predicted_B"],
    ].copy()
    selected_predictions = selected_predictions.set_index("Date").sort_index()

    if chart_view == "7-day average":
        chart_data = selected_predictions.rolling(7, min_periods=1).mean()
        chart_title = "Actual and predicted traffic · 7-day moving average"
        chart_caption = (
            "Each point averages the current day and the previous six days. "
            "This reveals the yearly pattern while retaining the original daily results."
        )
    else:
        chart_data = selected_predictions
        chart_title = "Actual and predicted daily traffic"
        chart_caption = (
            "Daily values show the full variability, including weekends, holidays, "
            "and unusual dates."
        )

    chart_data.columns = ["Actual traffic", "Predicted traffic"]
    st.subheader(f"{chart_title} at {TARGET_LOCATION}")
    traffic_chart, calibration_start, calibration_end = make_traffic_chart(
        chart_data,
        selected_duration,
    )
    st.altair_chart(traffic_chart, width="stretch")

    if CALIBRATION_LENGTHS[selected_duration] < pd.Timedelta(days=1):
        calibration_text = (
            f"{calibration_start:%b %d, %Y %I:%M %p} to "
            f"{calibration_end:%I:%M %p}"
        )
    else:
        last_calibration_day = calibration_end - pd.Timedelta(days=1)
        calibration_text = (
            f"{calibration_start:%b %d} through "
            f"{last_calibration_day:%b %d, %Y}"
        )

    st.caption(
        f"Pale yellow = calibration period ({calibration_text}) · "
        "Dark blue = observed traffic · Orange = model estimate. "
        + chart_caption
    )

    selected_summary = duration_row(summary, selected_duration)
    fixed_error = float(selected_result["AADT_Percent_Error"])
    mean_error = float(selected_summary["Mean_Error"])
    st.info(
        f"Interpretation: the January 1 test has {fixed_error:.2f}% AADT error, "
        f"while the four-season average for {selected_duration} is {mean_error:.2f}%. "
        "The seasonal average is the more reliable summary because it is not based "
        "on only one starting date."
    )


with profile_tab:
    st.subheader("2025 traffic profile")
    st.write(
        "This page describes the traffic observed at the two intersections. "
        "It provides context for the A-to-B estimation model but does not change "
        "the scale-factor calculation."
    )

    try:
        with st.spinner("Building the traffic profile from the cleaned CSV files..."):
            section_aadt, motor_daily, monthly, monthly_by_section = (
                load_traffic_profile()
            )
    except FileNotFoundError as error:
        st.warning(str(error))
        st.caption(
            "Place the cleaned folder beside this dashboard file, then reload the page."
        )
    else:
        total_by_location = (
            motor_daily.groupby("Location", observed=True)["Motor_Traffic"].mean()
        )
        reference_aadt = float(total_by_location.loc[REFERENCE_DATA_NAME])
        target_aadt_profile = float(total_by_location.loc[TARGET_DATA_NAME])
        target_difference = (target_aadt_profile / reference_aadt - 1) * 100

        profile1, profile2, profile3 = st.columns(3)
        profile1.metric("Reference AADT", f"{reference_aadt:,.0f}")
        profile2.metric("Target AADT", f"{target_aadt_profile:,.0f}")
        profile3.metric("Target compared with reference", f"+{target_difference:.1f}%")
        st.caption(
            "AADT values are average valid daily motor-vehicle counts in 2025."
        )

        st.markdown(f"#### Traffic categories at {TARGET_LOCATION}")
        st.markdown(
            make_vehicle_cards(section_aadt, motor_daily),
            unsafe_allow_html=True,
        )
        st.caption(
            "The cards report target-site averages. Their comparison is between "
            "locations, not between different years. Bicycles and pedestrians are "
            "not included in the motor-vehicle AADT model."
        )

        monthly_display = monthly.copy()
        monthly_display["Location"] = monthly_display["Location"].replace(
            {
                REFERENCE_DATA_NAME: REFERENCE_LOCATION,
                TARGET_DATA_NAME: TARGET_LOCATION,
            }
        )
        monthly_chart = (
            alt.Chart(monthly_display)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b")),
                y=alt.Y(
                    "Average daily traffic:Q",
                    title="Average vehicles per day",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Location:N",
                    scale=alt.Scale(
                        domain=[REFERENCE_LOCATION, TARGET_LOCATION],
                        range=["#0b3c5d", "#f28e2b"],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Month:T", title="Month", format="%B %Y"),
                    alt.Tooltip("Location:N", title="Intersection"),
                    alt.Tooltip(
                        "Average daily traffic:Q",
                        title="Average daily traffic",
                        format=",.0f",
                    ),
                ],
            )
            .properties(height=400)
        )
        st.subheader("Monthly average daily motor traffic")
        st.altair_chart(monthly_chart, width="stretch")
        st.caption(
            "Dark blue = reference intersection · Orange = target intersection. "
            "The target site has five excluded dates from July 30 through August 3."
        )

        st.divider()
        st.subheader("Seasonal traffic pattern by vehicle type")
        st.write(
            "Each colored line shows how one motor-vehicle category changes "
            "during the year. The values are indexed so categories with very "
            "different traffic volumes can be compared on the same chart."
        )

        selected_profile_location = st.selectbox(
            "Intersection shown in the vehicle-type chart",
            [TARGET_LOCATION, REFERENCE_LOCATION],
            key="vehicle_type_profile_location",
        )
        selected_profile_data_name = {
            TARGET_LOCATION: TARGET_DATA_NAME,
            REFERENCE_LOCATION: REFERENCE_DATA_NAME,
        }[selected_profile_location]

        vehicle_type_display = monthly_by_section.loc[
            monthly_by_section["Location"].eq(selected_profile_data_name)
        ].copy()
        vehicle_type_display["Vehicle type"] = vehicle_type_display[
            "Section"
        ].replace(PROFILE_SECTION_LABELS)

        vehicle_type_order = list(PROFILE_SECTION_LABELS.values())
        vehicle_type_colors = ["#28a9f5", "#16a34a", "#f97316", "#8b5cf6"]
        vehicle_color = alt.Color(
            "Vehicle type:N",
            scale=alt.Scale(
                domain=vehicle_type_order,
                range=vehicle_type_colors,
            ),
            legend=alt.Legend(
                title=None,
                orient="bottom",
                columns=2,
            ),
        )

        vehicle_lines = (
            alt.Chart(vehicle_type_display)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X(
                    "Month:T",
                    title="Month",
                    axis=alt.Axis(format="%b"),
                ),
                y=alt.Y(
                    "Index:Q",
                    title="Monthly index (annual average = 100)",
                    scale=alt.Scale(zero=False),
                ),
                color=vehicle_color,
                tooltip=[
                    alt.Tooltip("Month:T", title="Month", format="%B %Y"),
                    alt.Tooltip("Vehicle type:N", title="Vehicle type"),
                    alt.Tooltip(
                        "Monthly average:Q",
                        title="Average vehicles/day",
                        format=",.0f",
                    ),
                    alt.Tooltip(
                        "Index:Q",
                        title="Index",
                        format=".1f",
                    ),
                ],
            )
        )
        average_line = (
            alt.Chart(pd.DataFrame({"Index": [100]}))
            .mark_rule(
                color="#64748b",
                strokeDash=[6, 5],
                strokeWidth=1.5,
            )
            .encode(y="Index:Q")
        )
        vehicle_type_chart = (vehicle_lines + average_line).properties(
            height=420
        )

        st.altair_chart(vehicle_type_chart, width="stretch")
        st.caption(
            "How to read it: 100 = that vehicle type's own annual average; "
            "120 = 20% above its annual average; 80 = 20% below. The dashed "
            "line marks the annual-average level."
        )
        st.info(
            "This chart is descriptive context. It helps compare seasonal "
            "patterns by vehicle type, but the annual A-to-B model still uses "
            "total motor-vehicle traffic."
        )


with seasonal_tab:
    st.subheader("Does the starting season change the result?")
    st.write(
        "We repeat the same experiment from four Monday start dates—one in each "
        "season. This exposes whether a result depends too strongly on when the "
        "short count was collected."
    )

    shown_durations = (
        DURATION_ORDER if include_short_tests else DURATION_ORDER[2:]
    )

    st.markdown("#### Accuracy snapshot")
    st.markdown(make_duration_cards(summary), unsafe_allow_html=True)
    st.caption(
        "Each card reports the four-season mean AADT error. Green means the "
        "error improved from the previous tested duration; red means it became worse."
    )

    st.subheader("AADT error by amount of target-site data")
    error_chart = make_error_chart(summary, shown_durations)
    st.altair_chart(error_chart, width="stretch")
    st.caption(
        "Black = average error · Green = best season · Orange = worst season. "
        "The vertical distance between best and worst shows seasonal sensitivity."
    )

    selected_summary = duration_row(summary, selected_duration)
    selected_seasons = seasonal.loc[
        seasonal["Duration"] == selected_duration
    ].sort_values("Season")

    season1, season2, season3 = st.columns(3)
    season1.metric("Mean error", f"{selected_summary['Mean_Error']:.2f}%")
    season2.metric("Median error", f"{selected_summary['Median_Error']:.2f}%")
    season3.metric(
        "Best–worst range",
        f"{selected_summary['Minimum_Error']:.2f}%–{selected_summary['Maximum_Error']:.2f}%",
    )

    st.subheader(f"Four seasonal tests using {selected_duration}")
    st.dataframe(
        format_result_table(selected_seasons),
        hide_index=True,
        width="stretch",
    )

    seven_day = duration_row(summary, "7 days")
    fourteen_day = duration_row(summary, "14 days")
    thirty_day = duration_row(summary, "30 days")
    st.markdown("#### What the comparison says")
    st.markdown(
        f"""
- **7 days is a practical lower-cost candidate:** its mean error is {seven_day['Mean_Error']:.2f}%.
- **14 days is the best tested balance:** it has the lowest mean error, {fourteen_day['Mean_Error']:.2f}%.
- **30 days does not improve this dataset:** its mean error rises slightly to {thirty_day['Mean_Error']:.2f}%.
- **Season still matters:** the 14-day result ranges from {fourteen_day['Minimum_Error']:.2f}% to {fourteen_day['Maximum_Error']:.2f}% depending on the start date.
        """
    )
    st.warning(
        "This robustness check uses four seasonal starting dates, not every possible "
        "start date. The conclusion should be described as preliminary."
    )


with method_tab:
    st.subheader("Model logic")
    st.write(
        "The model assumes that the two intersections share a reasonably stable "
        "traffic-volume relationship. A short period measured at both sites "
        "estimates that relationship; the reference site's full year then fills "
        "in the target site's hidden year."
    )

    st.markdown(
        f"""
1. Add the five motor-vehicle categories at both intersections for every 15-minute interval.
2. Align **A ({REFERENCE_LOCATION})** and **B ({TARGET_LOCATION})** at identical timestamps.
3. Use only the selected short common period to calculate one scale factor.
        """
    )
    st.latex(r"c_k=\frac{\sum_{t\in k} B_t}{\sum_{t\in k} A_t}")
    st.markdown(
        "4. Treat the rest of the target-site year as hidden and estimate it from the reference site."
    )
    st.latex(r"\widehat{B}_t=c_k A_t")
    st.markdown("5. Compare estimated AADT with the hidden actual AADT.")
    st.latex(
        r"\text{Percent Error}="
        r"\frac{|\widehat{\mathrm{AADT}}_B-\mathrm{AADT}_B|}"
        r"{\mathrm{AADT}_B}\times 100\%"
    )

    st.markdown("#### What the measures mean")
    measure1, measure2 = st.columns(2)
    with measure1:
        st.markdown(
            "**AADT percent error** answers the main research question: how close "
            "the estimated annual average daily traffic is to the actual annual average."
        )
    with measure2:
        st.markdown(
            "**Daily MAE** is the average absolute difference between actual and "
            "predicted daily traffic. It is reported in vehicles per day."
        )

    st.markdown("#### Main assumptions")
    st.markdown(
        """
- The traffic relationship between the two intersections is stable enough to summarize with one scale factor.
- The short count is representative; unusual holidays, construction, or special events can distort it.
- AADT accuracy and day-by-day accuracy are different goals. A model can estimate AADT well while missing some individual days.
        """
    )


with quality_tab:
    st.subheader("Data preparation and quality controls")
    quality1, quality2, quality3, quality4 = st.columns(4)
    quality1.metric("Study period", "2025")
    quality2.metric("Time resolution", "15 minutes")
    quality3.metric("Target days compared", f"{valid_target_days}")
    quality4.metric("Dates excluded", f"{missing_target_days}")

    st.markdown("#### Included traffic")
    st.write(
        "Lights, Motorized Vehicles, Single-Unit Trucks, Articulated Trucks, and "
        "Buses are included. Bicycles and pedestrians are not included in the "
        "motor-vehicle AADT estimate."
    )

    st.markdown("#### Checks applied before modeling")
    st.markdown(
        """
- Keep only rows marked **Valid** after cleaning.
- Require all five motor-vehicle categories before accepting a 15-minute interval.
- Sum the 16 vehicle-movement columns: right, through, left, and U-turn for four approaches.
- Parse the UTC offset in every timestamp, convert to UTC for alignment, and return to **America/Chicago** dates for daily totals. This handles daylight-saving transitions without treating repeated local times as duplicates.
- Compare the two intersections only at shared valid timestamps.
- Exclude five target-site dates with incomplete/invalid data: July 30–August 3, 2025.
        """
    )

    st.markdown("#### Download model outputs")
    download1, download2, download3, download4 = st.columns(4)
    download1.download_button(
        "Fixed-start results",
        RESULTS_FILE.read_bytes(),
        file_name=RESULTS_FILE.name,
        mime="text/csv",
    )
    download2.download_button(
        "Daily predictions",
        PREDICTIONS_FILE.read_bytes(),
        file_name=PREDICTIONS_FILE.name,
        mime="text/csv",
    )
    download3.download_button(
        "Seasonal results",
        SEASONAL_RESULTS_FILE.read_bytes(),
        file_name=SEASONAL_RESULTS_FILE.name,
        mime="text/csv",
    )
    download4.download_button(
        "Seasonal summary",
        SEASONAL_SUMMARY_FILE.read_bytes(),
        file_name=SEASONAL_SUMMARY_FILE.name,
        mime="text/csv",
    )


st.divider()
st.caption(
    "Math 435 Final Project · Lincoln traffic count estimation · "
    "Results are based on the supplied 2025 cleaned traffic files."
)
