"""Interactive summary of short-count AADT estimates by movement."""

import sys

import altair as alt
import pandas as pd
import streamlit as st

from dashboard_common import OUTPUT_FOLDER, SRC_FOLDER, page_header

if str(SRC_FOLDER) not in sys.path:
    sys.path.insert(0, str(SRC_FOLDER))

from config import ERROR_CURVE_MAX_DAYS
from data_loader import load_all_locations

# put result to outputs/aadt
OUTPUTS_DIR = OUTPUT_FOLDER / "aadt"

# Title 
page_header(
    "AADT",
    "AADT by Direction and Turning Movement",
    "Estimate annual average daily traffic using short-count data "
    "for direction totals, turning movements, and direction-by-turn movements.",
)

CURVES_FILE = OUTPUTS_DIR / "movement_aadt_error_curves.csv"
SUMMARY_FILE = OUTPUTS_DIR / "movement_aadt_threshold_summary.csv"
GROUP_ORDER = [
    "Overall total",
    "Direction total",
    "Turn total",
    "Direction and turn",
]
DEFAULT_REFERENCE = "40th Street & South Street"
DEFAULT_TARGET = "South Street & Normal Boulevard"
BAND_ORDER = [
    "1-5 days",
    "6-10 days",
    "11-20 days",
    "21-30 days",
    "31-60 days",
    "61-90 days",
    "Not reached",
]
BAND_COLORS = [
    "#147d79",
    "#49a69b",
    "#f2c56b",
    "#e79567",
    "#b783c8",
    "#7b6bb2",
    "#e1e7ed",
]



@st.cache_data
def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the reproducible curve and threshold files."""
    if not CURVES_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError("Generate the movement results first.")
    curves = pd.read_csv(CURVES_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    curves["Calibration_Days"] = curves["Calibration_Days"].astype(int)
    return curves, summary


@st.cache_data(show_spinner="Loading cleaned daily traffic counts...")
def load_daily_counts() -> dict[str, pd.DataFrame]:
    """Keep only valid daily counts from the cleaned intersection records."""
    locations, _ = load_all_locations()
    return {name: location["daily"] for name, location in locations.items()}


def shared_daily_counts(
    daily_counts: dict[str, pd.DataFrame],
    reference: str,
    target: str,
    measure: str,
) -> pd.DataFrame:
    """Use the same shared valid local dates as the AADT analysis."""
    return pd.concat(
        [
            daily_counts[reference][measure].rename("Reference"),
            daily_counts[target][measure].rename("Actual"),
        ],
        axis=1,
        join="inner",
    ).dropna().sort_index()


def possible_count_starts(paired: pd.DataFrame, days: int) -> list[pd.Timestamp]:
    """Find truly consecutive d-day counts with a positive reference sum."""
    starts = []
    for offset in range(len(paired) - days + 1):
        window = paired.iloc[offset : offset + days]
        if (
            window.index[-1] - window.index[0] == pd.Timedelta(days=days - 1)
            and window["Reference"].sum() > 0
        ):
            starts.append(window.index[0])
    return starts


def daily_prediction_chart(displayed: pd.DataFrame) -> alt.Chart:
    """Compare actual and predicted daily counts with their gap shaded."""
    chart_data = displayed.reset_index()
    chart_data["Difference"] = chart_data["Predicted"] - chart_data["Actual"]
    chart_data["Absolute difference"] = chart_data["Difference"].abs()
    x_axis = alt.X(
        "Date:T",
        title="Evaluation date",
        axis=alt.Axis(format="%b %d", labelAngle=-30, tickCount=10),
    )
    hover = [
        alt.Tooltip("Date:T", title="Date", format="%b %d, %Y"),
        alt.Tooltip("Actual:Q", title="Actual vehicles", format=",.0f"),
        alt.Tooltip("Predicted:Q", title="Predicted vehicles", format=",.1f"),
        alt.Tooltip("Difference:Q", title="Predicted - actual", format="+,.1f"),
        alt.Tooltip("Absolute difference:Q", title="Absolute difference", format=",.1f"),
    ]
    gap = alt.Chart(chart_data).mark_area(color="#a6cdd3", opacity=0.5).encode(
        x=x_axis,
        y=alt.Y("Actual:Q", title="Vehicles per day", scale=alt.Scale(zero=False)),
        y2="Predicted:Q",
        tooltip=hover,
    )
    # Both values must stay in the tooltip. Melt would drop them as plotted
    # series if they were also listed as id_vars, leaving zero line rows.
    line_data = pd.concat(
        [
            chart_data.assign(Series="Actual", Vehicles=chart_data["Actual"]),
            chart_data.assign(Series="Predicted", Vehicles=chart_data["Predicted"]),
        ],
        ignore_index=True,
    )
    lines = alt.Chart(line_data).mark_line(strokeWidth=2.5).encode(
        x=x_axis,
        y=alt.Y("Vehicles:Q", title="Vehicles per day"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=["Actual", "Predicted"],
                range=["#165e85", "#df784a"],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=hover,
    )
    return (gap + lines).properties(height=370)


def threshold_words(days: float, threshold: int) -> str:
    """Describe a first crossing without claiming a guaranteed minimum."""
    if pd.isna(days):
        return (
            f"mean error did not reach ≤{threshold}% within "
            f"{ERROR_CURVE_MAX_DAYS} tested days"
        )
    count = int(days)
    return f"mean error first reaches ≤{threshold}% at {count} day{'s' if count != 1 else ''}"


def duration_band(days: float) -> str:
    """Assign one clear color to each range of first-crossing days."""
    if pd.isna(days):
        return "Not reached"
    if days <= 5:
        return "1-5 days"
    if days <= 10:
        return "6-10 days"
    if days <= 20:
        return "11-20 days"
    if days <= 30:
        return "21-30 days"
    if days <= 60:
        return "31-60 days"
    return "61-90 days"


def build_overview_cells(overview: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create two readable threshold cells for every measure."""
    ordered = overview.sort_values(
        ["Days_to_10pct", "Best_Mean_Error", "Measure"],
        na_position="last",
    )
    measure_order = ordered["Measure"].tolist()
    cells = []
    for _, row in ordered.iterrows():
        for threshold in (10, 5):
            days = row[f"Days_to_{threshold}pct"]
            cells.append(
                {
                    "Measure": row["Measure"],
                    "Threshold": f"{threshold}%",
                    "Days": (
                        f"{int(days)}d"
                        if pd.notna(days)
                        else f">{ERROR_CURVE_MAX_DAYS}d"
                    ),
                    "First_Crossing": (
                        f"{int(days)} days"
                        if pd.notna(days)
                        else f"Not reached in {ERROR_CURVE_MAX_DAYS} days"
                    ),
                    "Band": duration_band(days),
                    "Best_Mean_Error": float(row["Best_Mean_Error"]),
                }
            )
    return pd.DataFrame(cells), measure_order


def format_threshold_days(value: float) -> str:
    """Format a threshold crossing without turning missing results into zero."""
    if pd.isna(value):
        return f">{ERROR_CURVE_MAX_DAYS} days"
    count = int(value)
    return f"{count} day{'s' if count != 1 else ''}"


def threshold_detail_table(
    group_summary: pd.DataFrame,
    group_curves: pd.DataFrame,
) -> pd.DataFrame:
    """Combine threshold results with target volume and data availability."""
    measure_details = (
        group_curves.sort_values("Calibration_Days")
        .drop_duplicates("Measure")[
            ["Measure", "Actual_Target_AADT", "Shared_Valid_Dates"]
        ]
    )
    table = group_summary.merge(measure_details, on="Measure", how="left")
    table["Days_to_10pct"] = table["Days_to_10pct"].map(format_threshold_days)
    table["Days_to_5pct"] = table["Days_to_5pct"].map(format_threshold_days)
    return table[
        [
            "Measure",
            "Actual_Target_AADT",
            "Days_to_10pct",
            "Days_to_5pct",
            "Best_Mean_Error",
            "Best_Calibration_Days",
            "Shared_Valid_Dates",
        ]
    ].rename(
        columns={
            "Measure": "Traffic measure",
            "Actual_Target_AADT": "Actual target AADT",
            "Days_to_10pct": "First day at <=10%",
            "Days_to_5pct": "First day at <=5%",
            "Best_Mean_Error": "Best mean error (%)",
            "Best_Calibration_Days": "Days at best error",
            "Shared_Valid_Dates": "Shared valid dates",
        }
    )


def median_threshold_days(values: pd.Series) -> str:
    """Summarize reached measures without treating unreached results as 90."""
    reached = values.dropna()
    if reached.empty:
        return "Not reached"
    median = float(reached.median())
    displayed = f"{median:.0f}" if median.is_integer() else f"{median:.1f}"
    return f"{displayed} days"


def overview_chart(cells: pd.DataFrame, order: list[str]) -> alt.Chart:
    """Draw a compact, labeled map of first crossings."""
    common = alt.Chart(cells).encode(
        x=alt.X(
            "Threshold:N",
            sort=["10%", "5%"],
            title="Mean-error target",
            axis=alt.Axis(labelAngle=0, labelFontSize=13),
        ),
        y=alt.Y(
            "Measure:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=220, labelFontSize=13),
        ),
        tooltip=[
            alt.Tooltip("Measure:N", title="Traffic measure"),
            alt.Tooltip("Threshold:N", title="Target"),
            alt.Tooltip("First_Crossing:N", title="First crossing"),
            alt.Tooltip("Best_Mean_Error:Q", title="Best mean error", format=".2f"),
        ],
    )
    tiles = common.mark_rect(
        cornerRadius=5, stroke="#ffffff", strokeWidth=4
    ).encode(
        color=alt.Color(
            "Band:N",
            title="First crossing",
            scale=alt.Scale(domain=BAND_ORDER, range=BAND_COLORS),
            legend=alt.Legend(orient="top", columns=4),
        )
    )
    labels = common.mark_text(fontSize=14, fontWeight="bold").encode(
        text="Days:N",
        color=alt.condition(
            alt.datum.Band == "1-5 days",
            alt.value("#ffffff"),
            alt.value("#17324a"),
        ),
    )
    # Vega-Lite otherwise gives two nominal columns a tiny default step,
    # making labels such as >90d overlap in the four-row direction view.
    return (tiles + labels).properties(width=alt.Step(210), height=alt.Step(42))


def error_curve_chart(curve: pd.DataFrame) -> alt.Chart:
    """Show the mean error, starting-date spread, and two reference lines."""
    low = int(curve["Calibration_Days"].min())
    high = int(curve["Calibration_Days"].max())
    band = (
        alt.Chart(curve)
        .mark_area(color="#32a899", opacity=0.22)
        .encode(
            x=alt.X(
                "Calibration_Days:Q",
                title="Consecutive target-site days counted",
                scale=alt.Scale(domain=[low, high]),
            ),
            y=alt.Y("P10_Error:Q", title="AADT error (%)", scale=alt.Scale(zero=True)),
            y2="P90_Error:Q",
        )
    )
    mean = (
        alt.Chart(curve)
        .mark_line(point=True, strokeWidth=3, color="#165e85")
        .encode(
            x="Calibration_Days:Q",
            y="Mean_Error:Q",
            tooltip=[
                alt.Tooltip("Calibration_Days:Q", title="Days", format=".0f"),
                alt.Tooltip("Mean_Error:Q", title="Mean error", format=".2f"),
                alt.Tooltip("Median_Error:Q", title="Median error", format=".2f"),
                alt.Tooltip("P10_Error:Q", title="10th percentile", format=".2f"),
                alt.Tooltip("P90_Error:Q", title="90th percentile", format=".2f"),
                alt.Tooltip("Calibration_Windows:Q", title="Starting windows", format=","),
            ],
        )
    )
    targets = alt.Chart(
        pd.DataFrame(
            {"Target": ["5% target", "10% target"], "Error": [5.0, 10.0]}
        )
    ).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
        y="Error:Q",
        color=alt.Color(
            "Target:N",
            scale=alt.Scale(
                domain=["5% target", "10% target"],
                range=["#d1495b", "#d78b21"],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    return (band + mean + targets).properties(height=390)


st.title("How many days of traffic counting are enough?")
st.caption(
    "MATH 435 | Lincoln traffic estimation | 2025 data | Motor vehicles only"
)

try:
    curves, summary = load_results()
except FileNotFoundError:
    st.error("The movement result files have not been generated yet.")
    st.code("python3 analysis/generate_movement_results.py", language="bash")
    st.stop()

reference_options = sorted(curves["Reference_Location"].dropna().unique())
left, right = st.columns(2)
with left:
    reference = st.selectbox(
        "Reference location (long-term record)",
        reference_options,
        index=(
            reference_options.index(DEFAULT_REFERENCE)
            if DEFAULT_REFERENCE in reference_options
            else 0
        ),
    )
target_options = sorted(
    curves.loc[
        curves["Reference_Location"].eq(reference), "Target_Location"
    ].dropna().unique()
)
with right:
    target = st.selectbox(
        "Target location (simulated short count)",
        target_options,
        index=(
            target_options.index(DEFAULT_TARGET)
            if DEFAULT_TARGET in target_options
            else 0
        ),
    )

pair_curves = curves.loc[
    curves["Reference_Location"].eq(reference)
    & curves["Target_Location"].eq(target)
].copy()
pair_summary = summary.loc[
    summary["Reference_Location"].eq(reference)
    & summary["Target_Location"].eq(target)
].copy()

available_groups = [
    group for group in GROUP_ORDER if group in pair_curves["Measure_Group"].unique()
]
control_a, control_b, control_c = st.columns([1.2, 1.5, 1.3])
with control_a:
    group = st.selectbox(
        "Analysis level",
        available_groups,
        index=(
            available_groups.index("Direction and turn")
            if "Direction and turn" in available_groups
            else 0
        ),
    )
measure_options = sorted(
    pair_curves.loc[
        pair_curves["Measure_Group"].eq(group), "Measure"
    ].dropna().unique()
)
with control_b:
    measure = st.selectbox(
        "Traffic measure",
        measure_options,
        index=(
            measure_options.index("Eastbound Thru")
            if "Eastbound Thru" in measure_options
            else 0
        ),
    )

selected_curve = pair_curves.loc[pair_curves["Measure"].eq(measure)].sort_values(
    "Calibration_Days"
)
with control_c:
    selected_days = st.slider(
        "Inspect one count length",
        min_value=int(selected_curve["Calibration_Days"].min()),
        max_value=int(selected_curve["Calibration_Days"].max()),
        value=min(14, int(selected_curve["Calibration_Days"].max())),
        step=1,
        help="This changes the figures below. The error curve still shows every tested length.",
    )
selected_row = selected_curve.loc[
    selected_curve["Calibration_Days"].eq(selected_days)
].iloc[0]
summary_row = pair_summary.loc[pair_summary["Measure"].eq(measure)].iloc[0]

st.caption(f"Estimation direction: {reference} → {target}")

st.subheader("The takeaway for your selection")
st.info(
    f"**{measure}:** {threshold_words(summary_row['Days_to_10pct'], 10)}; "
    f"{threshold_words(summary_row['Days_to_5pct'], 5)}. "
    f"With {selected_days} counted days, the mean AADT error across "
    f"{int(selected_row['Calibration_Windows']):,} starting windows is "
    f"**{selected_row['Mean_Error']:.2f}%**."
)
st.caption(
    "A first crossing describes the average over starting windows. It does not "
    "guarantee that every starting date meets the target or that the curve stays below it."
)

cards = st.columns(4)
cards[0].metric("Actual target AADT", f"{selected_row['Actual_Target_AADT']:,.0f}")
cards[1].metric(
    f"Mean estimate ({selected_days} days)",
    f"{selected_row['Mean_Estimated_Target_AADT']:,.0f}",
)
cards[2].metric("Mean AADT error", f"{selected_row['Mean_Error']:.2f}%")
cards[3].metric("Starting windows tested", f"{int(selected_row['Calibration_Windows']):,}")

st.subheader("Minimum count duration by movement")
st.write(
    "This decision table reports the first tested count length whose mean AADT "
    "error reaches 10% or 5%. It includes every estimable direction total, turn "
    "total, and direction-by-turn movement for the selected site pair."
)

decision_groups = ["Direction total", "Turn total", "Direction and turn"]
decision_summary = pair_summary.loc[
    pair_summary["Measure_Group"].isin(decision_groups)
].copy()
decision_hits_10 = int(decision_summary["Days_to_10pct"].notna().sum())
decision_hits_5 = int(decision_summary["Days_to_5pct"].notna().sum())
decision_cards = st.columns(4)
decision_cards[0].metric(
    f"Reach <=10% within {ERROR_CURVE_MAX_DAYS} days",
    f"{decision_hits_10} of {len(decision_summary)}",
)
decision_cards[1].metric(
    f"Reach <=5% within {ERROR_CURVE_MAX_DAYS} days",
    f"{decision_hits_5} of {len(decision_summary)}",
)
decision_cards[2].metric(
    "Median days to <=10%",
    median_threshold_days(decision_summary["Days_to_10pct"]),
)
decision_cards[3].metric(
    "Median days to <=5%",
    median_threshold_days(decision_summary["Days_to_5pct"]),
)
st.caption(
    "The two medians use only measures that reached the target. An unreached "
    f"measure is displayed as >{ERROR_CURVE_MAX_DAYS} days, not as zero."
)

tab_specs = [
    ("Direction totals", "Direction total"),
    ("Turn totals", "Turn total"),
    ("Direction x turn", "Direction and turn"),
]
threshold_tabs = st.tabs([label for label, _ in tab_specs])
for tab, (label, group_name) in zip(threshold_tabs, tab_specs):
    with tab:
        group_overview = decision_summary.loc[
            decision_summary["Measure_Group"].eq(group_name)
        ].copy()
        group_curve_data = pair_curves.loc[
            pair_curves["Measure_Group"].eq(group_name)
        ].copy()
        group_hits_10 = int(group_overview["Days_to_10pct"].notna().sum())
        group_hits_5 = int(group_overview["Days_to_5pct"].notna().sum())
        st.caption(
            f"{group_hits_10} of {len(group_overview)} measures reach <=10%; "
            f"{group_hits_5} of {len(group_overview)} reach <=5% within "
            f"{ERROR_CURVE_MAX_DAYS} tested days."
        )
        cells, measure_order = build_overview_cells(group_overview)
        st.altair_chart(overview_chart(cells, measure_order), width="stretch")
        detail_table = threshold_detail_table(group_overview, group_curve_data)
        st.dataframe(
            detail_table.sort_values("Traffic measure"),
            hide_index=True,
            width="stretch",
            height=min(665, 38 * (len(detail_table) + 1)),
            column_config={
                "Actual target AADT": st.column_config.NumberColumn(format="%.1f"),
                "Best mean error (%)": st.column_config.NumberColumn(format="%.2f"),
                "Days at best error": st.column_config.NumberColumn(format="%d"),
                "Shared valid dates": st.column_config.NumberColumn(format="%d"),
            },
        )

st.caption(
    "Each colored cell shows the first tested count length at or below the "
    f"indicated mean-error target. Gray (>{ERROR_CURVE_MAX_DAYS}d) means no "
    f"tested length from 1 through {ERROR_CURVE_MAX_DAYS} days reached it. "
    "Low-volume movements can have unstable percentage errors, so the table "
    "also reports actual target AADT. U-turns are omitted when both sites do "
    "not contain a positive, estimable U-turn series."
)
st.download_button(
    "Download the full movement threshold table",
    data=decision_summary.to_csv(index=False).encode("utf-8"),
    file_name="movement_threshold_summary_selected_site_pair.csv",
    mime="text/csv",
)

st.subheader(f"Why? Error across 1–{ERROR_CURVE_MAX_DAYS} counted days")
st.altair_chart(error_curve_chart(selected_curve), width="stretch")
st.caption(
    "Blue = mean absolute percentage error across all possible consecutive "
    "starting windows. Teal band = 10th–90th percentile of those errors, showing "
    "sensitivity to the start date; it is not a confidence interval. Dashed "
    "lines mark the 5% and 10% mean-error targets."
)

st.subheader(f"Daily actual vs predicted: {measure}")
st.write(
    "Choose one consecutive target-site count to see how its ratio predicts "
    "daily traffic for this measure. The shaded gap is the daily prediction "
    "difference; these are daily counts, not AADT."
)
daily_counts = load_daily_counts()
paired_daily = shared_daily_counts(daily_counts, reference, target, measure)
count_starts = possible_count_starts(paired_daily, selected_days)
if not count_starts:
    st.info("No consecutive count period is available for this selection.")
else:
    start_date = st.selectbox(
        "Count period starts on",
        count_starts,
        format_func=lambda date: date.strftime("%Y-%m-%d"),
        key=f"count_start_{reference}_{target}_{measure}_{selected_days}",
        help="The earliest valid period is shown first. You can try other start dates.",
    )
    calibration = paired_daily.loc[
        start_date : start_date + pd.Timedelta(days=selected_days - 1)
    ]
    ratio = calibration["Actual"].sum() / calibration["Reference"].sum()
    predicted_daily = paired_daily.copy()
    predicted_daily["Predicted"] = ratio * predicted_daily["Reference"]
    evaluation = predicted_daily.drop(index=calibration.index)
    later_dates = evaluation.loc[evaluation.index > calibration.index[-1]]
    view_options = ["Next 30 days after the count", "All dates outside the count"]
    view = st.radio("Dates to show", view_options, horizontal=True)
    if view == view_options[0] and later_dates.empty:
        st.info("No later valid dates are available; showing all dates outside the count.")
        displayed = evaluation
    elif view == view_options[0]:
        displayed = later_dates.head(30)
    else:
        displayed = evaluation

    st.caption(
        f"Count period: {calibration.index[0]:%Y-%m-%d} to "
        f"{calibration.index[-1]:%Y-%m-%d} ({selected_days} consecutive days). "
        f"Prediction for each date = reference daily count × {ratio:.3f}. "
        "The count-period dates are excluded from the comparison."
    )
    if displayed.empty:
        st.info("No dates remain for an out-of-count comparison.")
    else:
        st.altair_chart(daily_prediction_chart(displayed), width="stretch")
        differences = displayed["Predicted"] - displayed["Actual"]
        st.caption(
            f"{len(displayed)} displayed dates | Mean absolute daily difference: "
            f"{differences.abs().mean():,.1f} vehicles/day. "
            "Predicted − Actual > 0 means overprediction; < 0 means underprediction. "
            "The AADT error curve above averages results from every possible "
            "count start, while this daily chart shows the one selected start."
        )

st.subheader("Actual versus estimated AADT")
comparison = pd.DataFrame(
    {
        "Series": ["Actual target", f"Mean estimate ({selected_days} days)"],
        "AADT": [
            selected_row["Actual_Target_AADT"],
            selected_row["Mean_Estimated_Target_AADT"],
        ],
    }
)
bars = (
    alt.Chart(comparison)
    .mark_bar(size=85, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
    .encode(
        x=alt.X("Series:N", title=None, sort=None),
        y=alt.Y("AADT:Q", title="Vehicles per day"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(range=["#165e85", "#e79567"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("AADT:Q", title="AADT", format=",.1f"),
        ],
    )
    .properties(height=280)
)
st.altair_chart(bars, width="stretch")
st.caption(
    "Blue is the actual target AADT. Orange is the mean estimated AADT across "
    f"all available {selected_days}-day starting windows."
)

st.download_button(
    "Download the selected error curve",
    data=selected_curve.to_csv(index=False).encode("utf-8"),
    file_name="selected_movement_aadt_error_curve.csv",
    mime="text/csv",
)
st.caption(
    "These are retrospective tests using the three monitored intersections. "
    "They do not guarantee the same error at a new, unmonitored intersection."
)

