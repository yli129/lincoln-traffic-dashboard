"""Day-to-Week page: estimate Weekly ADT from one day of counts.

Reads the tables written by Project2_data/analysis/day_to_week_recommend.py
(Project2_data/outputs/day_to_week/). Regenerate them after the data changes:

    python Project2_data/analysis/day_to_week_recommend.py
"""

from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard_common import DIRECTIONS, OUTPUT_FOLDER, page_header, readable_name

RESULT_FOLDER = OUTPUT_FOLDER / "day_to_week"

TABLE_FILES = {
    "factors": "annual_factors.csv",
    "rec_ref": "recommendation_by_reference.csv",
    "rec_pooled": "recommendation_pooled.csv",
    "metrics_ref": "weekday_metrics_by_reference.csv",
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SHORT_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SECTION_ORDER = [
    "All Motor Vehicles",
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]
SECTION_LABELS = {
    "All Motor Vehicles": "All motor vehicles",
    "Lights": "Light vehicles",
    "Single-Unit Trucks": "Single-unit trucks",
    "Articulated Trucks": "Articulated trucks",
    "Buses": "Buses",
    "Motorized Vehicles": "Other motorized vehicles",
}
DIRECTION_OPTIONS = ["Total"] + DIRECTIONS
RELIABILITY_ICON = {"High": "🟢", "Moderate": "🟡", "Low": "🟠", "Insufficient volume": "🔴"}

TEAL = "#176b87"
GREY = "#b8c4cc"
ORANGE = "#c97a00"


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------
def section_label(section):
    return SECTION_LABELS.get(section, section)


def direction_label(direction):
    return "All directions" if direction == "Total" else direction


def estimate_weekly_adt(volume, factor):
    """Weekly ADT = counted daily volume x annual factor of that weekday."""
    return volume * factor


def likely_range(estimate, p90_ape_pct):
    """
    Range of true Weekly ADT that is consistent with a 90th-percentile
    absolute percentage error. APE = |estimate - actual| / actual, so
    actual lies in [estimate / (1 + y), estimate / (1 - y)].
    The upper bound is open when y >= 100%.
    """
    y = p90_ape_pct / 100.0
    low = estimate / (1 + y)
    high = estimate / (1 - y) if y < 1 else None
    return low, high


def pick_case(df, **conditions):
    mask = pd.Series(True, index=df.index)
    for column, value in conditions.items():
        mask &= df[column] == value
    return df[mask]


def day_index(short_name):
    return SHORT_DAYS.index(short_name)


def parse_days(text):
    return [day_index(d.strip()) for d in str(text).split(",") if d.strip() in SHORT_DAYS]


def reco_card_html(rec):
    """Coloured card with the one-line advice."""
    if rec is None:
        css, label = "bad", "No recommendation"
        text = "No test results are available for this combination."
    elif rec["Reliability"] == "Insufficient volume":
        css, label = "bad", "Not recommended"
        text = rec["Advice"]
    elif rec["Reliability"] == "Low":
        css, label = "warn", "Recommendation (low accuracy)"
        text = rec["Advice"]
    else:
        css, label = "", "Recommended data collection"
        text = rec["Advice"]
    return (
        f'<div class="reco-card {css}"><div class="label">{escape(label)}</div>'
        f'<div class="text">{escape(str(text))}</div></div>'
    )


def error_chart(metrics, equivalent_days, selected_day):
    """WAPE by weekday; recommended days in teal, the day being used outlined."""
    values = metrics.set_index("Weekday")["WAPE"].reindex(range(7))
    colors = [TEAL if d in equivalent_days else GREY for d in range(7)]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(range(7), values.fillna(0).to_numpy(), color=colors, width=0.68)
    if selected_day is not None:
        bars[selected_day].set_edgecolor(ORANGE)
        bars[selected_day].set_linewidth(3)

    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(7))
    ax.set_xticklabels(SHORT_DAYS)
    ax.set_ylabel("Expected error (WAPE, %)")
    ax.set_title("Which day gives the best estimate?")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values.max() * 1.18, 1))
    fig.tight_layout()
    return fig


def summary_lines(pooled_total):
    """Plain-language conclusions from the pooled Total-direction table."""
    lines = []
    usable = pooled_total[pooled_total["Reliability"] != "Insufficient volume"]
    for section in SECTION_ORDER:
        row = pick_case(pooled_total, Section=section)
        if row.empty:
            continue
        row = row.iloc[0]
        label = section_label(section)
        if row["Reliability"] == "Insufficient volume":
            lines.append(
                f"- **{label}**: not recommended. Traffic volume is too low for a "
                f"reliable one-day estimate; collect a full week instead."
            )
            continue
        others = [d.strip() for d in row["Equivalent_Days"].split(",")
                  if d.strip() != row["Best_Day"]]
        alt = f" (also {', '.join(others)})" if others else ""
        lines.append(
            f"- **{label}**: collect on **{row['Best_Day']}**{alt}; "
            f"expected error about {row['Best_WAPE']:.1f}%."
        )
    if not usable.empty:
        worst = usable["Worst_Day"].mode().iloc[0]
        lines.append(
            f"- The weakest day to collect on is most often **{worst}**; "
            f"avoid it when you can choose."
        )
    return lines


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tables():
    missing = [name for name in TABLE_FILES.values() if not (RESULT_FOLDER / name).exists()]
    if missing:
        return None, missing
    tables = {key: pd.read_csv(RESULT_FOLDER / name) for key, name in TABLE_FILES.items()}
    return tables, []


page_header(
    "Day → Week",
    "Weekly ADT from one day of counts",
    "Choose an intersection with a known full year of data, then enter one day's "
    "count from a similar intersection. The dashboard estimates the weekly average "
    "daily traffic (Weekly ADT) and tells you which weekday is best to collect on.",
)

tables, missing = load_tables()
if tables is None:
    st.error(
        "The Day-to-Week result tables are missing: "
        + ", ".join(f"`{m}`" for m in missing)
        + f". Generate them with `python Project2_data/analysis/day_to_week_recommend.py` "
        f"(they are written to `{RESULT_FOLDER}`)."
    )
    st.stop()

factors = tables["factors"]
rec_ref = tables["rec_ref"]
rec_pooled = tables["rec_pooled"]
metrics_ref = tables["metrics_ref"]

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Day-to-Week settings")

    references = sorted(factors["A"].unique())
    reference = st.selectbox(
        "Reference intersection (full-year data)",
        options=references,
        format_func=readable_name,
        help="The intersection whose full-year counts are treated as known. "
             "Its weekday factors are used for the estimate.",
    )

    available_sections = [s for s in SECTION_ORDER if s in set(factors["Section"])]
    section = st.selectbox(
        "Vehicle type", options=available_sections, format_func=section_label
    )

    available_directions = [
        d for d in DIRECTION_OPTIONS
        if not pick_case(factors, A=reference, Section=section, Direction=d).empty
    ]
    direction = st.selectbox(
        "Traffic direction", options=available_directions, format_func=direction_label
    )

case_key = f"{reference}|{section}|{direction}"
case_factors = pick_case(factors, A=reference, Section=section, Direction=direction).sort_values("Weekday")
case_metrics = pick_case(metrics_ref, A=reference, Section=section, Direction=direction).sort_values("Weekday")
rec_rows = pick_case(rec_ref, A=reference, Section=section, Direction=direction)
rec = rec_rows.iloc[0] if not rec_rows.empty else None

mean_adt = float(case_factors["Mean_Weekly_ADT"].iloc[0])

# ----------------------------------------------------------------------
# Recommendation header
# ----------------------------------------------------------------------
st.markdown(reco_card_html(rec), unsafe_allow_html=True)

a, b, c, d = st.columns(4)
if rec is not None:
    a.metric("Recommended weekday", rec["Best_Day"])
    others = [x.strip() for x in rec["Equivalent_Days"].split(",") if x.strip() != rec["Best_Day"]]
    b.metric("Also acceptable", ", ".join(others) if others else "None")
    c.metric("Expected error (WAPE)", f"{rec['Best_WAPE']:.1f}%")
    d.metric("Reliability", f"{RELIABILITY_ICON.get(rec['Reliability'], '')} {rec['Reliability']}")
else:
    a.metric("Recommended weekday", "—")
    b.metric("Also acceptable", "—")
    c.metric("Expected error (WAPE)", "—")
    d.metric("Reliability", "—")

estimate_tab, window_tab, method_tab = st.tabs(
    ["🧮 Estimate Weekly ADT", "📌 Collection window", "ℹ️ Method"]
)

# ----------------------------------------------------------------------
# Tab 1: estimate
# ----------------------------------------------------------------------
with estimate_tab:
    left, right = st.columns([1, 1.35])

    default_day = day_index(rec["Best_Day"]) if rec is not None else 3

    with left:
        st.subheader("Your count")
        chosen_day = st.selectbox(
            "Weekday of the count",
            options=list(range(7)),
            index=default_day,
            format_func=lambda i: WEEKDAYS[i],
            key=f"day|{case_key}",
            help="The weekday on which the 24-hour count was taken.",
        )

        factor_row = case_factors[case_factors["Weekday"] == chosen_day]
        factor = float(factor_row["Factor"].iloc[0]) if not factor_row.empty else np.nan

        typical = mean_adt / factor if factor and np.isfinite(factor) else mean_adt
        volume = st.number_input(
            "Daily volume counted (vehicles, full 24 hours)",
            min_value=0,
            value=int(round(typical)),
            step=1 if mean_adt < 200 else 10,
            key=f"volume|{case_key}|{chosen_day}",
            help="The total for the whole day. The default is a typical value for "
                 "this weekday at the reference intersection.",
        )

    with right:
        st.subheader("Estimate")
        if not np.isfinite(factor):
            st.warning("No factor is available for this weekday, so no estimate can be made.")
        else:
            estimate = estimate_weekly_adt(volume, factor)
            day_metrics = case_metrics[case_metrics["Weekday"] == chosen_day]

            m1, m2 = st.columns(2)
            m1.metric("Estimated Weekly ADT", f"{estimate:,.0f} vehicles/day")
            m2.metric("Factor used", f"{factor:.3f}")
            st.caption(f"{volume:,} vehicles × {factor:.3f} ({WEEKDAYS[chosen_day]} factor)")

            if not day_metrics.empty:
                wape = float(day_metrics["WAPE"].iloc[0])
                p90 = float(day_metrics["P90_APE"].iloc[0])
                low, high = likely_range(estimate, p90)
                e1, e2 = st.columns(2)
                e1.metric(f"Typical error on {WEEKDAYS[chosen_day]}", f"{wape:.1f}%")
                if high is None:
                    e2.metric("Likely range (9 in 10 weeks)", f"above {low:,.0f}")
                else:
                    e2.metric("Likely range (9 in 10 weeks)", f"{low:,.0f} – {high:,.0f}")

            if rec is not None and rec["Reliability"] == "Insufficient volume":
                st.warning(
                    "Traffic volume here is too low for a reliable one-day estimate. "
                    "Treat this number as a rough guide only."
                )
            elif rec is not None and chosen_day not in parse_days(rec["Equivalent_Days"]):
                best_row = case_metrics[case_metrics["Weekday"] == day_index(rec["Best_Day"])]
                if not best_row.empty and not day_metrics.empty:
                    st.info(
                        f"{WEEKDAYS[chosen_day]} is not one of the recommended days. "
                        f"Its expected error is {float(day_metrics['WAPE'].iloc[0]):.1f}%, "
                        f"compared with {float(best_row['WAPE'].iloc[0]):.1f}% on {rec['Best_Day']}."
                    )

    st.divider()
    chart_col, table_col = st.columns([1.1, 1])
    with chart_col:
        if not case_metrics.empty:
            equivalent = parse_days(rec["Equivalent_Days"]) if rec is not None else []
            fig = error_chart(case_metrics, equivalent, chosen_day)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Teal bars: recommended days. Orange outline: the day you selected.")
    with table_col:
        st.markdown("**All weekdays for this case**")
        table = case_factors[["Weekday", "Factor"]].merge(
            case_metrics[["Weekday", "WAPE", "MAE", "P90_APE", "Bias_pct"]],
            on="Weekday", how="left",
        )
        table.insert(0, "Day", table["Weekday"].map(dict(enumerate(WEEKDAYS))))
        table = table.drop(columns="Weekday").rename(columns={
            "WAPE": "Error WAPE (%)",
            "MAE": "Error MAE (vehicles)",
            "P90_APE": "Worst 10% of weeks (%)",
            "Bias_pct": "Bias (%)",
        })
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
        st.caption(f"Average Weekly ADT at {readable_name(reference)}: {mean_adt:,.0f} vehicles/day.")

# ----------------------------------------------------------------------
# Tab 2: collection window
# ----------------------------------------------------------------------
with window_tab:
    st.subheader("Which weekday should we collect on?")
    basis = st.radio(
        "Based on",
        ["All intersection pairs (general guidance)", "Selected reference intersection only"],
        horizontal=True,
    )
    general = basis.startswith("All")
    source = rec_pooled if general else rec_ref[rec_ref["A"] == reference]

    st.markdown("**Recommended collection day by vehicle type (all directions)**")
    total = source[source["Direction"] == "Total"].copy()
    total["order"] = total["Section"].map({s: i for i, s in enumerate(SECTION_ORDER)})
    total = total.sort_values("order")

    summary = pd.DataFrame({
        "Vehicle type": total["Section"].map(section_label),
        "Recommended day": np.where(total["Reliability"] == "Insufficient volume", "—", total["Best_Day"]),
        "Also acceptable": [
            "—" if r == "Insufficient volume" else
            ", ".join(x.strip() for x in days.split(",") if x.strip() != best) or "—"
            for r, days, best in zip(total["Reliability"], total["Equivalent_Days"], total["Best_Day"])
        ],
        "Expected error (WAPE %)": total["Best_WAPE"].round(1),
        "Reliability": [f"{RELIABILITY_ICON.get(r, '')} {r}" for r in total["Reliability"]],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("**Recommended day by direction**")
    cells = source.copy()
    cells["Cell"] = np.where(
        cells["Reliability"] == "Insufficient volume",
        "—",
        cells["Best_Day"] + " (" + cells["Best_WAPE"].round(1).astype(str) + "%)",
    )
    matrix = cells.pivot(index="Section", columns="Direction", values="Cell")
    matrix = matrix.reindex(index=[s for s in SECTION_ORDER if s in matrix.index],
                            columns=[x for x in DIRECTION_OPTIONS if x in matrix.columns])
    matrix.index = [section_label(s) for s in matrix.index]
    matrix.columns = [direction_label(x) for x in matrix.columns]
    st.dataframe(matrix.fillna("—"), use_container_width=True)
    st.caption("A dash means traffic volume is too low for a reliable day-to-week estimate.")

    st.markdown("**Summary**")
    pooled_total = rec_pooled[rec_pooled["Direction"] == "Total"].copy()
    for line in summary_lines(pooled_total):
        st.markdown(line)
    st.caption("Summary uses all six intersection pairs (three intersections, each predicting the other two).")

    st.download_button(
        "Download recommendation table (CSV)",
        data=source.drop(columns=["Advice"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="day_to_week_recommendation.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------
# Tab 3: method
# ----------------------------------------------------------------------
with method_tab:
    st.subheader("How the estimate works")
    st.markdown(
        """
**Weekly ADT** is the average daily traffic of one Monday–Sunday week.

**Factor.** For each weekday, the annual factor is the total Weekly ADT divided by the
total traffic counted on that weekday, summed over all complete weeks of the reference
intersection's full year. A factor below 1 means that weekday is busier than an average
day; above 1 means quieter (typically Saturday and Sunday).

**Estimate.** Weekly ADT ≈ counted daily volume × factor of that weekday.

**How we tested it.** Each intersection in turn supplied the full-year factors, and one
day per week of the *other* intersections was used to estimate that week's Weekly ADT.
The estimate was compared with the real Monday–Sunday average. This gives six
intersection pairs, tested for every vehicle type and direction.
        """
    )
    st.subheader("How to read the error numbers")
    st.markdown(
        """
| Measure | Meaning |
|---|---|
| **WAPE** | Total error divided by total traffic. Main measure used for the recommendation. |
| **MAE** | Average error in vehicles per day. |
| **MAPE** | Average of each week's percentage error. Exaggerated when traffic is very low, so it is not used for the recommendation. |
| **Worst 10% of weeks** | 90% of weeks have a percentage error at or below this value. |
| **Bias** | Positive means the estimate is usually too high; negative means too low. |
        """
    )
    st.subheader("Reliability levels")
    st.markdown(
        """
| Level | Rule (best weekday) |
|---|---|
| 🟢 High | WAPE of 10% or less |
| 🟡 Moderate | WAPE of 10% to 20% |
| 🟠 Low | WAPE above 20% |
| 🔴 Insufficient volume | Average Weekly ADT below 10 vehicles/day. A one-day estimate is not recommended. |
        """
    )
    st.warning(
        "Limits: the factors come from three Lincoln intersections during 2025, and they "
        "were tested within the same year. Results may differ for other cities, other years, "
        "or holiday and severe-weather weeks."
    )
