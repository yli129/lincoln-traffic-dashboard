"""Shared helpers for the Lincoln Traffic Dashboard pages.

Every page in dashboard_pages/ can import from here:

    from dashboard_common import (
        CLEANED_FOLDER, discover_intersections, readable_name,
        page_header, placeholder_page,
    )
"""

from pathlib import Path

import streamlit as st

ROOT_FOLDER = Path(__file__).resolve().parent
PROJECT_DATA_FOLDER = ROOT_FOLDER / "Project2_data"
SRC_FOLDER = PROJECT_DATA_FOLDER / "src"
CLEANED_FOLDER = PROJECT_DATA_FOLDER / "cleaned_data"
OUTPUT_FOLDER = PROJECT_DATA_FOLDER / "outputs"

# Vehicle and direction names used across the project
MOTOR_SECTIONS = [
    "Lights",
    "Single-Unit Trucks",
    "Articulated Trucks",
    "Buses",
    "Motorized Vehicles",
]
ALL_MOTOR_VEHICLES = "All Motor Vehicles"
DIRECTIONS = ["Southbound", "Westbound", "Northbound", "Eastbound"]


def discover_intersections(folder=CLEANED_FOLDER):
    """Map each intersection filename prefix to its CSV files."""
    groups = {}
    for file in sorted(Path(folder).glob("*_Binned_*.csv")):
        name = file.name.split("_Binned_", 1)[0]
        groups.setdefault(name, []).append(file)
    return groups


def readable_name(name):
    return name.replace("_and_", " & ").replace("_", " ")


_STYLE = """
<style>
    .block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 3rem; }
    .page-hero {
        background: linear-gradient(120deg, #0b3c5d 0%, #176b87 100%);
        border-radius: 20px; color: white;
        padding: 1.5rem 2rem 1.6rem 2rem; margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(11, 60, 93, 0.14);
    }
    .page-hero .eyebrow {
        color: #a9d7e8; font-size: 0.82rem; font-weight: 700;
        letter-spacing: 0.09em; margin-bottom: 0.45rem; text-transform: uppercase;
    }
    .page-hero h1 {
        color: white; font-size: clamp(1.8rem, 3.4vw, 2.8rem);
        line-height: 1.08; margin: 0;
    }
    .page-hero p { color: #e7f4f9; font-size: 1.02rem; margin: 0.6rem 0 0 0; max-width: 60rem; }
    .reco-card {
        border-left: 6px solid #176b87; background: #f1f8fb; color: #12303f;
        border-radius: 10px; padding: 0.9rem 1.2rem; margin: 0.4rem 0 1rem 0;
    }
    .reco-card.warn { border-left-color: #c97a00; background: #fff6e5; }
    .reco-card.bad { border-left-color: #b3261e; background: #fdecea; }
    .reco-card .label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75; }
    .reco-card .text { font-size: 1.12rem; font-weight: 600; margin-top: 0.15rem; }
</style>
"""


def page_header(eyebrow, title, subtitle):
    """Hero banner shown at the top of every page."""
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def placeholder_page(eyebrow, title, subtitle, owner_note, expected_inputs, expected_outputs):
    """
    Blank page reserved for a teammate.

    To build the page: copy the layout of dashboard_pages/2_Day_to_Week.py,
    delete the call to placeholder_page(), and write your code in its place.
    """
    page_header(eyebrow, title, subtitle)

    st.info(f"🚧 **Under construction.** {owner_note}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Planned inputs**")
        for item in expected_inputs:
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Planned outputs**")
        for item in expected_outputs:
            st.markdown(f"- {item}")

    with st.expander("How to add your page (for teammates)"):
        st.markdown(
            """
1. Open this page's file in `dashboard_pages/` and keep the file name, or the sidebar order will change.
2. Delete the `placeholder_page(...)` call and write your Streamlit code in its place.
3. Reuse the helpers from `dashboard_common.py`:
   `discover_intersections()`, `readable_name()`, `page_header()`, and the folder constants
   (`CLEANED_FOLDER`, `SRC_FOLDER`, `OUTPUT_FOLDER`).
4. Put reusable code in `Project2_data/src/`, experiments and error analysis in `Project2_data/analysis/`,
   and generated tables in `Project2_data/outputs/<your_topic>/`.
   The page should read the generated tables, so it opens quickly.
5. Do not call `st.set_page_config()` in a page. It is set once in `dashboard.py`.
6. Test with `python3 -m streamlit run dashboard.py`, then `git add .`, `git commit`, `git push`.
            """
        )
