"""Run with: python3 -m streamlit run dashboard.py

Lincoln Traffic Dashboard: one app, four pages.

    1_AADT           placeholder
    2_Day_to_Week    done
    3_Week_to_Month  placeholder
    4_Day_to_Hour    placeholder

Requires Streamlit 1.36 or newer (st.navigation).
"""

from pathlib import Path

import streamlit as st

PAGES_FOLDER = Path(__file__).resolve().parent / "dashboard_pages"

# Set once here; pages must not call st.set_page_config themselves.
st.set_page_config(page_title="Lincoln Traffic Dashboard", page_icon="🚦", layout="wide")

pages = [
    st.Page(str(PAGES_FOLDER / "1_AADT.py"), title="AADT", icon="📅", default=True),
    st.Page(str(PAGES_FOLDER / "2_Day_to_Week.py"), title="Day to Week", icon="📆"),
    st.Page(str(PAGES_FOLDER / "3_Week_to_Month.py"), title="Week to Month", icon="🗓️"),
    st.Page(str(PAGES_FOLDER / "4_Day_to_Hour.py"), title="Day to Hour", icon="⏱️"),
]

navigation = st.navigation(pages)

with st.sidebar:
    st.caption("Lincoln, Nebraska · 15-minute traffic counts · 2025")

navigation.run()
