# Lincoln Traffic Dashboard (4 pages)

One Streamlit app with four pages:

| Page | File | Status |
|---|---|---|
| AADT | `dashboard_pages/1_AADT.py` | placeholder |
| Day to Week | `dashboard_pages/2_Day_to_Week.py` | done |
| Week to Month | `dashboard_pages/3_Week_to_Month.py` | placeholder |
| Day to Hour | `dashboard_pages/4_Day_to_Hour.py` | placeholder |

## Run

```bash
pip install "streamlit>=1.36" pandas numpy matplotlib scipy
python3 -m streamlit run dashboard.py
```

`dashboard.py` is the entry point. It sets the page config once and lists the pages.
The older `new_app.py` and `pages/2_Annual_Estimation.py` are not changed.

## Files

```
dashboard.py                      entry point and page list
dashboard_common.py               shared helpers: folders, readable_name(), page_header(), placeholder_page()
dashboard_pages/                  one file per page
Project2_data/src/                reusable code
    shared_traffic_utils.py       load data, aggregate 15-minute bins into complete days
    day_to_week.py                week tables, annual factors, estimation, errors
Project2_data/analysis/           experiments and error analysis
    day_to_week_recommend.py      tests all intersection pairs, writes the tables the page reads
Project2_data/outputs/day_to_week/   tables read by the Day-to-Week page
```

## Day-to-Week page: refresh the tables

The page only reads CSV files, so it opens quickly. Regenerate them whenever the
cleaned data changes (takes a few minutes):

```bash
python3 Project2_data/analysis/day_to_week_recommend.py
```

It reads `Project2_data/cleaned_data/*_Binned_*.csv` and writes `Project2_data/outputs/day_to_week/`.

## Adding your page (AADT, Week to Month, Day to Hour)

1. Open your file in `dashboard_pages/` and keep its name.
2. Delete the `placeholder_page(...)` call and write your code in its place.
3. Use `from dashboard_common import ...` for the shared helpers.
4. Do not call `st.set_page_config()` in a page; `dashboard.py` does it once.
5. Put reusable code in `Project2_data/src/`, experiments and error analysis in
   `Project2_data/analysis/`, and generated tables in
   `Project2_data/outputs/<your_topic>/`, then let the page read the tables.
6. Run the app, then `git add .`, `git commit -m "..."`, `git push`.
