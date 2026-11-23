# Lincoln Traffic Analysis Dashboard

A traffic data analysis and estimation project for Lincoln, Nebraska.

## Project Structure

All project-related files should be stored in `Project2_data/`.

| Folder | Description |
|---|---|
| `raw_data/` | Original traffic datasets |
| `src/` | Python code for data checking, cleaning, and estimation |
| `cleaned_data/` | Cleaned traffic datasets |
| `analysis/` | Experiments and error analysis |
| `outputs/` | Generated CSV results, figures, and reports |

## Data Usage

- Use datasets from `cleaned_data/`, not `raw_data/`.
- Only use records where `Quality_Flag == "Valid"` for analysis.
- Keep original datasets unchanged.

## Data Processing

Run Data Checking before Data Repair:

```bash
python Project2_data/src/check_10_csv_simple.py
python Project2_data/src/data_repair.py
```

## Collaboration

Please place new files in the appropriate folders before pushing changes to GitHub.

