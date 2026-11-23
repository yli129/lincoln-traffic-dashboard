"""AADT page (placeholder). Replace the placeholder_page() call with your code."""

from dashboard_common import placeholder_page

placeholder_page(
    eyebrow="AADT",
    title="Annual Average Daily Traffic",
    subtitle="Estimate AADT from a short count period.",
    owner_note=(
        "This page is reserved for the AADT part of the project. "
        "An earlier prototype exists in `pages/2_Annual_Estimation.py`; "
        "it can be moved here when the page is ready."
    ),
    expected_inputs=[
        "Reference intersection with full-year data",
        "Target intersection and count duration",
    ],
    expected_outputs=[
        "Estimated AADT and its expected error",
        "Recommended count duration",
    ],
)

# === YOUR CODE STARTS HERE ===
