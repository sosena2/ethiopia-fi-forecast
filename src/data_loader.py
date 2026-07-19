"""
Task 1: Load and explore the unified financial inclusion dataset.
"""
import pandas as pd
import glob
import os
from pathlib import Path

# Anchor paths to the project root, regardless of where this is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILE = RAW_DIR / "ethiopia_fi_unified_data.xlsx"
DATA_SHEET = "ethiopia_fi_unified_data"
IMPACT_SHEET = "Impact_sheet"


def find_reference_file():
    """Locate reference_codes file regardless of exact name/extension."""
    candidates = list(RAW_DIR.glob("*eference*cod*"))
    if not candidates:
        raise FileNotFoundError(
            f"No reference codes file found in {RAW_DIR}. "
            f"Files present: {os.listdir(RAW_DIR)}"
        )
    return candidates[0]


def load_datasets():
    """Load both sheets from the workbook, combine, and load reference codes."""
    main_df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    impact_df = pd.read_excel(DATA_FILE, sheet_name=IMPACT_SHEET)

    df = pd.concat([main_df, impact_df], ignore_index=True, sort=False)

    ref_path = find_reference_file()
    if str(ref_path).endswith(".csv"):
        ref = pd.read_csv(ref_path)
    else:
        ref = pd.read_excel(ref_path)

    return df, ref


def profile_dataset(df: pd.DataFrame):
    """Print counts by record_type, pillar, source_type, confidence."""
    print("=== All columns (after combining sheets) ===")
    print(list(df.columns), "\n")

    print("=== Records by record_type ===")
    print(df["record_type"].value_counts(), "\n")

    if "pillar" in df.columns:
        print("=== Records by pillar ===")
        print(df["pillar"].value_counts(dropna=False), "\n")

    if "source_type" in df.columns:
        print("=== Records by source_type ===")
        print(df["source_type"].value_counts(dropna=False), "\n")

    if "confidence" in df.columns:
        print("=== Records by confidence ===")
        print(df["confidence"].value_counts(dropna=False), "\n")

    obs = df[df["record_type"] == "observation"]
    if "observation_date" in df.columns and len(obs):
        print("=== Temporal range of observations ===")
        print(f"{obs['observation_date'].min()} to {obs['observation_date'].max()}\n")

    if "indicator_code" in df.columns:
        print("=== Unique indicators (indicator_code) ===")
        print(obs["indicator_code"].value_counts(), "\n")

    events = df[df["record_type"] == "event"]
    print(f"=== Cataloged events: {len(events)} ===")
    print(events.head(10), "\n")

    links = df[df["record_type"] == "impact_link"]
    print(f"=== Impact links: {len(links)} total ===")
    print(links.head(10))

    return {
        "obs": obs,
        "events": events,
        "links": links,
        "targets": df[df["record_type"] == "target"],
    }


if __name__ == "__main__":
    df, ref = load_datasets()
    print(f"Loaded {len(df)} total rows, {len(df.columns)} columns\n")
    parts = profile_dataset(df)