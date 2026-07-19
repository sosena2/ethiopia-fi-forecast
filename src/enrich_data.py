"""
Task 1: Enrich the dataset with additional observations, events, and impact_links.
Each new row must follow the unified schema exactly.
"""
import pandas as pd
from datetime import date

COLLECTOR = "Sosena Gossaye"
TODAY = date.today().isoformat()


def build_new_rows(existing_columns: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame of new rows matching `existing_columns`.
    Fill in only the fields relevant to each record_type; leave rest blank/NaN.
    """
    rows = []

    # --- Example: new OBSERVATION (Sheet B - Direct Correlation: agent density) ---
    rows.append({
        "record_type": "observation",
        "pillar": "access",
        "indicator": "Mobile Money Agent Count",
        "indicator_code": "ACC_MM_AGENTS",
        "value_numeric": None,          # <-- fill with real figure once sourced
        "observation_date": "2024-12-31",
        "source_name": "GSMA Mobile Money Deployment Tracker",
        "source_url": "",               # <-- paste exact URL
        "original_text": "",            # <-- exact quote/figure from source
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "notes": "Agent density is a Sheet B direct-correlation indicator for access.",
    })

    # --- Example: new EVENT (leave pillar empty per instructions) ---
    rows.append({
        "record_type": "event",
        "pillar": None,
        "event_name": "NBE Digital Payment Regulation Update",
        "category": "policy",
        "observation_date": "",         # <-- fill with real date
        "source_name": "",
        "source_url": "",
        "original_text": "",
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "notes": "Regulatory change potentially affecting digital payment adoption.",
    })

    # --- Example: new IMPACT_LINK (parent_id must match an event's id) ---
    rows.append({
        "record_type": "impact_link",
        "parent_id": "",                # <-- id of the event above once assigned
        "pillar": "usage",
        "related_indicator": "USG_DIGITAL_PAYMENT",
        "impact_direction": "positive",
        "impact_magnitude": None,       # <-- estimate, document reasoning in log
        "lag_months": 6,
        "evidence_basis": "comparable_country",
        "confidence": "low",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "notes": "Estimated from comparable regulatory changes in Kenya/Tanzania.",
    })

    new_df = pd.DataFrame(rows)
    # align to existing schema, add any missing cols as NaN
    for col in existing_columns:
        if col not in new_df.columns:
            new_df[col] = None
    return new_df[existing_columns]


def enrich_and_save(df: pd.DataFrame, out_path: str = "data/processed/ethiopia_fi_enriched.csv"):
    new_rows = build_new_rows(df.columns.tolist())
    enriched = pd.concat([df, new_rows], ignore_index=True)
    enriched.to_csv(out_path, index=False)
    print(f"Saved {len(enriched)} total rows ({len(new_rows)} new) to {out_path}")
    return enriched


if __name__ == "__main__":
    from .data_loader import load_datasets
    df, ref = load_datasets()
    enrich_and_save(df)