"""
Task 1: Enrich the dataset with additional observations, events, and impact_links.
Each new row must follow the unified schema exactly (35 columns from data_loader).
"""
import pandas as pd
from datetime import date
from .data_loader import load_datasets

COLLECTOR = "Sosena Gossaye"
TODAY = date.today().isoformat()


def build_new_rows(existing_columns: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame of new rows matching `existing_columns`.
    Only fill fields relevant to each record_type; rest stay NaN.
    """
    rows = []

    # --- New OBSERVATION: agent density (Sheet B - direct correlation) ---
    rows.append({
        "record_id": "OBS_NEW_001",
        "record_type": "observation",
        "category": None,
        "pillar": "ACCESS",
        "indicator": "Mobile Money Agent Count",
        "indicator_code": "ACC_MM_AGENTS",
        "indicator_direction": "positive",
        "value_numeric": None,          # <-- fill with real figure once sourced
        "value_text": None,
        "value_type": "count",
        "unit": "agents",
        "observation_date": "2024-12-31",
        "source_name": "GSMA Mobile Money Deployment Tracker",
        "source_type": "research",
        "source_url": "",               # <-- paste exact URL
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "original_text": "",            # <-- exact quote/figure from source
        "notes": "Agent density is a Sheet B direct-correlation indicator for access.",
    })

    # --- New EVENT: leave pillar empty per instructions ---
    rows.append({
        "record_id": "EVT_NEW_001",
        "record_type": "event",
        "category": "policy",
        "pillar": None,
        "indicator": "NBE Digital Payment Regulation Update",
        "indicator_code": None,
        "observation_date": "",         # <-- fill with real date
        "source_name": "",
        "source_type": "policy",
        "source_url": "",
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "original_text": "",
        "notes": "Regulatory change potentially affecting digital payment adoption.",
    })

    # --- New IMPACT_LINK: parent_id must match the event's record_id above ---
    rows.append({
        "record_id": "IMP_NEW_001",
        "parent_id": "EVT_NEW_001",     # <-- must match the event record_id
        "record_type": "impact_link",
        "category": None,
        "pillar": "USAGE",
        "related_indicator": "USG_DIGITAL_PAYMENT",
        "relationship_type": "causal",
        "impact_direction": "positive",
        "impact_magnitude": None,       # <-- estimate, document reasoning in log
        "impact_estimate": None,
        "lag_months": 6,
        "evidence_basis": "comparable_country",
        "comparable_country": "Kenya",
        "confidence": "low",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "notes": "Estimated from comparable regulatory changes in Kenya/Tanzania.",
    })

    new_df = pd.DataFrame(rows)
    # align to existing schema, add any missing cols as NaN, drop any extras
    for col in existing_columns:
        if col not in new_df.columns:
            new_df[col] = None
    return new_df[existing_columns]


from .data_loader import PROCESSED_DIR

def enrich_and_save(df: pd.DataFrame, out_path=None):
    if out_path is None:
        out_path = PROCESSED_DIR / "ethiopia_fi_enriched.csv"
    new_rows = build_new_rows(df.columns.tolist())
    enriched = pd.concat([df, new_rows], ignore_index=True)
    enriched.to_csv(out_path, index=False)
    print(f"Saved {len(enriched)} total rows ({len(new_rows)} new) to {out_path}")
    return enriched


if __name__ == "__main__":
    df, ref = load_datasets()
    enrich_and_save(df)