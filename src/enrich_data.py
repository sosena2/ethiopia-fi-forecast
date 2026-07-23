"""
Task 1: Enrich the dataset with additional observations, events, and impact_links.
Each new row follows the unified schema exactly, sourced from GSMA and NBE.
"""
import pandas as pd
from datetime import date
from .data_loader import load_datasets, PROCESSED_DIR

COLLECTOR = "Sosena Gossaye"
TODAY = date.today().isoformat()


def build_new_rows(existing_columns: list[str]) -> pd.DataFrame:
    rows = []

    # --- New OBSERVATION: mobile money agent count (Sheet B - direct correlation) ---
    rows.append({
        "record_id": "OBS_NEW_001",
        "record_type": "observation",
        "category": None,
        "pillar": "ACCESS",
        "indicator": "Mobile Money Agent Count",
        "indicator_code": "ACC_MM_AGENTS",
        "indicator_direction": "positive",
        "value_numeric": 200000,
        "value_text": None,
        "value_type": "count",
        "unit": "agents",
        "observation_date": "2022-09-30",
        "source_name": "GSMA Mobile for Development (citing NBE)",
        "source_type": "research",
        "source_url": "https://www.gsma.com/mobilefordevelopment/blog/mobile-money-in-ethiopia-what-we-learnt-from-our-expert-roundtable/",
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "original_text": "mobile money agents grew by 200% in the year to September 2022 to over 200,000",
        "notes": "NBE official data via GSMA roundtable summary. Direct-correlation access indicator "
                 "(Sheet B) — agent density is the primary determinant of cash-in/cash-out accessibility, "
                 "which Findex research links to account activation. Also compare to Telebirr's own "
                 "111,000-agent figure with only 20% weekly-active rate reported in the same article, "
                 "reinforcing the registered-vs-active dormancy pattern already observed in Task 2.",
    })

    # --- New EVENT: NBE Payment Instrument Issuer Directive (leave pillar empty) ---
    rows.append({
        "record_id": "EVT_NEW_001",
        "record_type": "event",
        "category": "policy",
        "pillar": None,
        "indicator": "NBE Payment Instrument Issuer Directive NPS/10/2025",
        "indicator_code": None,
        "observation_date": "2025-05-12",
        "source_name": "National Bank of Ethiopia / Addis Insight",
        "source_type": "regulator",
        "source_url": "https://addisinsight.net/2025/05/27/national-bank-of-ethiopia-issues-new-directive-to-strengthen-digital-payment-ecosystem/",
        "confidence": "high",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "original_text": "Raising the daily electronic money transaction limits to 300,000 Birr and "
                          "150,000 Birr daily electronic money balance... Mandates mandatory interoperability "
                          "between mobile money wallets... Mandates financial institutions to participate in "
                          "the Ethiopian Instant Payment Systems (EIPS)",
        "notes": "Directive took effect May 12, 2025. Raises daily transaction/balance limits and mandates "
                 "wallet-to-wallet interoperability across mobile money operators — directly targets the "
                 "usage/frequency constraints that may explain the registered-vs-active gap found in Task 2.",
    })

    # --- New IMPACT_LINK: directive -> digital payment usage ---
    rows.append({
        "record_type": "impact_link",
        "parent_id": "EVT_NEW_001",   # must match the record_id assigned to the event above once merged
        "category": None,
        "pillar": "USAGE",
        "related_indicator": "USG_P2P_COUNT",
        "relationship_type": "causal",
        "impact_direction": "increase",
        "impact_magnitude": "medium",
        "impact_estimate": None,
        "lag_months": 6,
        "evidence_basis": "theoretical",
        "comparable_country": None,
        "confidence": "medium",
        "collected_by": COLLECTOR,
        "collection_date": TODAY,
        "notes": "Mandatory wallet-to-wallet interoperability under NPS/10/2025 should reduce switching "
                 "friction between Telebirr, M-Pesa, and bank-linked wallets, plausibly increasing P2P "
                 "transaction counts within 6 months. No pre/post Ethiopian data yet available (directive "
                 "is recent as of this analysis), so effect is estimated on theoretical/regulatory-intent "
                 "grounds rather than observed data — flagged medium confidence accordingly.",
    })

    new_df = pd.DataFrame(rows)
    for col in existing_columns:
        if col not in new_df.columns:
            new_df[col] = None
    return new_df[existing_columns]


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