"""
Task 3: Event Impact Modeling.
Translates impact_link records into a model that estimates how events
shift indicators over time (immediate or gradual, combined across events).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .data_loader import load_datasets

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

KEY_INDICATORS = ["ACC_OWNERSHIP", "ACC_MM_ACCOUNT", "USG_DIGITAL_PAYMENT",
                   "USG_ACTIVE_RATE", "USG_TELEBIRR_USERS", "USG_MPESA_USERS"]


def build_event_indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    """Join impact_links with parent events to get a flat event->indicator table."""
    # Pre-select only the columns needed from links, so its own `category`
    # doesn't collide with the events' `category` after merging.
    links = df[df["record_type"] == "impact_link"][
        ["parent_id", "related_indicator", "impact_direction", "impact_magnitude",
         "lag_months", "evidence_basis", "comparable_country", "confidence"]
    ].copy()

    events = df[df["record_type"] == "event"][
        ["record_id", "indicator", "category", "observation_date"]
    ].rename(columns={"record_id": "parent_id", "indicator": "event_name",
                       "category": "event_category",
                       "observation_date": "event_date"})

    merged = links.merge(events, on="parent_id", how="left")
    merged["event_date"] = pd.to_datetime(merged["event_date"])
    return merged[["parent_id", "event_name", "event_category", "event_date",
                    "related_indicator", "impact_direction", "impact_magnitude",
                    "lag_months", "evidence_basis", "comparable_country", "confidence"]]


# Maps the dataset's text magnitude labels to numeric effect sizes.
MAGNITUDE_SCALE = {"low": 1.0, "medium": 2.0, "high": 3.0}


def build_association_matrix(event_table: pd.DataFrame) -> pd.DataFrame:
    """
    Rows: events, Columns: indicators, Values: signed estimated effect.
    impact_magnitude arrives as text ('low'/'medium'/'high') and is mapped
    to a numeric scale; impact_direction arrives as 'increase'/'decrease'
    (not 'positive'/'negative') and is mapped to a sign.
    Missing magnitude defaults to a nominal 1.0, flagged low confidence.
    """
    et = event_table.copy()

    def signed_magnitude(row):
        mag = MAGNITUDE_SCALE.get(str(row["impact_magnitude"]).lower(), 1.0)
        sign = 1 if str(row["impact_direction"]).lower() == "increase" else -1
        return sign * mag

    et["signed_effect"] = et.apply(signed_magnitude, axis=1).astype(float)

    matrix = et.pivot_table(
        index="event_name", columns="related_indicator",
        values="signed_effect", aggfunc="sum", fill_value=0.0
    )
    return matrix


def plot_association_heatmap(matrix: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="white")
    plt.title("Event–Indicator Association Matrix (Estimated Effect Size)")
    plt.ylabel("Event")
    plt.xlabel("Indicator")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/association_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/association_matrix.png")


def event_effect_curve(magnitude: float, lag_months: int, horizon_months: int = 36,
                        ramp_months: int = 12) -> np.ndarray:
    """
    Model an event's effect over time as a delayed sigmoid ramp:
    - No effect before `lag_months`
    - Effect ramps up over `ramp_months` after the lag
    - Effect plateaus at `magnitude` afterward
    This reflects that policy/product effects build gradually, not instantly.
    """
    t = np.arange(horizon_months)
    effect = np.zeros(horizon_months)
    for i, month in enumerate(t):
        if month < lag_months:
            effect[i] = 0
        else:
            progress = min((month - lag_months) / ramp_months, 1.0)
            # smootherstep ramp
            s = progress * progress * (3 - 2 * progress)
            effect[i] = magnitude * s
    return effect

def combine_event_effects(event_table: pd.DataFrame, indicator_code: str,
                           start_date: str, horizon_months: int = 60) -> pd.Series:
    """
    Sum effect curves from all events impacting `indicator_code`, additively,
    aligned to a monthly timeline starting at `start_date`.
    """
    dates = pd.date_range(start=start_date, periods=horizon_months, freq="MS")
    total_effect = pd.Series(0.0, index=dates)

    relevant = event_table[event_table["related_indicator"] == indicator_code]

    for _, ev in relevant.iterrows():
        if pd.isna(ev["event_date"]):
            continue
        mag = MAGNITUDE_SCALE.get(str(ev["impact_magnitude"]).lower(), 1.0)
        sign = 1 if str(ev["impact_direction"]).lower() == "increase" else -1
        lag = int(ev["lag_months"]) if pd.notna(ev["lag_months"]) else 0

        months_from_start = (ev["event_date"].year - dates[0].year) * 12 + \
                             (ev["event_date"].month - dates[0].month)
        curve = event_effect_curve(sign * mag, lag, horizon_months=horizon_months)

        # shift curve to align with event's actual start month
        shifted = np.zeros(horizon_months)
        for i in range(horizon_months):
            src = i - months_from_start
            if 0 <= src < horizon_months:
                shifted[i] = curve[src]

        total_effect += pd.Series(shifted, index=dates)

    return total_effect


def validate_against_historical(df: pd.DataFrame, event_table: pd.DataFrame):
    """
    Sanity check: Telebirr launched May 2021; ACC_MM_ACCOUNT went 4.7% (2021)
    -> 9.45% (2024). Compare modeled cumulative effect vs. actual change.
    """
    print("\n=== Validation: Telebirr launch vs. ACC_MM_ACCOUNT ===")
    obs = df[df["record_type"] == "observation"].copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"])

    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")
    print(mm[["observation_date", "value_numeric"]])

    if len(mm) >= 2:
        actual_change = mm["value_numeric"].iloc[-1] - mm["value_numeric"].iloc[0]
        print(f"\nActual observed change: {actual_change:+.2f}pp")

    effect_series = combine_event_effects(event_table, "ACC_MM_ACCOUNT",
                                           start_date="2021-01-01", horizon_months=48)
    modeled_change = effect_series.iloc[-1] - effect_series.iloc[0]
    print(f"Modeled cumulative effect (all events): {modeled_change:+.2f}")
    print("\nNote: modeled units are relative effect-size scores, not directly "
          "comparable to pp unless impact_magnitude was calibrated in pp terms — "
          "document this unit assumption explicitly in your methodology write-up.")

    return effect_series


def run_impact_modeling():
    df, ref = load_datasets()
    event_table = build_event_indicator_table(df)
    print("=== Event-Indicator Table ===")
    print(event_table.to_string(index=False))

    matrix = build_association_matrix(event_table)
    print("\n=== Association Matrix ===")
    print(matrix)
    plot_association_heatmap(matrix)

    validate_against_historical(df, event_table)

    matrix.to_csv("data/processed/association_matrix.csv")
    print("\nSaved: data/processed/association_matrix.csv")

    return {"event_table": event_table, "matrix": matrix}


if __name__ == "__main__":
    results = run_impact_modeling()