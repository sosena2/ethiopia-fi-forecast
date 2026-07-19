"""
Task 2: Exploratory Data Analysis on Ethiopia's financial inclusion dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .data_loader import load_datasets, profile_dataset

sns.set_theme(style="whitegrid")
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def dataset_overview(df: pd.DataFrame):
    """Section 1: Summarize by record_type, pillar, source_type; temporal coverage; confidence."""
    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    summary = df.groupby(["record_type", "pillar"]).size().unstack(fill_value=0)
    print("\n--- record_type x pillar ---")
    print(summary)

    obs = df[df["record_type"] == "observation"].copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"])
    obs["year"] = obs["observation_date"].dt.year

    coverage = obs.pivot_table(index="indicator_code", columns="year",
                                values="value_numeric", aggfunc="count", fill_value=0)
    plt.figure(figsize=(14, 8))
    sns.heatmap(coverage.astype(bool).astype(int), cmap="Greens", cbar=False,
                linewidths=0.5, linecolor="white")
    plt.title("Temporal Coverage by Indicator")
    plt.xlabel("Year")
    plt.ylabel("Indicator Code")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/temporal_coverage.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/temporal_coverage.png")

    conf_counts = df["confidence"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(conf_counts, labels=conf_counts.index, autopct="%1.0f%%",
            colors=["#2ecc71", "#f39c12", "#e74c3c"])
    plt.title("Data Quality: Confidence Level Distribution")
    plt.savefig(f"{FIG_DIR}/confidence_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/confidence_distribution.png")

    sparse = coverage.sum(axis=1).sort_values()
    print("\n--- Sparsest indicators (fewest data points) ---")
    print(sparse.head(8))

    return obs


def access_analysis(obs: pd.DataFrame):
    """Section 2: Account ownership trajectory, growth rates, gender gap."""
    print("\n" + "=" * 60)
    print("2. ACCESS ANALYSIS")
    print("=" * 60)

    acc = obs[obs["indicator_code"] == "ACC_OWNERSHIP"].sort_values("observation_date")
    print("\n--- Account Ownership over time ---")
    print(acc[["observation_date", "value_numeric"]])

    acc = acc.reset_index(drop=True)
    acc["pp_change"] = acc["value_numeric"].diff()
    acc["years_elapsed"] = acc["observation_date"].dt.year.diff()
    acc["annualized_pp"] = acc["pp_change"] / acc["years_elapsed"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(acc["observation_date"], acc["value_numeric"], marker="o",
              linewidth=2, color="#2980b9", label="Account Ownership %")
    ax1.set_ylabel("Account Ownership (%)")
    ax1.set_title("Ethiopia Account Ownership Trajectory (2011–2024)")
    for _, row in acc.iterrows():
        ax1.annotate(f"{row['value_numeric']:.0f}%",
                     (row["observation_date"], row["value_numeric"]),
                     textcoords="offset points", xytext=(0, 10))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/access_trajectory.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/access_trajectory.png")

    print("\n--- Growth rate (pp/year) between survey waves ---")
    print(acc[["observation_date", "pp_change", "annualized_pp"]])

    # Gender gap
    gen = obs[obs["indicator_code"] == "GEN_GAP_ACC"].sort_values("observation_date")
    if len(gen):
        print("\n--- Gender Gap in Account Ownership ---")
        print(gen[["observation_date", "value_numeric", "gender", "notes"]])
        plt.figure(figsize=(8, 5))
        plt.plot(gen["observation_date"], gen["value_numeric"], marker="s",
                 color="#c0392b", linewidth=2)
        plt.title("Gender Gap in Account Ownership Over Time")
        plt.ylabel("Gap (percentage points)")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/gender_gap_access.png", dpi=150)
        plt.close()
        print(f"Saved: {FIG_DIR}/gender_gap_access.png")

    # 2021-2024 slowdown flag
    slowdown = acc[acc["observation_date"].dt.year.isin([2021, 2024])]
    if len(slowdown) == 2:
        delta = slowdown["value_numeric"].diff().iloc[-1]
        print(f"\n>>> 2021-2024 change: {delta:+.1f}pp "
              f"(vs. prior periods averaging {acc['pp_change'].iloc[:-1].mean():.1f}pp)")

    return acc


def usage_analysis(obs: pd.DataFrame):
    """Section 3: Mobile money penetration, digital payments, active rate."""
    print("\n" + "=" * 60)
    print("3. USAGE (DIGITAL PAYMENTS) ANALYSIS")
    print("=" * 60)

    usage_codes = ["ACC_MM_ACCOUNT", "USG_ACTIVE_RATE", "USG_TELEBIRR_USERS",
                   "USG_MPESA_USERS", "USG_MPESA_ACTIVE", "USG_CROSSOVER"]
    usage = obs[obs["indicator_code"].isin(usage_codes)].sort_values("observation_date")
    print(usage[["indicator_code", "observation_date", "value_numeric", "unit"]])

    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")
    if len(mm) >= 2:
        plt.figure(figsize=(9, 5))
        plt.plot(mm["observation_date"], mm["value_numeric"], marker="o",
                 color="#27ae60", linewidth=2)
        plt.title("Mobile Money Account Penetration (2014–2024)")
        plt.ylabel("% of adults")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/mobile_money_penetration.png", dpi=150)
        plt.close()
        print(f"\nSaved: {FIG_DIR}/mobile_money_penetration.png")

    # Registered vs active gap
    reg = obs[obs["indicator_code"] == "USG_TELEBIRR_USERS"]["value_numeric"]
    active = obs[obs["indicator_code"] == "USG_ACTIVE_RATE"]["value_numeric"]
    if len(reg) and len(active):
        print(f"\n>>> Registered Telebirr users vs. active rate — "
              f"registered: {reg.values}, active_rate: {active.values}")
        print(">>> Large registered-vs-active gap indicates dormant accounts inflate")
        print("    headline registration numbers relative to true usage.")

    return usage


def infrastructure_analysis(obs: pd.DataFrame):
    """Section 4: 4G coverage, affordability, mobile penetration as leading indicators."""
    print("\n" + "=" * 60)
    print("4. INFRASTRUCTURE & ENABLERS")
    print("=" * 60)

    infra_codes = ["ACC_4G_COV", "ACC_MOBILE_PEN", "AFF_DATA_INCOME", "ACC_FAYDA"]
    infra = obs[obs["indicator_code"].isin(infra_codes)].sort_values("observation_date")
    print(infra[["indicator_code", "observation_date", "value_numeric", "unit"]])

    fig, ax = plt.subplots(figsize=(10, 6))
    for code in infra_codes:
        sub = obs[obs["indicator_code"] == code].sort_values("observation_date")
        if len(sub):
            ax.plot(sub["observation_date"], sub["value_numeric"], marker="o", label=code)
    ax.set_title("Infrastructure & Enabler Trends")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/infrastructure_trends.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/infrastructure_trends.png")

    return infra


def event_timeline(df: pd.DataFrame, obs: pd.DataFrame):
    """Section 5: Event timeline overlaid on Access/Usage trends."""
    print("\n" + "=" * 60)
    print("5. EVENT TIMELINE & OVERLAY")
    print("=" * 60)

    events = df[df["record_type"] == "event"].copy()
    events["observation_date"] = pd.to_datetime(events["observation_date"])
    events = events.sort_values("observation_date")
    print(events[["record_id", "category", "indicator", "observation_date"]])

    fig, ax = plt.subplots(figsize=(13, 7))
    acc = obs[obs["indicator_code"] == "ACC_OWNERSHIP"].sort_values("observation_date")
    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")

    ax.plot(acc["observation_date"], acc["value_numeric"], marker="o",
            linewidth=2, label="Account Ownership", color="#2980b9")
    ax.plot(mm["observation_date"], mm["value_numeric"], marker="s",
            linewidth=2, label="Mobile Money Accounts", color="#27ae60")

    colors = plt.cm.tab10(np.linspace(0, 1, len(events)))
    for (_, ev), c in zip(events.iterrows(), colors):
        ax.axvline(ev["observation_date"], color=c, linestyle="--", alpha=0.5)
        ax.text(ev["observation_date"], ax.get_ylim()[1] * 0.95,
                str(ev["indicator"])[:20], rotation=90, fontsize=7,
                va="top", color=c)

    ax.set_title("Access & Usage Trends with Event Timeline Overlay")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/event_timeline_overlay.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/event_timeline_overlay.png")

    return events


def correlation_analysis(obs: pd.DataFrame):
    """Section 6: Correlation matrix across indicators (where overlapping years exist)."""
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS")
    print("=" * 60)

    obs["year"] = obs["observation_date"].dt.year
    pivot = obs.pivot_table(index="year", columns="indicator_code",
                             values="value_numeric", aggfunc="mean")

    # Drop indicators with too few overlapping data points
    min_points = 3
    valid_cols = pivot.columns[pivot.notna().sum() >= min_points]
    pivot = pivot[valid_cols]

    if pivot.shape[1] < 2:
        print("Not enough overlapping indicators for meaningful correlation. "
              "Consider interpolation or focus on qualitative event-overlay analysis instead.")
        return None

    corr = pivot.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True)
    plt.title("Indicator Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/correlation_matrix.png")
    print("\n--- Correlation matrix ---")
    print(corr.round(2))

    return corr


def impact_link_summary(df: pd.DataFrame):
    """Section 7: Summarize existing impact_link relationships."""
    print("\n" + "=" * 60)
    print("7. IMPACT LINK SUMMARY")
    print("=" * 60)

    links = df[df["record_type"] == "impact_link"]
    events = df[df["record_type"] == "event"][["record_id", "indicator", "category"]]
    events = events.rename(columns={"record_id": "parent_id", "indicator": "event_name"})

    merged = links.merge(events, on="parent_id", how="left")
    summary = merged[["event_name", "category", "related_indicator",
                       "impact_direction", "impact_magnitude", "lag_months",
                       "evidence_basis", "confidence"]]
    print(summary.to_string(index=False))

    return summary


def run_full_eda():
    df, ref = load_datasets()
    profile_dataset(df)

    obs = dataset_overview(df)
    acc = access_analysis(obs)
    usage = usage_analysis(obs)
    infra = infrastructure_analysis(obs)
    events = event_timeline(df, obs)
    corr = correlation_analysis(obs)
    links = impact_link_summary(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE — all figures saved to reports/figures/")
    print("=" * 60)

    return {
        "obs": obs, "acc": acc, "usage": usage, "infra": infra,
        "events": events, "corr": corr, "links": links,
    }


if __name__ == "__main__":
    results = run_full_eda()