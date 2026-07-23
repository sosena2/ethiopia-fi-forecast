"""
Task 4: Forecast Access (Account Ownership) and Usage (Digital Payments)
for 2025-2027, using trend regression augmented with event effects,
under optimistic/base/pessimistic scenarios.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from .data_loader import load_datasets
from .impact_model import build_event_indicator_table, combine_event_effects

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

FORECAST_YEARS = [2025, 2026, 2027]
SCENARIO_MULTIPLIERS = {"pessimistic": 0.5, "base": 1.0, "optimistic": 1.5}


def fit_trend(obs: pd.DataFrame, indicator_code: str):
    """Linear trend regression on survey-wave data for one indicator."""
    series = obs[obs["indicator_code"] == indicator_code].sort_values("observation_date")
    series["year"] = pd.to_datetime(series["observation_date"]).dt.year
    x = series["year"].values
    y = series["value_numeric"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope": slope, "intercept": intercept, "r2": r_value ** 2,
        "std_err": std_err, "last_year": x.max(), "last_value": y[-1],
        "x": x, "y": y,
    }


def forecast_trend_only(trend: dict, years: list) -> pd.DataFrame:
    """Baseline: simple trend continuation with widening confidence interval."""
    rows = []
    for yr in years:
        pred = trend["intercept"] + trend["slope"] * yr
        years_out = yr - trend["last_year"]
        # widen uncertainty the further out we forecast
        margin = trend["std_err"] * years_out * 1.96
        rows.append({
            "year": yr, "forecast": pred,
            "lower_95": pred - margin, "upper_95": pred + margin,
        })
    return pd.DataFrame(rows)


def forecast_event_augmented(trend: dict, event_table: pd.DataFrame,
                              indicator_code: str, years: list,
                              scenario: str = "base") -> pd.DataFrame:
    """Trend + cumulative event effects, scaled by scenario multiplier."""
    mult = SCENARIO_MULTIPLIERS[scenario]
    baseline = forecast_trend_only(trend, years)

    effect_series = combine_event_effects(
        event_table, indicator_code,
        start_date=f"{trend['last_year']}-01-01",
        horizon_months=(max(years) - trend["last_year"] + 1) * 12,
    )

    augmented_rows = []
    for _, row in baseline.iterrows():
        yr = int(row["year"])
        month_idx = (yr - trend["last_year"]) * 12
        month_idx = min(month_idx, len(effect_series) - 1)
        event_boost = effect_series.iloc[month_idx] * mult * 0.3  # damping factor
        augmented_rows.append({
            "year": yr, "scenario": scenario,
            "forecast": row["forecast"] + event_boost,
            "lower_95": row["lower_95"] + event_boost * 0.5,
            "upper_95": row["upper_95"] + event_boost * 1.5,
        })
    return pd.DataFrame(augmented_rows)


def run_all_scenarios(trend: dict, event_table: pd.DataFrame,
                       indicator_code: str, years: list) -> pd.DataFrame:
    frames = [forecast_event_augmented(trend, event_table, indicator_code, years, s)
              for s in SCENARIO_MULTIPLIERS]
    return pd.concat(frames, ignore_index=True)


def plot_forecast(trend: dict, scenarios_df: pd.DataFrame, indicator_name: str,
                   filename: str):
    plt.figure(figsize=(11, 6))
    plt.plot(trend["x"], trend["y"], marker="o", color="black",
             linewidth=2, label="Historical", zorder=5)

    colors = {"pessimistic": "#e74c3c", "base": "#2980b9", "optimistic": "#27ae60"}
    for scenario, color in colors.items():
        sub = scenarios_df[scenarios_df["scenario"] == scenario]
        years_full = [trend["last_year"]] + sub["year"].tolist()
        vals_full = [trend["last_value"]] + sub["forecast"].tolist()
        plt.plot(years_full, vals_full, marker="s", linestyle="--",
                  color=color, label=f"{scenario.title()}")
        plt.fill_between(sub["year"], sub["lower_95"], sub["upper_95"],
                          color=color, alpha=0.1)

    plt.title(f"{indicator_name} Forecast: 2025–2027")
    plt.ylabel("% of adults")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/{filename}")


def run_forecasting():
    df, ref = load_datasets()
    obs = df[df["record_type"] == "observation"].copy()
    event_table = build_event_indicator_table(df)

    targets = {
        "ACC_OWNERSHIP": ("Account Ownership (Access)", "access_forecast.png"),
        "ACC_MM_ACCOUNT": ("Mobile Money Account Penetration (Usage proxy)", "usage_forecast.png"),
    }

    all_results = {}
    for code, (label, fname) in targets.items():
        trend = fit_trend(obs, code)
        print(f"\n=== {label} — Trend fit ===")
        print(f"Slope: {trend['slope']:.3f} pp/year | R²: {trend['r2']:.3f}")

        scenarios_df = run_all_scenarios(trend, event_table, code, FORECAST_YEARS)
        print(scenarios_df.to_string(index=False))

        plot_forecast(trend, scenarios_df, label, fname)
        scenarios_df.to_csv(f"data/processed/forecast_{code}.csv", index=False)
        all_results[code] = scenarios_df

    return all_results


if __name__ == "__main__":
    results = run_forecasting()