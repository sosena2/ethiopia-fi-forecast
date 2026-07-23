"""
Task 5: Interactive Streamlit dashboard for Ethiopia Financial Inclusion Forecast.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_datasets

st.set_page_config(page_title="Ethiopia Financial Inclusion Forecast", layout="wide")


@st.cache_data
def get_data():
    df, ref = load_datasets()
    obs = df[df["record_type"] == "observation"].copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"])
    events = df[df["record_type"] == "event"].copy()
    events["observation_date"] = pd.to_datetime(events["observation_date"])
    return df, obs, events


df, obs, events = get_data()

st.sidebar.title("Ethiopia FI Forecast")
page = st.sidebar.radio("Navigate", ["Overview", "Trends", "Forecasts", "Inclusion Projections"])

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("Financial Inclusion in Ethiopia — Overview")

    acc = obs[obs["indicator_code"] == "ACC_OWNERSHIP"].sort_values("observation_date")
    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")
    p2p = obs[obs["indicator_code"] == "USG_P2P_COUNT"].sort_values("observation_date")
    atm = obs[obs["indicator_code"] == "USG_ATM_COUNT"].sort_values("observation_date")

    col1, col2, col3, col4 = st.columns(4)
    if len(acc):
        col1.metric("Account Ownership", f"{acc['value_numeric'].iloc[-1]:.0f}%",
                     f"{acc['value_numeric'].diff().iloc[-1]:+.0f}pp" if len(acc) > 1 else None)
    if len(mm):
        col2.metric("Mobile Money Accounts", f"{mm['value_numeric'].iloc[-1]:.1f}%",
                     f"{mm['value_numeric'].diff().iloc[-1]:+.1f}pp" if len(mm) > 1 else None)
    if len(p2p) and len(atm):
        ratio = p2p["value_numeric"].iloc[-1] / atm["value_numeric"].iloc[-1] if atm["value_numeric"].iloc[-1] else None
        col3.metric("P2P/ATM Crossover Ratio", f"{ratio:.1f}x" if ratio else "N/A")
    col4.metric("Events Tracked", len(events))

    st.subheader("Growth Highlights")
    if len(acc) > 1:
        acc_growth = acc[["observation_date", "value_numeric"]].copy()
        acc_growth["pp_change"] = acc_growth["value_numeric"].diff()
        st.dataframe(acc_growth, use_container_width=True)

# ---------------- TRENDS ----------------
elif page == "Trends":
    st.title("Indicator Trends")

    indicators = obs["indicator_code"].unique().tolist()
    selected = st.multiselect("Select indicators to compare", indicators,
                               default=["ACC_OWNERSHIP", "ACC_MM_ACCOUNT"])

    date_range = st.slider(
        "Date range",
        min_value=obs["observation_date"].min().to_pydatetime(),
        max_value=obs["observation_date"].max().to_pydatetime(),
        value=(obs["observation_date"].min().to_pydatetime(),
               obs["observation_date"].max().to_pydatetime()),
    )

    filtered = obs[
        (obs["indicator_code"].isin(selected)) &
        (obs["observation_date"] >= date_range[0]) &
        (obs["observation_date"] <= date_range[1])
    ]

    if len(filtered):
        fig = px.line(filtered, x="observation_date", y="value_numeric",
                       color="indicator_code", markers=True,
                       title="Selected Indicators Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Event Timeline")
    if len(events):
        fig2 = px.scatter(events, x="observation_date", y="category",
                           hover_data=["indicator"], color="category",
                           title="Cataloged Events")
        st.plotly_chart(fig2, use_container_width=True)

    st.download_button("Download filtered data (CSV)",
                        filtered.to_csv(index=False), "filtered_indicators.csv")

# ---------------- FORECASTS ----------------
elif page == "Forecasts":
    st.title("Forecasts: Access & Usage (2025–2027)")

    try:
        acc_fc = pd.read_csv("data/processed/forecast_ACC_OWNERSHIP.csv")
        mm_fc = pd.read_csv("data/processed/forecast_ACC_MM_ACCOUNT.csv")

        model_choice = st.selectbox("Select indicator", ["Account Ownership (Access)",
                                                            "Mobile Money Accounts (Usage)"])
        fc_df = acc_fc if "Access" in model_choice else mm_fc

        fig = go.Figure()
        for scenario, color in [("pessimistic", "red"), ("base", "blue"), ("optimistic", "green")]:
            sub = fc_df[fc_df["scenario"] == scenario]
            fig.add_trace(go.Scatter(x=sub["year"], y=sub["forecast"], mode="lines+markers",
                                      name=scenario.title(), line=dict(color=color)))
            fig.add_trace(go.Scatter(
                x=list(sub["year"]) + list(sub["year"])[::-1],
                y=list(sub["upper_95"]) + list(sub["lower_95"])[::-1],
                fill="toself", fillcolor=color, opacity=0.1,
                line=dict(width=0), showlegend=False,
            ))
        fig.update_layout(title=f"{model_choice} Forecast", xaxis_title="Year", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(fc_df, use_container_width=True)
        st.download_button("Download forecast (CSV)", fc_df.to_csv(index=False), "forecast.csv")

    except FileNotFoundError:
        st.warning("Forecast files not found — run `python -m src.forecast` first "
                    "to generate data/processed/forecast_*.csv")

# ---------------- INCLUSION PROJECTIONS ----------------
elif page == "Inclusion Projections":
    st.title("Progress Toward Inclusion Targets")

    targets = df[df["record_type"] == "target"]
    st.subheader("Official Policy Targets")
    st.dataframe(targets[["indicator", "indicator_code", "value_numeric",
                           "observation_date", "notes"]], use_container_width=True)

    scenario_pick = st.select_slider("Scenario", options=["pessimistic", "base", "optimistic"],
                                      value="base")

    try:
        acc_fc = pd.read_csv("data/processed/forecast_ACC_OWNERSHIP.csv")
        sub = acc_fc[acc_fc["scenario"] == scenario_pick]

        target_val = targets[targets["indicator_code"] == "ACC_OWNERSHIP"]["value_numeric"]
        target_line = target_val.iloc[0] if len(target_val) else 60

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["year"], y=sub["forecast"], mode="lines+markers",
                                  name=f"{scenario_pick.title()} forecast"))
        fig.add_hline(y=target_line, line_dash="dot", line_color="red",
                      annotation_text=f"Target: {target_line}%")
        fig.update_layout(title="Progress Toward Account Ownership Target",
                           xaxis_title="Year", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

        gap = target_line - sub["forecast"].iloc[-1]
        st.metric(f"Gap to target by {sub['year'].iloc[-1]}", f"{gap:.1f}pp remaining")

    except FileNotFoundError:
        st.warning("Run `python -m src.forecast` first to generate forecast data.")

    st.subheader("Key Questions Answered")
    st.markdown("""
    - **How did financial inclusion change in 2025?** See Trends page for latest observations.
    - **How will it look in 2026-2027?** See scenario forecast above.
    - **What drives inclusion?** See event timeline & association matrix (Task 3 outputs).
    """)