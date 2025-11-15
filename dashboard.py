import os
from typing import Literal, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px


# ============================================================
#   Page configuration
# ============================================================

st.set_page_config(
    page_title="Inbound Carrier Sales Dashboard",
    layout="wide"
)

st.title("Inbound Carrier Sales Dashboard")
st.caption("HappyRobot – Technical Challenge Dashboard")


# ============================================================
#   Configuration (Railway + local)
# ============================================================

# Default is API mode for Railway
DEFAULT_DATA_SOURCE: Literal["csv", "api"] = "api"

# Local fallback CSV (only for dev)
DEFAULT_CSV_PATH = "data/call_logs_500.csv"

# API endpoint, read from environment in Railway
DEFAULT_CALL_LOG_API_URL = os.getenv(
    "CALL_LOG_API_URL",
    "http://localhost:8000/call-log/historic"
)

# Environment variable with your API key
API_KEY_ENV_VAR = "INBOUND_API_KEY"

EXPECTED_COLUMNS = [
    "call_id",
    "happyrobot_run_id",
    "started_at",
    "ended_at",
    "duration_seconds",
    "mc_number",
    "load_id",
    "origin",
    "destination",
    "pickup_datetime",
    "delivery_datetime",
    "equipment_type",
    "listed_rate",
    "proposed_rate",
    "final_rate",
    "negotiation_rounds",
    "eligible",
    "outcome",
    "sentiment",
    "notes",
    "transcript",
    "logged_at"
]


# ============================================================
#   Data loading utilities
# ============================================================

def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_from_api(
    url: str,
    api_key_env: str = API_KEY_ENV_VAR,
    timeout: int = 10,
    header_name: str = "Authorization",
    header_prefix: str = "Bearer "
) -> pd.DataFrame:

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set."
        )

    if header_name.lower() == "authorization":
        header_value = f"{header_prefix}{api_key}"
    else:
        header_value = api_key

    headers = {
        header_name: header_value,
        "Accept": "application/json"
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Flexible parsing depending on backend shape
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if "data" in data:
            records = data["data"]
        elif "items" in data:
            records = data["items"]
        else:
            raise ValueError(
                "API response dict does not contain 'data' or 'items'."
            )
    else:
        raise ValueError("Unexpected API response type.")

    df = pd.DataFrame(records)
    return df


def load_call_log(
    source: Literal["csv", "api"],
    csv_path: str,
    api_url: str,
    api_key_header_name: str,
    api_key_prefix: str
) -> pd.DataFrame:

    if source == "csv":
        df = load_from_csv(csv_path)
    elif source == "api":
        df = load_from_api(
            api_url,
            api_key_env=API_KEY_ENV_VAR,
            header_name=api_key_header_name,
            header_prefix=api_key_prefix
        )
    else:
        raise ValueError("source must be 'csv' or 'api'.")

    available = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[available]


# ============================================================
#   Preprocessing
# ============================================================

def to_datetime_safe(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def to_numeric_safe(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_outcome(df: pd.DataFrame):
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
    return df


def standardize_sentiment(df: pd.DataFrame):
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    return df


def compute_discount(df: pd.DataFrame):
    if "listed_rate" not in df.columns or "final_rate" not in df.columns:
        df["discount_pct"] = np.nan
        return df

    df = to_numeric_safe(df, "listed_rate")
    df = to_numeric_safe(df, "final_rate")

    mask = (df["listed_rate"] > 0) & (df["final_rate"] > 0)
    df["discount_pct"] = np.nan
    df.loc[mask, "discount_pct"] = (
        (df.loc[mask, "listed_rate"] - df.loc[mask, "final_rate"])
        / df.loc[mask, "listed_rate"]
    )

    return df


def preprocess_call_log(df: pd.DataFrame) -> pd.DataFrame:
    for dt in ["started_at", "ended_at", "pickup_datetime",
               "delivery_datetime", "logged_at"]:
        df = to_datetime_safe(df, dt)

    for num in ["duration_seconds", "listed_rate", "proposed_rate",
                "final_rate", "negotiation_rounds"]:
        df = to_numeric_safe(df, num)

    if "eligible" in df.columns:
        df["eligible"] = df["eligible"].astype("boolean")

    df = standardize_outcome(df)
    df = standardize_sentiment(df)
    df = compute_discount(df)

    return df


# ============================================================
#   Metrics helpers
# ============================================================

def compute_outcome_percentages(
    df: pd.DataFrame,
    focus: Optional[list] = None
) -> pd.DataFrame:

    total = len(df)
    counts = df["outcome"].value_counts(dropna=False)

    result = (
        counts.rename("count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "outcome"})
    )

    result["percentage"] = (result["count"] / total) * 100

    if focus:
        for outcome in focus:
            if outcome not in result["outcome"].values:
                result = pd.concat([
                    result,
                    pd.DataFrame([{
                        "outcome": outcome,
                        "count": 0,
                        "percentage": 0.0
                    }])
                ], ignore_index=True)

    return result.sort_values("outcome").reset_index(drop=True)


def compute_average_negotiation_rounds(df: pd.DataFrame):
    if "negotiation_rounds" not in df.columns:
        return float("nan")
    return float(df["negotiation_rounds"].dropna().mean())


def compute_average_discount(df: pd.DataFrame):
    if "discount_pct" not in df.columns:
        return float("nan")
    return float(df["discount_pct"].dropna().mean())


def compute_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df["sentiment"].value_counts(dropna=False)

    result = (
        counts.rename("count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "sentiment"})
    )

    result["percentage"] = (result["count"] / total) * 100
    return result.sort_values("sentiment").reset_index(drop=True)


def get_last_calls_table(df: pd.DataFrame, n: int = 10):
    df2 = df.copy()

    if "logged_at" in df2.columns and df2["logged_at"].notna().any():
        df2 = df2.sort_values("logged_at", ascending=False)
    elif "started_at" in df2.columns and df2["started_at"].notna().any():
        df2 = df2.sort_values("started_at", ascending=False)
    else:
        df2 = df2.sort_index(ascending=False)

    cols = [
        "call_id", "logged_at", "started_at",
        "mc_number", "load_id",
        "origin", "destination", "equipment_type",
        "listed_rate", "final_rate",
        "negotiation_rounds", "eligible",
        "outcome", "sentiment"
    ]

    cols = [c for c in cols if c in df2.columns]

    return df2[cols].head(n)


def get_outcome_pct(summary: pd.DataFrame, name: str) -> float:
    row = summary.loc[summary["outcome"] == name]
    if row.empty:
        return 0.0
    return float(row["percentage"].iloc[0])


def compute_calls_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "started_at" not in df.columns:
        return pd.DataFrame(columns=["hour", "count"])

    df2 = df[df["started_at"].notna()].copy()
    if df2.empty:
        return pd.DataFrame(columns=["hour", "count"])

    df2["hour"] = df2["started_at"].dt.hour
    counts = (
        df2.groupby("hour")
        .size()
        .reset_index(name="count")
        .sort_values("hour")
    )
    return counts


# ============================================================
#   Load + preprocess (cached)
# ============================================================

@st.cache_data(show_spinner=True)
def load_and_preprocess():
    df_raw = load_call_log(
        "csv",
        DEFAULT_CSV_PATH,
        DEFAULT_CALL_LOG_API_URL,
        "x-api-key",
        " "
    )
    return preprocess_call_log(df_raw)


try:
    df_calls = load_and_preprocess()
except Exception as exc:
    st.error(f"Error loading data: {exc}")
    st.stop()

if df_calls.empty:
    st.warning("No calls found.")
    st.stop()

# ----------------
#   Filters
# ----------------

st.sidebar.subheader("Filters")

# Date range filter based on started_at if available, otherwise logged_at
date_col = None
if "started_at" in df_calls.columns and df_calls["started_at"].notna().any():
    date_col = "started_at"
elif "logged_at" in df_calls.columns and df_calls["logged_at"].notna().any():
    date_col = "logged_at"

if date_col is not None:
    min_date = df_calls[date_col].min().date()
    max_date = df_calls[date_col].max().date()
    date_range = st.sidebar.date_input(
        "Call date range",
        value=(min_date, max_date)
    )
    # Ensure we always have a tuple (start, end)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
else:
    start_date = None
    end_date = None

# Outcome filter
if "outcome" in df_calls.columns:
    outcome_options = sorted(df_calls["outcome"].dropna().unique())
    selected_outcomes = st.sidebar.multiselect(
        "Outcome",
        options=outcome_options,
        default=[],
        format_func=lambda v: v.replace("_", " ")
    )
else:
    selected_outcomes = None

# Sentiment filter
if "sentiment" in df_calls.columns:
    sentiment_options = sorted(df_calls["sentiment"].dropna().unique())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=sentiment_options,
        default=[],
        format_func=lambda v: v.replace("_", " ")
    )
else:
    selected_sentiments = None

# Apply filters
mask = pd.Series(True, index=df_calls.index)

if start_date is not None and end_date is not None and date_col is not None:
    mask &= df_calls[date_col].dt.date.between(start_date, end_date)

if selected_outcomes is not None and len(selected_outcomes) > 0:
    mask &= df_calls["outcome"].isin(selected_outcomes)

if selected_sentiments is not None and len(selected_sentiments) > 0:
    mask &= df_calls["sentiment"].isin(selected_sentiments)

df_calls = df_calls.loc[mask].copy()

if df_calls.empty:
    st.warning("No calls match the selected filters.")
    st.stop()


# ============================================================
#   Compute metrics
# ============================================================

total_calls = len(df_calls)

outcome_summary = compute_outcome_percentages(
    df_calls,
    focus=[
        "booked_load",
        "no_match",
        "not_eligible",
        "no_load_price",
        "not_interested"
    ]
)

pct_booked_load = get_outcome_pct(outcome_summary, "booked_load")
pct_no_match = get_outcome_pct(outcome_summary, "no_match")
pct_not_eligible = get_outcome_pct(outcome_summary, "not_eligible")
pct_no_load_price = get_outcome_pct(outcome_summary, "no_load_price")
pct_not_interested = get_outcome_pct(outcome_summary, "not_interested")

avg_rounds = compute_average_negotiation_rounds(df_calls)
avg_discount = compute_average_discount(df_calls)

sentiment_summary = compute_sentiment_distribution(df_calls)
last_calls = get_last_calls_table(df_calls, n=10)

calls_by_hour = compute_calls_by_hour(df_calls)

avg_duration_sec = float(
    df_calls["duration_seconds"].dropna().mean()
) if "duration_seconds" in df_calls.columns else float("nan")

busiest_hour = None
busiest_hour_calls = None
if not calls_by_hour.empty:
    idx_max = calls_by_hour["count"].idxmax()
    busiest_hour = int(calls_by_hour.loc[idx_max, "hour"])
    busiest_hour_calls = int(calls_by_hour.loc[idx_max, "count"])


# Load-level metrics
booked_mask = df_calls["outcome"] == "booked_load" if "outcome" in df_calls.columns else pd.Series(False, index=df_calls.index)

has_listed = df_calls["listed_rate"].notna() if "listed_rate" in df_calls.columns else pd.Series(False, index=df_calls.index)
has_final = df_calls["final_rate"].notna() if "final_rate" in df_calls.columns else pd.Series(False, index=df_calls.index)
valid_rate_mask = has_listed & has_final

avg_listed_rate = float(
    df_calls.loc[has_listed, "listed_rate"].mean()
) if "listed_rate" in df_calls.columns and has_listed.any() else float("nan")

avg_final_rate_booked = float(
    df_calls.loc[booked_mask & has_final, "final_rate"].mean()
) if "final_rate" in df_calls.columns and (booked_mask & has_final).any() else float("nan")

if (booked_mask & valid_rate_mask).any():
    discount_amount_series = df_calls.loc[booked_mask & valid_rate_mask, "listed_rate"] - df_calls.loc[booked_mask & valid_rate_mask, "final_rate"]
    total_discount_amount = float(discount_amount_series.sum())
else:
    total_discount_amount = 0.0

if "negotiation_rounds" in df_calls.columns:
    round3_mask = df_calls["negotiation_rounds"] >= 3
    calls_round3 = int(df_calls.loc[round3_mask].shape[0])
    booked_round3 = int(df_calls.loc[booked_mask & round3_mask].shape[0])
else:
    round3_mask = pd.Series(False, index=df_calls.index)
    calls_round3 = 0
    booked_round3 = 0

if "listed_rate" in df_calls.columns and "proposed_rate" in df_calls.columns:
    mask_round3_diff = round3_mask & df_calls["listed_rate"].notna() & df_calls["proposed_rate"].notna()
    if mask_round3_diff.any():
        diff_round3 = df_calls.loc[mask_round3_diff, "proposed_rate"] - df_calls.loc[mask_round3_diff, "listed_rate"]
        avg_diff_round3 = float(diff_round3.mean())
    else:
        avg_diff_round3 = float("nan")
else:
    avg_diff_round3 = float("nan")

if "equipment_type" in df_calls.columns:
    equipment_booked = (
        df_calls.loc[booked_mask & df_calls["equipment_type"].notna(), "equipment_type"]
        .value_counts()
        .rename_axis("equipment_type")
        .reset_index(name="count")
    )
else:
    equipment_booked = pd.DataFrame(columns=["equipment_type", "count"])

if "origin" in df_calls.columns:
    top_origins_booked = (
        df_calls.loc[booked_mask & df_calls["origin"].notna(), "origin"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "origin"})
        .head(5)
    )
else:
    top_origins_booked = pd.DataFrame(columns=["origin", "count"])

if "destination" in df_calls.columns:
    top_destinations_booked = (
        df_calls.loc[booked_mask & df_calls["destination"].notna(), "destination"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "destination"})
        .head(5)
    )
else:
    top_destinations_booked = pd.DataFrame(columns=["destination", "count"])

# Funnel / efficiency metrics
if "eligible" in df_calls.columns:
    eligible_mask = df_calls["eligible"] == True
    if eligible_mask.any():
        booked_from_eligible = df_calls.loc[eligible_mask & booked_mask].shape[0]
        total_eligible = df_calls.loc[eligible_mask].shape[0]
        pct_booked_from_eligible = 100.0 * booked_from_eligible / total_eligible
    else:
        pct_booked_from_eligible = float("nan")
else:
    eligible_mask = pd.Series(False, index=df_calls.index)
    pct_booked_from_eligible = float("nan")

# Negotiation performance by bucket of rounds
if "negotiation_rounds" in df_calls.columns:
    df_calls["negotiation_bucket"] = pd.cut(
        df_calls["negotiation_rounds"].fillna(0),
        bins=[-0.1, 0.5, 1.5, 2.5, 100],
        labels=["0", "1", "2", "3+"]
    )
    negotiation_perf = (
        df_calls.groupby("negotiation_bucket")
        .agg(
            calls=("call_id", "count"),
            booked=("outcome", lambda x: (x == "booked_load").sum())
        )
        .reset_index()
    )
    negotiation_perf["booked_rate_pct"] = (
        negotiation_perf["booked"] / negotiation_perf["calls"] * 100.0
    )
else:
    negotiation_perf = pd.DataFrame(columns=["negotiation_bucket", "calls", "booked", "booked_rate_pct"])

# Discount by sentiment on booked loads
if "discount_pct" in df_calls.columns and booked_mask.any():
    discount_by_sentiment = (
        df_calls.loc[booked_mask]
        .groupby("sentiment", dropna=False)["discount_pct"]
        .mean()
        .reset_index()
    )
    discount_by_sentiment["discount_pct"] *= 100.0
    discount_by_sentiment["sentiment"] = discount_by_sentiment["sentiment"].astype(str).str.replace("_", " ")
else:
    discount_by_sentiment = pd.DataFrame(columns=["sentiment", "discount_pct"])

# Carrier performance (top 5 by volume)
if "mc_number" in df_calls.columns:
    carrier_perf = (
        df_calls.groupby("mc_number")
        .agg(
            calls=("call_id", "count"),
            booked=("outcome", lambda x: (x == "booked_load").sum())
        )
        .reset_index()
    )
    carrier_perf["booked_rate_pct"] = (
        carrier_perf["booked"] / carrier_perf["calls"] * 100.0
    )
    top_carriers = carrier_perf.sort_values("calls", ascending=False).head(5)
else:
    top_carriers = pd.DataFrame(columns=["mc_number", "calls", "booked", "booked_rate_pct"])


# ============================================================
#   Layout
# ============================================================

st.subheader("High Level Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total calls", total_calls)
with col2:
    st.metric("Booked loads (%)", f"{pct_booked_load:0.2f}%")
with col3:
    st.metric("Avg negotiation rounds", f"{avg_rounds:0.2f}")

col4, col5 = st.columns(2)
with col4:
    st.metric("Avg discount (%)", f"{avg_discount * 100:0.2f}%")
with col5:
    st.metric("Not eligible (%)", f"{pct_not_eligible:0.2f}%")


# ----------------
#   Charts
# ----------------

st.subheader("Outcome Breakdown")

# Add outcome and sentiment labels and color map
outcome_summary["outcome_label"] = outcome_summary["outcome"].str.replace("_", " ")
sentiment_summary["sentiment_label"] = sentiment_summary["sentiment"].str.replace("_", " ")
sentiment_color_map = {
    "very_negative": "#d73027",
    "negative": "#fc8d59",
    "neutral": "#ffffbf",
    "positive": "#91bfdb",
    "very_positive": "#4575b4",
}

colA, colB = st.columns(2)

with colA:
    st.markdown("**Outcome distribution (%)**")
    fig_out = px.pie(
        outcome_summary,
        values="percentage",
        names="outcome",
    )
    fig_out.update_traces(
        text=outcome_summary["outcome_label"],
        textposition="inside",
        texttemplate="%{text}<br>%{value:.1f}%"
    )
    fig_out.update_layout(showlegend=False)
    st.plotly_chart(fig_out, use_container_width=True)

with colB:
    st.markdown("**Sentiment distribution (%)**")
    fig_sent = px.pie(
        sentiment_summary,
        values="percentage",
        names="sentiment",
        color="sentiment",
        color_discrete_map=sentiment_color_map,
    )
    fig_sent.update_traces(
        text=sentiment_summary["sentiment_label"],
        textposition="inside",
        texttemplate="%{text}<br>%{value:.1f}%"
    )
    fig_sent.update_layout(showlegend=False)
    st.plotly_chart(fig_sent, use_container_width=True)


st.subheader("Call Volume by Hour")

colH1, colH2 = st.columns([3, 1])

with colH1:
    if not calls_by_hour.empty:
        st.markdown("**Number of calls per hour of day**")
        st.bar_chart(calls_by_hour.set_index("hour")["count"])
    else:
        st.info("No call timestamps available to compute call volume by hour.")

with colH2:
    if busiest_hour is not None:
        st.metric(
            "Busiest hour (24h)",
            f"{busiest_hour:02d}:00",
            help="Hour of day with the highest number of calls."
        )
        st.metric(
            "Calls in busiest hour",
            busiest_hour_calls
        )
    if not np.isnan(avg_duration_sec):
        st.metric(
            "Avg call duration (sec)",
            f"{avg_duration_sec:0.1f}"
        )


# ----------------
#   Load metrics
# ----------------

st.subheader("Load Metrics")

colL1, colL2, colL3 = st.columns(3)
with colL1:
    if not np.isnan(avg_final_rate_booked):
        st.metric("Avg booked final rate (USD)", f"${avg_final_rate_booked:,.0f}")
    elif not np.isnan(avg_listed_rate):
        st.metric("Avg listed rate (USD)", f"${avg_listed_rate:,.0f}")

with colL2:
    st.metric("Total discount vs listed (USD)", f"${total_discount_amount:,.0f}")

colL4, colL5 = st.columns(2)
with colL4:
    st.metric("Calls with ≥ 3 negotiation rounds", calls_round3)
    if booked_mask.any() and booked_round3 > 0:
        pct_booked_round3 = (booked_round3 / df_calls.loc[booked_mask].shape[0]) * 100.0
        st.caption(f"{pct_booked_round3:0.1f}% of booked loads reached 3+ rounds.")

with colL5:
    if not np.isnan(avg_diff_round3):
        st.metric(
            "Avg (proposed - listed) on 3+ rounds (USD)",
            f"${avg_diff_round3:,.0f}"
        )
    else:
        st.metric("Avg (proposed - listed) on 3+ rounds", "n/a")

st.markdown("**Booked loads by equipment type**")
if equipment_booked is not None and not equipment_booked.empty:
    fig_eq = px.bar(
        equipment_booked,
        x="equipment_type",
        y="count",
    )
    fig_eq.update_layout(
        xaxis_title="Equipment type",
        yaxis_title="Booked loads"
    )
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.info("No booked loads with equipment type information.")

st.subheader("Top 5 Origins and Destinations (Booked Loads)")
colO, colD = st.columns(2)

with colO:
    st.markdown("**Top 5 origins (booked loads)**")
    if top_origins_booked is not None and not top_origins_booked.empty:
        fig_orig = px.bar(
            top_origins_booked,
            x="origin",
            y="count",
        )
        fig_orig.update_layout(
            xaxis_title="Origin",
            yaxis_title="Booked loads"
        )
        st.plotly_chart(fig_orig, use_container_width=True)
    else:
        st.info("No origin data available for booked loads.")

with colD:
    st.markdown("**Top 5 destinations (booked loads)**")
    if top_destinations_booked is not None and not top_destinations_booked.empty:
        fig_dest = px.bar(
            top_destinations_booked,
            x="destination",
            y="count",
        )
        fig_dest.update_layout(
            xaxis_title="Destination",
            yaxis_title="Booked loads"
        )
        st.plotly_chart(fig_dest, use_container_width=True)
    else:
        st.info("No destination data available for booked loads.")

# Negotiation performance visualization
st.subheader("Negotiation Performance")
if negotiation_perf is not None and not negotiation_perf.empty:
    fig_neg = px.bar(
        negotiation_perf,
        x="negotiation_bucket",
        y="booked_rate_pct",
    )
    fig_neg.update_layout(
        xaxis_title="Negotiation rounds (bucket)",
        yaxis_title="Booked rate (%)"
    )
    st.plotly_chart(fig_neg, use_container_width=True)
    st.dataframe(negotiation_perf, use_container_width=True)
else:
    st.info("No negotiation data available.")

# Discount vs sentiment
st.subheader("Discount vs Sentiment (Booked Loads)")
if discount_by_sentiment is not None and not discount_by_sentiment.empty:
    fig_disc = px.bar(
        discount_by_sentiment,
        x="sentiment",
        y="discount_pct",
    )
    fig_disc.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Avg discount (%)"
    )
    st.plotly_chart(fig_disc, use_container_width=True)
    st.dataframe(discount_by_sentiment, use_container_width=True)
else:
    st.info("No discount data available for booked loads.")


# Top carriers
st.subheader("Top 5 Carriers by Call Volume")
if top_carriers is not None and not top_carriers.empty:
    st.dataframe(top_carriers, use_container_width=True)
else:
    st.info("No carrier performance data available.")


# ----------------
#   Last calls
# ----------------

st.subheader("Latest 10 Calls")
st.dataframe(last_calls, use_container_width=True)


with st.expander("Show full raw data"):
    st.dataframe(df_calls, use_container_width=True)