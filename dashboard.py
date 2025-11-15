import os
from typing import Literal, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st


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
DEFAULT_CSV_PATH = "data/call_logs_150.csv"

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


# ============================================================
#   Sidebar
# ============================================================

st.sidebar.header("Data source")

data_source = st.sidebar.selectbox(
    "Source",
    ["csv", "api"],
    index=1 if DEFAULT_DATA_SOURCE == "api" else 0
)

csv_path = st.sidebar.text_input(
    "CSV path",
    value=DEFAULT_CSV_PATH
)

api_url = st.sidebar.text_input(
    "/call-log API URL",
    value=DEFAULT_CALL_LOG_API_URL
)

api_header_name = st.sidebar.text_input(
    "API key header name",
    value="x-api-key"
)

api_header_prefix = st.sidebar.text_input(
    "API key prefix",
    value=" "
)

st.sidebar.caption(
    f"Remember to set environment variable {API_KEY_ENV_VAR}"
)


# ============================================================
#   Load + preprocess (cached)
# ============================================================

@st.cache_data(show_spinner=True)
def load_and_preprocess(
    source,
    csv_path,
    api_url,
    api_header_name,
    api_header_prefix
):
    df_raw = load_call_log(
        source,
        csv_path,
        api_url,
        api_header_name,
        api_header_prefix
    )
    return preprocess_call_log(df_raw)


try:
    df_calls = load_and_preprocess(
        data_source,
        csv_path,
        api_url,
        api_header_name,
        api_header_prefix
    )
except Exception as exc:
    st.error(f"Error loading data: {exc}")
    st.stop()

if df_calls.empty:
    st.warning("No calls found.")
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

colA, colB = st.columns(2)

with colA:
    st.markdown("**Outcome distribution (%)**")
    st.bar_chart(outcome_summary.set_index("outcome")["percentage"])
    st.dataframe(outcome_summary, use_container_width=True)

with colB:
    st.markdown("**Sentiment distribution (%)**")
    st.bar_chart(sentiment_summary.set_index("sentiment")["percentage"])
    st.dataframe(sentiment_summary, use_container_width=True)


# ----------------
#   Last calls
# ----------------

st.subheader("Latest 10 Calls")
st.dataframe(last_calls, use_container_width=True)


with st.expander("Show full raw data"):
    st.dataframe(df_calls, use_container_width=True)