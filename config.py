import os
import pandas as pd
from datetime import datetime

API_KEY = os.getenv("INBOUND_API_KEY")
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY")

df = pd.read_csv("data/loads.csv")

for col in ["pickup_datetime", "delivery_datetime"]:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass