"""
NSE option-chain data retrieval and preprocessing utilities.

This module provides:
- NSE option-chain API access
- Retry/session handling
- Option-chain preprocessing
- Mid-price calculation
- Expiry selection
- Market dataset generation for calibration

Designed for use with Bates model calibration workflows.
"""
import time
from datetime import datetime as dt
import numpy as np
import pandas as pd
import requests

def fetch_nse_option_chain(
    symbol: str = "NIFTY",
    max_retries: int = 3,
    timeout: int = 15
):
    url = (
        f"https://www.nseindia.com/api/"
        f"option-chain-indices?symbol={symbol}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive"
    }

    session = requests.Session()

    for attempt in range(max_retries):

        try:
            # Initial request to obtain NSE cookies
            session.get(
                "https://www.nseindia.com",
                headers=headers,
                timeout=timeout
            )

            time.sleep(2)

            # Fetch option-chain JSON data
            response = session.get(
                url,
                headers=headers,
                timeout=timeout
            )

            response.raise_for_status()

            if response.text.strip():
                return response.json()

            print(
                f"Empty response received "
                f"(Attempt {attempt + 1}/{max_retries})"
            )

        except Exception as error:

            print(
                f"Attempt {attempt + 1}/{max_retries} failed: "
                f"{error}"
            )

            time.sleep(5 * (attempt + 1))

    raise Exception(
        f"Failed to fetch NSE option-chain data "
        f"for {symbol}"
    )


def select_nearest_expiry(expiry_dates):
    """
    Select nearest future expiry date.

    Parameters
    ----------
    expiry_dates : list
        List of expiry dates from NSE API.

    Returns
    -------
    str
        Nearest available expiry date.
    """

    today = dt.today()

    future_expiries = [
        expiry
        for expiry in expiry_dates
        if dt.strptime(expiry, "%d-%b-%Y") > today
    ]

    if future_expiries:
        return future_expiries[0]

    # Fallback if no future expiry exists
    return expiry_dates[-1]


def process_option_chain_data(
    data,
    risk_free_rate: float = 0.0654
):
    """
    Convert NSE option-chain JSON into a clean DataFrame.

    Processing steps:
    - Select nearest expiry
    - Extract call/put option data
    - Compute bid-ask mid prices
    - Compute maturity in years

    Parameters
    ----------
    data : dict
        NSE option-chain JSON response.

    risk_free_rate : float
        Assumed annualized risk-free interest rate.

    Returns
    -------
    df : pandas.DataFrame
        Clean option-chain dataset.

    spot_price : float
        Current underlying index value.

    expiry : str
        Selected expiry date.
    """

    if data is None:
        raise ValueError("No NSE data received.")

    records = data["records"]["data"]

    spot_price = data["records"]["underlyingValue"]

    expiry_dates = data["records"]["expiryDates"]

    target_expiry = select_nearest_expiry(
        expiry_dates
    )

    options = []

    for record in records:

        if record["expiryDate"] != target_expiry:
            continue

        strike = record["strikePrice"]

        call_data = record.get("CE", {})
        put_data = record.get("PE", {})

        if not (call_data and put_data):
            continue

        call_bid = call_data.get("bidprice")
        call_ask = call_data.get("askPrice")

        put_bid = put_data.get("bidprice")
        put_ask = put_data.get("askPrice")

        # Ensure valid bid-ask quotes exist
        valid_quotes = (
            call_bid
            and call_ask
            and put_bid
            and put_ask
            and call_ask > 0
            and put_ask > 0
        )

        if not valid_quotes:
            continue

        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2

        options.append({
            "expiry_date": target_expiry,
            "strike": strike,
            "call_price": call_mid,
            "put_price": put_mid,
            "call_iv": call_data.get(
                "impliedVolatility"
            ),
            "put_iv": put_data.get(
                "impliedVolatility"
            )
        })

    df = pd.DataFrame(options)

    if df.empty:
        raise ValueError(
            f"No valid option data found "
            f"for expiry {target_expiry}"
        )

    # Compute time-to-maturity in years
    df["maturity"] = df["expiry_date"].apply(
        lambda expiry: (
            dt.strptime(expiry, "%d-%b-%Y")
            - dt.today()
        ).days / 365.25
    )

    df["risk_free_rate"] = risk_free_rate

    return df, spot_price, target_expiry


if __name__ == "__main__":

    try:

        # Fetch market option-chain data
        raw_data = fetch_nse_option_chain(
            symbol="NIFTY"
        )

        # Process into clean calibration dataset
        option_data, spot_price, expiry = (
            process_option_chain_data(raw_data)
        )

        print("\nNSE Option Chain Summary")
        print("-" * 40)

        print(f"Underlying Spot Price : {spot_price}")
        print(f"Selected Expiry       : {expiry}")
        print(f"Options Retrieved     : {len(option_data)}")

        print("\nDataset Preview:\n")

        print(
            option_data.head(10).to_string(
                index=False
            )
        )

        print("\nDescriptive Statistics:\n")

        print(
            option_data[
                ["strike", "call_price", "put_price"]
            ].describe()
        )

    except Exception as error:

        print(f"\nError: {error}")
