
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_intensity_imbalance(
    df: pd.DataFrame,
    span: int = 24,
    window: int = 24,
) -> pd.Series:
    """
    Signed taker-flow imbalance weighted by relative average trade size.

    The raw imbalance is:
        (2 * taker_buy_quote_volume - quote_volume) / quote_volume

    It is multiplied by the current average trade size (quote_volume / trades)
    relative to its trailing rolling median over `window` bars.

    The resulting signal is smoothed with a causal EWM of `span` bars.
    All rolling and smoothing operations are performed independently
    for each symbol.

    The input must be ordered chronologically within each symbol.
    The returned Series preserves df.index.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")

    # 1. Extract market data.
    quote_volume = df["quote_volume"].astype(np.float64)
    taker_buy_volume = df["taker_buy_quote_volume"].astype(np.float64)
    trades = df["trades"].astype(np.float64)

    symbols = df["symbol"]

    # 2. Signed taker-flow imbalance.
    # A non-positive quote volume produces zero imbalance.
    imbalance = (
        (2.0 * taker_buy_volume - quote_volume)
        / quote_volume.where(quote_volume > 0.0)
    )

    imbalance = (
        imbalance
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 3. Average trade size.
    # Invalid or zero trade counts produce an undefined average.
    avg_trade_size = (
        quote_volume.where(quote_volume > 0.0)
        / trades.where(trades > 0.0)
    )

    avg_trade_size = avg_trade_size.replace(
        [np.inf, -np.inf], np.nan
    )

    # 4. Trailing rolling median, calculated separately per symbol.
    rolling_median = avg_trade_size.groupby(
        symbols, sort=False
    ).transform(
        lambda x: x.rolling(
            window=window,
            min_periods=1,
        ).median()
    )

    # 5. Relative trade size.
    relative_size = (
        avg_trade_size
        / rolling_median.where(rolling_median > 0.0)
    )

    relative_size = (
        relative_size
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 6. Combine directional flow and relative trade size.
    raw_signal = imbalance * relative_size

    # 7. Causal EWM, independently for each symbol.
    signal = raw_signal.groupby(
        symbols, sort=False
    ).transform(
        lambda x: x.ewm(
            span=span,
            min_periods=1,
        ).mean()
    )

    return signal.rename("trade_intensity_imbalance")
