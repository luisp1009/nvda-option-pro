import math
import os
import time as time_module
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

APP_VERSION = "tradier-options-cache-v1"

DEFAULT_SYMBOLS = ["SPY", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "AMZN", "META", "QQQ"]

# Wider range so contracts are easier to find
MIN_DTE = 1
MAX_DTE = 60

# Limit option-chain API calls per scan to avoid rate limits
MAX_EXPIRATIONS_TO_SCAN = 3

RISK_FREE_RATE = 0.045

ATR_LEN = 14
RSI_LEN = 14
EMA_FAST = 9
EMA_SLOW = 21

ENTRY_BUFFER_ATR = 0.12
STOP_ATR_MULT = 1.35
TARGET_R_MULT = 1.8

OPTION_STOP_PCT = 0.35
OPTION_TARGET_PCT = 0.60
TOP_OPTION_PICKS = 8

EASTERN = ZoneInfo("America/New_York")

# Browser can refresh every 15 seconds, but Render only calls data providers every 5 minutes per symbol.
SCAN_CACHE = {}
QUOTE_CACHE = {}
OPTIONS_CACHE = {}
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "60"))
QUOTE_CACHE_SECONDS = int(os.getenv("QUOTE_CACHE_SECONDS", "60"))
OPTIONS_CACHE_SECONDS = int(os.getenv("OPTIONS_CACHE_SECONDS", "300"))

# Tradier token must be set in Render Environment Variables.
# Name: TRADIER_TOKEN
TRADIER_TOKEN = os.getenv("TRADIER_TOKEN", "").strip()
TRADIER_BASE_URL = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "SPY").upper().strip()
    symbol = symbol.replace("$", "").replace(" ", "")
    symbol = symbol.replace(".", "-")
    return symbol or "SPY"


def cache_get(cache: dict, key: str, max_age: int):
    item = cache.get(key)
    if not item:
        return None

    age = (datetime.now(EASTERN) - item["time"]).total_seconds()
    if age <= max_age:
        return item["data"]

    return None


def cache_set(cache: dict, key: str, data):
    cache[key] = {
        "time": datetime.now(EASTERN),
        "data": data,
    }


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    tmp = df.copy()
    idx = pd.to_datetime(tmp.index)

    try:
        session_dates = idx.tz_localize(None).date
    except TypeError:
        session_dates = idx.date

    tmp["session"] = session_dates
    typical = (tmp["High"] + tmp["Low"] + tmp["Close"]) / 3
    pv = typical * tmp["Volume"]
    cum_pv = pv.groupby(tmp["session"]).cumsum()
    cum_vol = tmp["Volume"].groupby(tmp["session"]).cumsum().replace(0, np.nan)

    return cum_pv / cum_vol


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(spot, strike, t, r, sigma, option_type):
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        return float("nan")

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (
        sigma * math.sqrt(t)
    )

    if option_type.lower() == "call":
        return norm_cdf(d1)

    return norm_cdf(d1) - 1.0


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def market_session_info(now_et: datetime):
    current_time = now_et.time()

    if now_et.weekday() >= 5:
        return {"status": "CLOSED", "extended_hours": False, "session_label": "Weekend"}

    if time(4, 0) <= current_time < time(9, 30):
        return {"status": "PREMARKET", "extended_hours": True, "session_label": "Premarket"}

    if time(9, 30) <= current_time < time(16, 0):
        return {"status": "OPEN", "extended_hours": False, "session_label": "Regular Hours"}

    if time(16, 0) <= current_time < time(20, 0):
        return {"status": "AFTER HOURS", "extended_hours": True, "session_label": "After Hours"}

    return {"status": "CLOSED", "extended_hours": False, "session_label": "Closed"}


def get_news(symbol: str):
    cache_key = f"news:{symbol}"
    cached = cache_get(QUOTE_CACHE, cache_key, CACHE_SECONDS)
    if cached is not None:
        return cached

    ticker = yf.Ticker(symbol)

    try:
        raw_news = ticker.news
    except Exception as e:
        print(f"RAW NEWS ERROR for {symbol}:", e)
        return []

    items = []

    bullish_words = [
        "surge", "beats", "gain", "upgrade", "record", "strong",
        "rally", "growth", "bullish", "raises", "outperform"
    ]

    bearish_words = [
        "drop", "falls", "miss", "downgrade", "probe", "risk",
        "lawsuit", "decline", "bearish", "cuts", "weak"
    ]

    for item in raw_news[:10]:
        try:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = item.get("title") or content.get("title") or "Headline unavailable"

            link = (
                item.get("link")
                or item.get("canonicalUrl", {}).get("url")
                or content.get("canonicalUrl", {}).get("url")
                or "#"
            )

            publisher = (
                item.get("publisher")
                or content.get("provider", {}).get("displayName")
                or "Source"
            )

            timestamp = (
                item.get("providerPublishTime")
                or item.get("pubDate")
                or content.get("pubDate")
                or content.get("displayTime")
            )

            published = "Recent"

            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp, tz=EASTERN)
                published = dt.strftime("%b %d %I:%M %p")
            elif isinstance(timestamp, str) and timestamp.strip():
                published = timestamp

            lowered = title.lower()
            tone = "neutral"

            if any(word in lowered for word in bullish_words):
                tone = "bullish"
            elif any(word in lowered for word in bearish_words):
                tone = "bearish"

            items.append(
                {
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "published": published,
                    "tone": tone,
                }
            )

        except Exception as e:
            print(f"NEWS ITEM ERROR for {symbol}:", e)

    cache_set(QUOTE_CACHE, cache_key, items)
    return items


def get_price_data(symbol: str):
    cache_key = f"price_data:{symbol}"
    cached = cache_get(QUOTE_CACHE, cache_key, CACHE_SECONDS)
    if cached is not None:
        return cached

    fast = yf.download(
        symbol,
        period="5d",
        interval="1m",
        auto_adjust=False,
        progress=False,
        prepost=True,
        threads=False,
    )

    if fast.empty:
        fast = yf.download(
            symbol,
            period="10d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            prepost=True,
            threads=False,
        )

    slow = yf.download(
        symbol,
        period="60d",
        interval="30m",
        auto_adjust=False,
        progress=False,
        prepost=True,
        threads=False,
    )

    fast = flatten_columns(fast).dropna().copy()
    slow = flatten_columns(slow).dropna().copy()

    if fast.empty or slow.empty:
        raise ValueError(f"Could not load market data for {symbol}. Try another ticker.")

    for frame in (fast, slow):
        frame["ema_fast"] = ema(frame["Close"], EMA_FAST)
        frame["ema_slow"] = ema(frame["Close"], EMA_SLOW)
        frame["rsi"] = rsi(frame["Close"], RSI_LEN)
        frame["atr"] = atr(frame, ATR_LEN)

    fast["vwap"] = intraday_vwap(fast)

    cache_set(QUOTE_CACHE, cache_key, (fast, slow))
    return fast, slow


def score_news(news, preferred_direction):
    if not news:
        return 10, "News neutral/unavailable"

    bullish = sum(1 for n in news if n.get("tone") == "bullish")
    bearish = sum(1 for n in news if n.get("tone") == "bearish")

    if preferred_direction == "CALL":
        if bullish > bearish:
            return 20, "News supports upside"
        if bearish > bullish:
            return 5, "News conflicts upside"
        return 10, "News neutral"

    if preferred_direction == "PUT":
        if bearish > bullish:
            return 20, "News supports downside"
        if bullish > bearish:
            return 5, "News conflicts downside"
        return 10, "News neutral"

    return 10, "News neutral"


def build_signal(symbol: str, news=None):
    fast, slow = get_price_data(symbol)

    if len(fast) < 30:
        raise ValueError(f"Not enough candle data to build a signal for {symbol}.")

    f = fast.iloc[-1]
    s = slow.iloc[-1]
    prev_fast = fast.iloc[-2]

    price = float(f["Close"])
    current_atr = float(f["atr"])

    if math.isnan(current_atr) or current_atr <= 0:
        current_atr = max(price * 0.005, 0.01)

    recent_high = float(fast["High"].tail(20).max())
    recent_low = float(fast["Low"].tail(20).min())

    avg_volume = float(fast["Volume"].tail(30).mean())
    current_volume = float(f["Volume"])
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    bull_points = 0
    bear_points = 0
    reasons = []
    components = {}

    trend_score = 0

    if f["Close"] > f["ema_fast"] > f["ema_slow"]:
        bull_points += 12.5
        trend_score += 12.5
        reasons.append("1m bullish EMA")

    if s["Close"] > s["ema_fast"] > s["ema_slow"]:
        bull_points += 12.5
        trend_score += 12.5
        reasons.append("30m bullish EMA")

    if f["Close"] < f["ema_fast"] < f["ema_slow"]:
        bear_points += 12.5
        trend_score += 12.5
        reasons.append("1m bearish EMA")

    if s["Close"] < s["ema_fast"] < s["ema_slow"]:
        bear_points += 12.5
        trend_score += 12.5
        reasons.append("30m bearish EMA")

    components["trend"] = round(min(trend_score, 25), 1)

    vwap_score = 0

    if pd.notna(f["vwap"]) and f["Close"] > f["vwap"]:
        bull_points += 20
        vwap_score = 20
        reasons.append("Above VWAP")
    elif pd.notna(f["vwap"]) and f["Close"] < f["vwap"]:
        bear_points += 20
        vwap_score = 20
        reasons.append("Below VWAP")
    else:
        vwap_score = 10
        reasons.append("Near VWAP")

    components["vwap"] = vwap_score

    if volume_ratio >= 1.5:
        volume_score = 15
        reasons.append("Strong volume")
    elif volume_ratio >= 1.1:
        volume_score = 10
        reasons.append("Volume above avg")
    else:
        volume_score = 5
        reasons.append("Volume normal/weak")

    components["volume"] = volume_score

    rsi_value = float(f["rsi"])
    rsi_score = 0

    if 55 <= rsi_value <= 70:
        bull_points += 20
        rsi_score = 20
        reasons.append("RSI bullish")
    elif 30 <= rsi_value <= 45:
        bear_points += 20
        rsi_score = 20
        reasons.append("RSI bearish")
    elif 45 < rsi_value < 55:
        rsi_score = 10
        reasons.append("RSI neutral")
    elif rsi_value > 70:
        bull_points += 6
        rsi_score = 6
        reasons.append("RSI overbought")
    elif rsi_value < 30:
        bear_points += 6
        rsi_score = 6
        reasons.append("RSI oversold")

    components["momentum"] = rsi_score

    structure_score = 0

    near_breakout = f["Close"] > recent_high - current_atr * 0.30
    near_breakdown = f["Close"] < recent_low + current_atr * 0.30

    if near_breakout and prev_fast["Close"] <= recent_high:
        bull_points += 20
        structure_score = 20
        reasons.append("Near breakout")
    elif near_breakdown and prev_fast["Close"] >= recent_low:
        bear_points += 20
        structure_score = 20
        reasons.append("Near breakdown")
    else:
        structure_score = 8
        reasons.append("No clean breakout")

    components["structure"] = structure_score

    preferred_direction = "CALL" if bull_points > bear_points else "PUT" if bear_points > bull_points else "NO TRADE"

    news_score, news_reason = score_news(news or [], preferred_direction)
    components["news"] = news_score
    reasons.append(news_reason)

    raw_score = (
        components["trend"]
        + components["vwap"]
        + components["volume"]
        + components["momentum"]
        + components["structure"]
        + components["news"]
    )

    final_score = round(min((raw_score / 120) * 100, 100), 0)
    direction_gap = abs(bull_points - bear_points)

    if preferred_direction == "CALL" and direction_gap >= 12 and final_score >= 55:
        trigger = round(recent_high + current_atr * ENTRY_BUFFER_ATR, 2)
        stop = round(trigger - current_atr * STOP_ATR_MULT, 2)
        risk = trigger - stop
        target = round(trigger + risk * TARGET_R_MULT, 2)

        confidence = "HIGH" if final_score >= 80 else "MEDIUM"

        return {
            "direction": "CALL",
            "score": int(final_score),
            "stock_price": round(price, 2),
            "trigger_price": trigger,
            "stop_price": stop,
            "target_price": target,
            "atr": round(current_atr, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "reasons": reasons,
            "confidence": confidence,
            "score_components": components,
        }, fast

    if preferred_direction == "PUT" and direction_gap >= 12 and final_score >= 55:
        trigger = round(recent_low - current_atr * ENTRY_BUFFER_ATR, 2)
        stop = round(trigger + current_atr * STOP_ATR_MULT, 2)
        risk = stop - trigger
        target = round(trigger - risk * TARGET_R_MULT, 2)

        confidence = "HIGH" if final_score >= 80 else "MEDIUM"

        return {
            "direction": "PUT",
            "score": int(final_score),
            "stock_price": round(price, 2),
            "trigger_price": trigger,
            "stop_price": stop,
            "target_price": target,
            "atr": round(current_atr, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "reasons": reasons,
            "confidence": confidence,
            "score_components": components,
        }, fast

    return {
        "direction": "NO TRADE",
        "score": int(final_score),
        "stock_price": round(price, 2),
        "trigger_price": None,
        "stop_price": None,
        "target_price": None,
        "atr": round(current_atr, 2),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "reasons": reasons + ["No clean directional edge"],
        "confidence": "LOW",
        "score_components": components,
    }, fast


def explain_trade(signal: dict) -> dict:
    score = int(signal.get("score") or 0)
    direction = signal.get("direction", "NO TRADE")
    reasons = signal.get("reasons", []) or []
    reason_text = ", ".join(reasons[:5])

    if direction == "NO TRADE":
        return {
            "quality": "WAIT",
            "quality_class": "avoid",
            "explanation": f"Score {score}/100. No clean signal yet. Notes: {reason_text}.",
        }

    if score >= 80:
        return {
            "quality": "A+",
            "quality_class": "strong",
            "explanation": f"Score {score}/100. Strong {direction} setup. Notes: {reason_text}.",
        }

    if score >= 65:
        return {
            "quality": "B",
            "quality_class": "moderate",
            "explanation": f"Score {score}/100. Decent {direction} setup. Notes: {reason_text}.",
        }

    return {
        "quality": "C",
        "quality_class": "weak",
        "explanation": f"Score {score}/100. Weak {direction} setup. Use smaller risk.",
    }


# ---------------------------
# Tradier options provider
# ---------------------------

def tradier_enabled() -> bool:
    return bool(TRADIER_TOKEN)


def tradier_headers():
    return {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json",
    }


def tradier_get(path: str, params: dict):
    if not tradier_enabled():
        raise RuntimeError("TRADIER_TOKEN is not set.")

    url = f"{TRADIER_BASE_URL}{path}"
    response = requests.get(url, headers=tradier_headers(), params=params, timeout=12)

    if response.status_code == 429:
        raise RuntimeError("Tradier rate limit hit. Returning cached/fallback data if available.")

    if response.status_code >= 400:
        raise RuntimeError(f"Tradier error {response.status_code}: {response.text[:300]}")

    return response.json()


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def tradier_option_dates_in_range(symbol: str, min_dte: int, max_dte: int):
    cache_key = f"tradier_exp:{symbol}:{min_dte}:{max_dte}"
    cached = cache_get(OPTIONS_CACHE, cache_key, OPTIONS_CACHE_SECONDS)
    if cached is not None:
        return cached

    today = date.today()
    valid = []

    data = tradier_get(
        "/markets/options/expirations",
        {
            "symbol": symbol,
            "includeAllRoots": "false",
            "strikes": "false",
        },
    )

    raw_dates = data.get("expirations", {}).get("date", [])
    for exp in normalize_to_list(raw_dates):
        try:
            exp_date = datetime.strptime(str(exp), "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                valid.append((str(exp), dte))
        except Exception:
            continue

    cache_set(OPTIONS_CACHE, cache_key, valid)
    return valid


def tradier_chain_df(symbol: str, expiration: str):
    cache_key = f"tradier_chain:{symbol}:{expiration}"
    cached = cache_get(OPTIONS_CACHE, cache_key, OPTIONS_CACHE_SECONDS)
    if cached is not None:
        return cached.copy()

    data = tradier_get(
        "/markets/options/chains",
        {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true",
        },
    )

    raw_options = data.get("options", {}).get("option", [])
    rows = normalize_to_list(raw_options)

    normalized = []

    for item in rows:
        if not isinstance(item, dict):
            continue

        greeks = item.get("greeks") or {}

        option_type = str(item.get("option_type") or item.get("type") or "").lower()
        if option_type not in {"call", "put"}:
            description = str(item.get("description") or "").lower()
            if " call" in description:
                option_type = "call"
            elif " put" in description:
                option_type = "put"

        bid = safe_float(item.get("bid"), 0.0)
        ask = safe_float(item.get("ask"), 0.0)
        last = safe_float(item.get("last"), 0.0)

        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last

        normalized.append(
            {
                "contractSymbol": item.get("symbol") or item.get("contract_symbol") or "",
                "option_type": option_type,
                "strike": safe_float(item.get("strike"), 0.0),
                "bid": bid,
                "ask": ask,
                "lastPrice": last,
                "mid": mid,
                "volume": safe_int(item.get("volume"), 0),
                "openInterest": safe_int(item.get("open_interest") or item.get("openInterest"), 0),
                "impliedVolatility": safe_float(
                    item.get("implied_volatility")
                    or item.get("iv")
                    or greeks.get("mid_iv")
                    or greeks.get("iv"),
                    0.0,
                ),
                "delta_est": safe_float(greeks.get("delta"), float("nan")),
            }
        )

    df = pd.DataFrame(normalized)

    if df.empty:
        return df

    cache_set(OPTIONS_CACHE, cache_key, df)
    return df.copy()


# ---------------------------
# Yahoo fallback provider
# ---------------------------

def yahoo_option_dates_in_range(ticker, min_dte: int, max_dte: int):
    valid = []
    today = date.today()

    try:
        expirations = ticker.options
    except Exception as e:
        print("YAHOO OPTIONS LIST ERROR:", e)
        return valid

    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if min_dte <= dte <= max_dte:
                valid.append((exp, dte))

        except Exception:
            continue

    return valid


def yahoo_chain_df(ticker, symbol: str, expiration: str, option_side: str):
    time_module.sleep(0.35)

    chain = ticker.option_chain(expiration)
    df = getattr(chain, option_side, None)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["option_type"] = "call" if option_side == "calls" else "put"
    return df


# ---------------------------
# Options scoring
# ---------------------------

def score_option_row(row: pd.Series, stock_price: float, dte: int) -> float:
    mid = safe_float(row.get("mid"), 0.0)
    spread_pct = safe_float(row.get("spread_pct"), 9.99)
    oi = safe_int(row.get("openInterest"), 0)
    vol = safe_int(row.get("volume"), 0)
    strike = safe_float(row.get("strike"), 0.0)

    delta_abs = abs(safe_float(row.get("delta_est"), 0.0))

    liquidity_score = min(oi / 1000, 1.5) + min(vol / 300, 1.0)
    spread_score = max(0.0, 1.4 - spread_pct * 3)
    moneyness_score = max(0.0, 1.8 - abs(strike - stock_price) / max(stock_price * 0.05, 1))
    delta_score = max(0.0, 1.7 - abs(delta_abs - 0.55) * 2.5)
    dte_score = max(0.0, 1.4 - abs(dte - 21) / 25)
    premium_score = 1.0 if 0.10 <= mid <= 100 else 0.25

    return round(
        liquidity_score
        + spread_score
        + moneyness_score
        + delta_score
        + dte_score
        + premium_score,
        4,
    )


def build_option_why(row: pd.Series, dte: int, provider: str, filter_used: str) -> str:
    notes = []

    vol = safe_int(row.get("volume"), 0)
    oi = safe_int(row.get("openInterest"), 0)
    spread = safe_float(row.get("spread_pct"), 9.99)
    delta = safe_float(row.get("delta_est"), 0.0)

    if provider == "tradier":
        notes.append("Tradier data")
    else:
        notes.append("Yahoo fallback")

    if vol >= 100:
        notes.append("good volume")
    elif vol > 0:
        notes.append("some volume")
    else:
        notes.append("low volume")

    if oi >= 500:
        notes.append("solid OI")
    elif oi >= 50:
        notes.append("usable OI")
    else:
        notes.append("low OI")

    if spread <= 0.10:
        notes.append("tight spread")
    elif spread <= 0.35:
        notes.append("acceptable spread")
    else:
        notes.append("wide spread")

    if 14 <= dte <= 35:
        notes.append("good DTE")
    else:
        notes.append("valid DTE")

    if abs(delta) >= 0.40:
        notes.append("stronger delta")
    elif abs(delta) >= 0.20:
        notes.append("usable delta")
    else:
        notes.append("low delta")

    notes.append(filter_used)

    return ", ".join(notes)


def normalize_options_df(df: pd.DataFrame, stock_price: float, dte: int, scan_direction: str, provider: str):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    needed = ["bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike", "contractSymbol"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0

    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.0)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(0.0)

    if "mid" not in df.columns:
        df["mid"] = np.where(
            (df["bid"] > 0) & (df["ask"] > 0),
            (df["bid"] + df["ask"]) / 2,
            df["lastPrice"],
        )
    else:
        df["mid"] = pd.to_numeric(df["mid"], errors="coerce").fillna(0.0)

    df = df[(df["mid"] > 0) | (df["lastPrice"] > 0)].copy()

    if df.empty:
        return df

    df["spread_pct"] = np.where(
        df["mid"] > 0,
        (df["ask"] - df["bid"]).clip(lower=0) / df["mid"],
        9.99,
    )

    if "delta_est" not in df.columns or df["delta_est"].isna().all():
        option_type = "call" if scan_direction == "CALL" else "put"
        years_to_exp = max(dte, 1) / 365.0

        df["delta_est"] = df.apply(
            lambda row: bs_delta(
                stock_price,
                float(row["strike"]),
                years_to_exp,
                RISK_FREE_RATE,
                max(float(row["impliedVolatility"]), 0.0001),
                option_type,
            ),
            axis=1,
        )

    df["distance_from_price"] = (df["strike"] - stock_price).abs()
    df["score"] = df.apply(lambda row: score_option_row(row, stock_price, dte), axis=1)
    df["provider"] = provider

    return df


def row_to_pick(row: pd.Series, scan_direction: str, exp: str, dte: int, provider: str, filter_used: str):
    mid = round(safe_float(row.get("mid"), 0.0), 2)

    return {
        "direction": scan_direction,
        "contract_symbol": str(row.get("contractSymbol", "")),
        "expiration": exp,
        "dte": dte,
        "strike": round(safe_float(row.get("strike"), 0.0), 2),
        "bid": round(safe_float(row.get("bid"), 0.0), 2),
        "ask": round(safe_float(row.get("ask"), 0.0), 2),
        "mid": mid,
        "last": round(safe_float(row.get("lastPrice"), 0.0), 2),
        "volume": safe_int(row.get("volume"), 0),
        "open_interest": safe_int(row.get("openInterest"), 0),
        "iv": round(safe_float(row.get("impliedVolatility"), 0.0) * 100, 2),
        "delta_est": round(safe_float(row.get("delta_est"), 0.0), 3),
        "spread_pct": round(safe_float(row.get("spread_pct"), 9.99) * 100, 2),
        "option_stop": round(mid * (1 - OPTION_STOP_PCT), 2),
        "option_target": round(mid * (1 + OPTION_TARGET_PCT), 2),
        "score": round(safe_float(row.get("score"), 0.0), 3),
        "why": build_option_why(row, dte, provider, filter_used),
        "filter_used": filter_used,
        "provider": provider,
    }


def collect_option_picks_from_df(df: pd.DataFrame, scan_direction: str, exp: str, dte: int, provider: str):
    if df.empty:
        return []

    picks = []

    filter_sets = [
        {"name": "strict", "min_oi": 200, "min_vol": 20, "max_spread": 0.18, "delta_min": 0.35, "delta_max": 0.75},
        {"name": "medium", "min_oi": 25, "min_vol": 0, "max_spread": 0.45, "delta_min": 0.20, "delta_max": 0.90},
        {"name": "loose", "min_oi": 0, "min_vol": 0, "max_spread": 5.00, "delta_min": 0.01, "delta_max": 0.99},
    ]

    for filters in filter_sets:
        fdf = df[
            (df["openInterest"] >= filters["min_oi"])
            & (df["volume"] >= filters["min_vol"])
            & (df["spread_pct"] <= filters["max_spread"])
        ].copy()

        if scan_direction == "CALL":
            fdf = fdf[
                (fdf["delta_est"] >= filters["delta_min"])
                & (fdf["delta_est"] <= filters["delta_max"])
            ].copy()
        else:
            fdf = fdf[
                (fdf["delta_est"] <= -filters["delta_min"])
                & (fdf["delta_est"] >= -filters["delta_max"])
            ].copy()

        if not fdf.empty:
            fdf = fdf.sort_values(
                by=["score", "openInterest", "volume", "distance_from_price"],
                ascending=[False, False, False, True],
            )

            for _, row in fdf.head(4).iterrows():
                picks.append(row_to_pick(row, scan_direction, exp, dte, provider, filters["name"]))

            return picks

    # Final fallback: show closest usable contracts even if Greeks/liquidity are not ideal.
    fallback_df = df.sort_values(
        by=["score", "openInterest", "volume", "distance_from_price"],
        ascending=[False, False, False, True],
    )

    for _, row in fallback_df.head(4).iterrows():
        picks.append(row_to_pick(row, scan_direction, exp, dte, provider, "fallback"))

    return picks


def get_best_options(symbol: str, signal: dict, market_status: str):
    signal_direction = signal.get("direction", "NO TRADE")

    cache_key = f"best_options:{symbol}:{signal_direction}"
    cached = cache_get(OPTIONS_CACHE, cache_key, OPTIONS_CACHE_SECONDS)
    if cached is not None:
        return cached

    stock_price = float(signal.get("stock_price") or 0)

    if stock_price <= 0:
        return []

    if signal_direction == "NO TRADE":
        cache_set(OPTIONS_CACHE, cache_key, [])
        return []

    directions_to_scan = [signal_direction]

    all_picks = []
    provider_used = "none"

    # First choice: Tradier
    if tradier_enabled():
        try:
            valid_dates = tradier_option_dates_in_range(symbol, MIN_DTE, MAX_DTE)
            valid_dates = valid_dates[:MAX_EXPIRATIONS_TO_SCAN]

            for exp, dte in valid_dates:
                chain_df = tradier_chain_df(symbol, exp)

                for scan_direction in directions_to_scan:
                    option_type = "call" if scan_direction == "CALL" else "put"
                    side_df = chain_df[chain_df.get("option_type", "") == option_type].copy()

                    side_df = normalize_options_df(
                        side_df,
                        stock_price=stock_price,
                        dte=dte,
                        scan_direction=scan_direction,
                        provider="tradier",
                    )

                    picks = collect_option_picks_from_df(
                        side_df,
                        scan_direction=scan_direction,
                        exp=exp,
                        dte=dte,
                        provider="tradier",
                    )

                    all_picks.extend(picks)

            provider_used = "tradier"

        except Exception as e:
            print(f"TRADIER OPTIONS ERROR for {symbol}:", e)

    # Fallback: yfinance
    if not all_picks:
        try:
            ticker = yf.Ticker(symbol)
            valid_dates = yahoo_option_dates_in_range(ticker, MIN_DTE, MAX_DTE)
            valid_dates = valid_dates[:MAX_EXPIRATIONS_TO_SCAN]

            for exp, dte in valid_dates:
                for scan_direction in directions_to_scan:
                    option_side = "calls" if scan_direction == "CALL" else "puts"
                    raw_df = yahoo_chain_df(ticker, symbol, exp, option_side)

                    side_df = normalize_options_df(
                        raw_df,
                        stock_price=stock_price,
                        dte=dte,
                        scan_direction=scan_direction,
                        provider="yahoo",
                    )

                    picks = collect_option_picks_from_df(
                        side_df,
                        scan_direction=scan_direction,
                        exp=exp,
                        dte=dte,
                        provider="yahoo",
                    )

                    all_picks.extend(picks)

            provider_used = "yahoo"

        except Exception as e:
            print(f"YAHOO OPTIONS ERROR for {symbol}:", e)

    all_picks.sort(key=lambda x: x["score"], reverse=True)

    deduped = []
    seen = set()

    for pick in all_picks:
        key = pick.get("contract_symbol") or f"{pick['direction']}-{pick['expiration']}-{pick['strike']}"

        if key not in seen:
            pick["provider_used"] = provider_used
            deduped.append(pick)
            seen.add(key)

    result = deduped[:TOP_OPTION_PICKS]
    cache_set(OPTIONS_CACHE, cache_key, result)

    return result


def format_candles(df: pd.DataFrame):
    candles = []

    idx = pd.to_datetime(df.index)

    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(EASTERN)
        else:
            idx = idx.tz_localize(EASTERN)
    except Exception:
        pass

    for ts, row in zip(idx, df.itertuples()):
        try:
            iso_time = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

            candles.append(
                {
                    "time": iso_time,
                    "open": round(float(row.Open), 2),
                    "high": round(float(row.High), 2),
                    "low": round(float(row.Low), 2),
                    "close": round(float(row.Close), 2),
                }
            )

        except Exception:
            continue

    return candles[-240:]


def build_scan(symbol: str):
    now = datetime.now(EASTERN)
    session = market_session_info(now)

    news = get_news(symbol)
    signal, fast = build_signal(symbol, news)

    trade_info = explain_trade(signal)
    signal["trade_quality"] = trade_info["quality"]
    signal["trade_quality_class"] = trade_info["quality_class"]
    signal["trade_explanation"] = trade_info["explanation"]

    try:
        options = get_best_options(symbol, signal, session["status"])
    except Exception as e:
        print(f"OPTIONS ERROR for {symbol}:", e)
        options = []

    try:
        candles = format_candles(fast)
    except Exception as e:
        print(f"CHART ERROR for {symbol}:", e)
        candles = []

    return {
        "symbol": symbol,
        "version": APP_VERSION,
        "signal": signal,
        "options": options,
        "best_option": options[0] if options else None,
        "candles": candles,
        "news": news,
        "market_status": session["status"],
        "extended_hours": session["extended_hours"],
        "session_label": session["session_label"],
        "cached": False,
        "updated_at": now.strftime("%Y-%m-%d %I:%M:%S %p ET"),
    }


@app.route("/")
def home():
    return render_template("index.html", supported_symbols=DEFAULT_SYMBOLS)


@app.route("/version")
def version():
    return jsonify(
        {
            "version": APP_VERSION,
            "min_dte": MIN_DTE,
            "max_dte": MAX_DTE,
            "cache_seconds": CACHE_SECONDS,
            "options_cache_seconds": OPTIONS_CACHE_SECONDS,
            "max_expirations_to_scan": MAX_EXPIRATIONS_TO_SCAN,
            "tradier_enabled": tradier_enabled(),
        }
    )


@app.route("/debug-options")
def debug_options():
    symbol = normalize_symbol(request.args.get("symbol", "SPY"))

    result = {
        "symbol": symbol,
        "version": APP_VERSION,
        "tradier_enabled": tradier_enabled(),
        "min_dte": MIN_DTE,
        "max_dte": MAX_DTE,
    }

    if tradier_enabled():
        try:
            dates = tradier_option_dates_in_range(symbol, MIN_DTE, MAX_DTE)
            result["tradier_expiration_count"] = len(dates)
            result["tradier_expirations"] = dates[:5]

            if dates:
                exp, _dte = dates[0]
                df = tradier_chain_df(symbol, exp)
                result["tradier_test_expiration"] = exp
                result["tradier_chain_rows"] = int(len(df))
                result["tradier_calls"] = int((df.get("option_type") == "call").sum()) if not df.empty else 0
                result["tradier_puts"] = int((df.get("option_type") == "put").sum()) if not df.empty else 0

        except Exception as e:
            result["tradier_error"] = str(e)

    try:
        ticker = yf.Ticker(symbol)
        yahoo_dates = yahoo_option_dates_in_range(ticker, MIN_DTE, MAX_DTE)
        result["yahoo_expiration_count"] = len(yahoo_dates)
        result["yahoo_expirations"] = yahoo_dates[:5]
    except Exception as e:
        result["yahoo_error"] = str(e)

    return jsonify(result)


@app.route("/scan")
def scan():
    symbol = normalize_symbol(request.args.get("symbol", "SPY"))
    force = request.args.get("force", "0") == "1"
    cache_key = f"scan:{symbol}"

    now = datetime.now(EASTERN)
    current_session = market_session_info(now)

    cached = cache_get(SCAN_CACHE, cache_key, CACHE_SECONDS)

    if cached is not None and not force:
        cached_status = cached.get("market_status")

        if cached_status == current_session["status"]:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            return jsonify({"success": True, "data": cached_copy})

    try:
        data = build_scan(symbol)
        data["cached"] = False
        data["updated_at"] = datetime.now(EASTERN).strftime("%Y-%m-%d %I:%M:%S %p ET")

        cache_set(SCAN_CACHE, cache_key, data)

        return jsonify({"success": True, "data": data})

    except Exception as e:
        print(f"SCAN ERROR for {symbol}:", str(e))

        if cached is not None:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            cached_copy["warning"] = "Live scan failed. Showing cached data."
            cached_copy["error"] = str(e)
            return jsonify({"success": True, "data": cached_copy})

        return jsonify({"success": False, "error": str(e), "version": APP_VERSION}), 500


@app.route("/quote")
def quote():
    symbol = normalize_symbol(request.args.get("symbol", "SPY"))
    cache_key = f"quote:{symbol}"

    cached = cache_get(QUOTE_CACHE, cache_key, 15)
    if cached is not None:
        return jsonify(cached)

    try:
        ticker = yf.Ticker(symbol)

        # LIVE intraday data
        hist = ticker.history(
            period="1d",
            interval="1m",
            prepost=True
        )

        if hist.empty:
            return jsonify({
                "success": False,
                "error": "No quote found"
            }), 404

        latest = hist.iloc[-1]

        price = float(latest["Close"])

        payload = {
            "success": True,
            "symbol": symbol,
            "price": round(price, 2),
            "updated": datetime.now(EASTERN).strftime("%I:%M:%S %p"),
            "version": APP_VERSION,
        }

        cache_set(QUOTE_CACHE, cache_key, payload)

        return jsonify(payload)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "version": APP_VERSION,
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)