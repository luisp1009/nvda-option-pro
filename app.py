import math
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DEFAULT_SYMBOLS = ["SPY", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "AMZN", "META", "QQQ"]

MIN_DTE = 7
MAX_DTE = 35
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


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "SPY").upper().strip()
    symbol = symbol.replace("$", "").replace(" ", "")
    symbol = symbol.replace(".", "-")
    return symbol or "SPY"


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

    return items


def get_price_data(symbol: str):
    fast = yf.download(
        symbol,
        period="5d",
        interval="1m",
        auto_adjust=False,
        progress=False,
        prepost=True,
    )

    if fast.empty:
        fast = yf.download(
            symbol,
            period="10d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            prepost=True,
        )

    slow = yf.download(
        symbol,
        period="60d",
        interval="30m",
        auto_adjust=False,
        progress=False,
        prepost=True,
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


def option_dates_in_range(ticker, min_dte: int, max_dte: int):
    valid = []
    today = date.today()

    try:
        expirations = ticker.options
    except Exception as e:
        print("OPTIONS LIST ERROR:", e)
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


def score_option_row(row: pd.Series, stock_price: float, dte: int) -> float:
    mid = float(row["mid"])
    spread_pct = float(row["spread_pct"])
    oi = int(row["openInterest"])
    vol = int(row["volume"])
    strike = float(row["strike"])

    delta_abs = abs(float(row["delta_est"])) if not math.isnan(float(row["delta_est"])) else 0.0

    liquidity_score = min(oi / 1000, 1.5) + min(vol / 300, 1.0)
    spread_score = max(0.0, 1.4 - spread_pct * 4)
    moneyness_score = max(0.0, 1.7 - abs(strike - stock_price) / max(stock_price * 0.04, 1))
    delta_score = max(0.0, 1.7 - abs(delta_abs - 0.55) * 3)
    dte_score = max(0.0, 1.3 - abs(dte - 21) / 18)
    premium_penalty = 0.0 if 0.25 <= mid <= 80 else 0.4

    return round(
        liquidity_score
        + spread_score
        + moneyness_score
        + delta_score
        + dte_score
        - premium_penalty,
        4,
    )


def build_option_why(row: pd.Series, dte: int) -> str:
    notes = []

    vol = int(row.get("volume", 0))
    oi = int(row.get("openInterest", 0))
    spread = float(row.get("spread_pct", 999))

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
    elif oi > 0:
        notes.append("low OI")

    if spread <= 0.08:
        notes.append("tight spread")
    elif spread <= 0.25:
        notes.append("acceptable spread")
    else:
        notes.append("wide spread")

    if 14 <= dte <= 28:
        notes.append("good DTE")
    elif 7 <= dte <= 35:
        notes.append("valid DTE")

    return ", ".join(notes)


def collect_option_picks_for_direction(ticker, symbol, scan_direction, stock_price, valid_dates):
    option_side = "calls" if scan_direction == "CALL" else "puts"
    option_type = "call" if scan_direction == "CALL" else "put"

    filter_sets = [
        {"name": "strict", "min_oi": 200, "min_vol": 20, "max_spread": 0.18, "delta_min": 0.35, "delta_max": 0.75},
        {"name": "medium", "min_oi": 50, "min_vol": 1, "max_spread": 0.30, "delta_min": 0.25, "delta_max": 0.85},
        {"name": "loose", "min_oi": 0, "min_vol": 0, "max_spread": 0.60, "delta_min": 0.15, "delta_max": 0.95},
    ]

    for filters in filter_sets:
        picks = []

        for exp, dte in valid_dates:
            try:
                chain = ticker.option_chain(exp)
                df = getattr(chain, option_side, None)

                if df is None or df.empty:
                    continue

                df = df.copy()

                df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
                df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
                df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce").fillna(0.0)
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
                df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.0)
                df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(0.0)

                df["mid"] = np.where(
                    (df["bid"] > 0) & (df["ask"] > 0),
                    (df["bid"] + df["ask"]) / 2,
                    df["lastPrice"],
                )

                df = df[df["mid"] > 0]

                if df.empty:
                    continue

                df["spread_pct"] = np.where(
                    df["mid"] > 0,
                    (df["ask"] - df["bid"]).clip(lower=0) / df["mid"],
                    999,
                )

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

                df = df[
                    (df["openInterest"] >= filters["min_oi"])
                    & (df["volume"] >= filters["min_vol"])
                    & (df["spread_pct"] <= filters["max_spread"])
                ].copy()

                if scan_direction == "CALL":
                    df = df[
                        (df["delta_est"] >= filters["delta_min"])
                        & (df["delta_est"] <= filters["delta_max"])
                    ]
                else:
                    df = df[
                        (df["delta_est"] <= -filters["delta_min"])
                        & (df["delta_est"] >= -filters["delta_max"])
                    ]

                if df.empty:
                    continue

                df["score"] = df.apply(lambda row: score_option_row(row, stock_price, dte), axis=1)
                df = df.sort_values("score", ascending=False)

                for _, row in df.head(5).iterrows():
                    mid = round(float(row["mid"]), 2)

                    picks.append(
                        {
                            "direction": scan_direction,
                            "contract_symbol": str(row.get("contractSymbol", "")),
                            "expiration": exp,
                            "dte": dte,
                            "strike": round(float(row["strike"]), 2),
                            "bid": round(float(row["bid"]), 2),
                            "ask": round(float(row["ask"]), 2),
                            "mid": mid,
                            "last": round(float(row["lastPrice"]), 2),
                            "volume": int(row["volume"]),
                            "open_interest": int(row["openInterest"]),
                            "iv": round(float(row["impliedVolatility"]) * 100, 2),
                            "delta_est": round(float(row["delta_est"]), 3),
                            "spread_pct": round(float(row["spread_pct"]) * 100, 2),
                            "option_stop": round(mid * (1 - OPTION_STOP_PCT), 2),
                            "option_target": round(mid * (1 + OPTION_TARGET_PCT), 2),
                            "score": round(float(row["score"]), 3),
                            "why": build_option_why(row, dte),
                            "filter_used": filters["name"],
                        }
                    )

            except Exception as e:
                print(f"OPTION PROCESSING ERROR for {symbol} {exp} {scan_direction}:", e)

        if picks:
            print(f"OPTION FILTER USED for {symbol} {scan_direction}:", filters["name"])
            return picks

    return []


def get_best_options(symbol: str, signal: dict, market_status: str):
    ticker = yf.Ticker(symbol)
    stock_price = float(signal.get("stock_price") or 0)

    if stock_price <= 0:
        return []

    signal_direction = signal.get("direction", "NO TRADE")

    if signal_direction in {"CALL", "PUT"}:
        directions_to_scan = [signal_direction]
    else:
        directions_to_scan = ["CALL", "PUT"]

    valid_dates = option_dates_in_range(ticker, MIN_DTE, MAX_DTE)

    if not valid_dates:
        print(f"NO VALID EXPIRATIONS FOUND FOR {symbol}")
        return []

    all_picks = []

    for scan_direction in directions_to_scan:
        all_picks.extend(
            collect_option_picks_for_direction(
                ticker=ticker,
                symbol=symbol,
                scan_direction=scan_direction,
                stock_price=stock_price,
                valid_dates=valid_dates,
            )
        )

    all_picks.sort(key=lambda x: x["score"], reverse=True)

    deduped = []
    seen = set()

    for pick in all_picks:
        key = pick.get("contract_symbol") or f"{pick['direction']}-{pick['expiration']}-{pick['strike']}"
        if key not in seen:
            deduped.append(pick)
            seen.add(key)

    return deduped[:TOP_OPTION_PICKS]


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
        "signal": signal,
        "options": options,
        "best_option": options[0] if options else None,
        "candles": candles,
        "news": news,
        "market_status": session["status"],
        "extended_hours": session["extended_hours"],
        "session_label": session["session_label"],
        "updated_at": now.strftime("%Y-%m-%d %I:%M:%S %p ET"),
    }


@app.route("/")
def home():
    return render_template("index.html", supported_symbols=DEFAULT_SYMBOLS)


@app.route("/scan")
def scan():
    symbol = normalize_symbol(request.args.get("symbol", "SPY"))

    try:
        data = build_scan(symbol)

        print("SCAN OK")
        print("Symbol:", symbol)
        print("Signal:", data["signal"]["direction"])
        print("Score:", data["signal"]["score"])
        print("Market status:", data["market_status"])
        print("Options count:", len(data["options"]))
        print("News count:", len(data["news"]))
        print("Candles count:", len(data["candles"]))

        return jsonify({"success": True, "data": data})

    except Exception as e:
        print(f"SCAN ERROR for {symbol}:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/quote")
def quote():
    symbol = normalize_symbol(request.args.get("symbol", "SPY"))

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d", interval="1d", prepost=True)

        if hist.empty:
            return jsonify({"success": False, "error": "No quote found"}), 404

        price = float(hist["Close"].dropna().iloc[-1])

        return jsonify({
            "success": True,
            "symbol": symbol,
            "price": round(price, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)