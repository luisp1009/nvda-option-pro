import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

APP_VERSION = "options-render-fallback-v4"

DEFAULT_SYMBOL = "NVDA"

MIN_DTE = 1
MAX_DTE = 60

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


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or DEFAULT_SYMBOL).upper().strip()
    symbol = symbol.replace("$", "").replace(" ", "")
    symbol = symbol.replace(".", "-")
    return symbol or DEFAULT_SYMBOL


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    return out.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
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


def bs_delta(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str
) -> float:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        return float("nan")

    d1 = (
        math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t
    ) / (sigma * math.sqrt(t))

    if option_type.lower() == "call":
        return norm_cdf(d1)

    return norm_cdf(d1) - 1.0


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def get_price_data(symbol: str):
    fast = yf.download(
        symbol,
        period="7d",
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
        raise ValueError(f"Could not load market data for {symbol} right now.")

    for frame in (fast, slow):
        frame["ema_fast"] = ema(frame["Close"], EMA_FAST)
        frame["ema_slow"] = ema(frame["Close"], EMA_SLOW)
        frame["rsi"] = rsi(frame["Close"], RSI_LEN)
        frame["atr"] = atr(frame, ATR_LEN)

    fast["vwap"] = intraday_vwap(fast)

    return fast, slow


def build_signal(symbol: str):
    fast, slow = get_price_data(symbol)

    if len(fast) < 30:
        raise ValueError(f"Not enough candle data for {symbol}.")

    f = fast.iloc[-1]
    s = slow.iloc[-1]
    prev_fast = fast.iloc[-2]

    price = float(f["Close"])
    current_atr = float(f["atr"])

    if math.isnan(current_atr) or current_atr <= 0:
        current_atr = max(price * 0.005, 0.01)

    recent_high = float(fast["High"].tail(20).max())
    recent_low = float(fast["Low"].tail(20).min())

    bull_score = 0.0
    bear_score = 0.0
    reasons = []

    if f["Close"] > f["ema_fast"] > f["ema_slow"]:
        bull_score += 2.0
        reasons.append("5m bullish EMA trend")

    if f["Close"] < f["ema_fast"] < f["ema_slow"]:
        bear_score += 2.0
        reasons.append("5m bearish EMA trend")

    if s["Close"] > s["ema_fast"] > s["ema_slow"]:
        bull_score += 2.0
        reasons.append("30m bullish EMA trend")

    if s["Close"] < s["ema_fast"] < s["ema_slow"]:
        bear_score += 2.0
        reasons.append("30m bearish EMA trend")

    if pd.notna(f["vwap"]) and f["Close"] > f["vwap"]:
        bull_score += 1.5
        reasons.append("Price above VWAP")

    if pd.notna(f["vwap"]) and f["Close"] < f["vwap"]:
        bear_score += 1.5
        reasons.append("Price below VWAP")

    if 52 <= f["rsi"] <= 72:
        bull_score += 1.0
        reasons.append("RSI supports upside")

    if 28 <= f["rsi"] <= 48:
        bear_score += 1.0
        reasons.append("RSI supports downside")

    if f["Close"] > recent_high - current_atr * 0.25 and prev_fast["Close"] <= recent_high:
        bull_score += 1.5
        reasons.append("Near breakout area")

    if f["Close"] < recent_low + current_atr * 0.25 and prev_fast["Close"] >= recent_low:
        bear_score += 1.5
        reasons.append("Near breakdown area")

    if bull_score >= bear_score + 1.0:
        trigger = round(recent_high + current_atr * ENTRY_BUFFER_ATR, 2)
        stop = round(trigger - current_atr * STOP_ATR_MULT, 2)
        risk = trigger - stop
        target = round(trigger + risk * TARGET_R_MULT, 2)
        score = min(round((bull_score / 8) * 100), 100)

        return {
            "direction": "CALL",
            "score": score,
            "stock_price": round(price, 2),
            "trigger_price": trigger,
            "stop_price": stop,
            "target_price": target,
            "atr": round(current_atr, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "confidence": "HIGH" if score >= 80 else "MEDIUM",
            "trade_quality": "A+" if score >= 80 else "B",
            "trade_quality_class": "strong" if score >= 80 else "moderate",
            "trade_explanation": f"CALL setup. Score {score}/100. " + ", ".join(reasons[:5]),
            "reasons": reasons,
        }

    if bear_score >= bull_score + 1.0:
        trigger = round(recent_low - current_atr * ENTRY_BUFFER_ATR, 2)
        stop = round(trigger + current_atr * STOP_ATR_MULT, 2)
        risk = stop - trigger
        target = round(trigger - risk * TARGET_R_MULT, 2)
        score = min(round((bear_score / 8) * 100), 100)

        return {
            "direction": "PUT",
            "score": score,
            "stock_price": round(price, 2),
            "trigger_price": trigger,
            "stop_price": stop,
            "target_price": target,
            "atr": round(current_atr, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "confidence": "HIGH" if score >= 80 else "MEDIUM",
            "trade_quality": "A+" if score >= 80 else "B",
            "trade_quality_class": "strong" if score >= 80 else "moderate",
            "trade_explanation": f"PUT setup. Score {score}/100. " + ", ".join(reasons[:5]),
            "reasons": reasons,
        }

    score = min(round((max(bull_score, bear_score) / 8) * 100), 100)

    return {
        "direction": "NO TRADE",
        "score": score,
        "stock_price": round(price, 2),
        "trigger_price": None,
        "stop_price": None,
        "target_price": None,
        "atr": round(current_atr, 2),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "confidence": "LOW",
        "trade_quality": "WAIT",
        "trade_quality_class": "avoid",
        "trade_explanation": f"Score {score}/100. No clean directional edge yet. " + ", ".join(reasons[:5]),
        "reasons": reasons,
    }


def option_dates_in_range(ticker, min_dte: int, max_dte: int):
    valid = []
    today = date.today()

    try:
        expirations = ticker.options
    except Exception as e:
        print("OPTIONS DATE ERROR:", e)
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


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def score_option_row(row: pd.Series, stock_price: float, dte: int) -> float:
    mid = safe_float(row.get("mid"), 0.0)
    spread_pct = safe_float(row.get("spread_pct"), 9.99)
    oi = int(safe_float(row.get("openInterest"), 0))
    vol = int(safe_float(row.get("volume"), 0))
    strike = safe_float(row.get("strike"), 0.0)
    delta_abs = abs(safe_float(row.get("delta_est"), 0.0))

    liquidity_score = min(oi / 1000, 1.5) + min(vol / 300, 1.0)
    spread_score = max(0.0, 1.4 - spread_pct * 3)
    moneyness_score = max(0.0, 1.7 - abs(strike - stock_price) / max(stock_price * 0.05, 1))
    delta_score = max(0.0, 1.7 - abs(delta_abs - 0.55) * 2.5)
    dte_score = max(0.0, 1.3 - abs(dte - 21) / 25)
    premium_score = 1.0 if 0.10 <= mid <= 100 else 0.3

    return round(
        liquidity_score
        + spread_score
        + moneyness_score
        + delta_score
        + dte_score
        + premium_score,
        4,
    )


def build_option_why(row: pd.Series, dte: int, filter_name: str) -> str:
    vol = int(safe_float(row.get("volume"), 0))
    oi = int(safe_float(row.get("openInterest"), 0))
    spread = safe_float(row.get("spread_pct"), 9.99)
    delta = safe_float(row.get("delta_est"), 0.0)

    notes = []

    notes.append("good volume" if vol >= 100 else "some volume" if vol > 0 else "low volume")
    notes.append("solid open interest" if oi >= 500 else "usable open interest" if oi >= 50 else "low open interest")
    notes.append("tight spread" if spread <= 0.10 else "acceptable spread" if spread <= 0.35 else "wide spread")
    notes.append("good expiration range" if 14 <= dte <= 35 else "short/long DTE")
    notes.append("stronger delta" if abs(delta) >= 0.40 else "usable delta" if abs(delta) >= 0.20 else "low delta")
    notes.append(f"{filter_name} filter")

    return ", ".join(notes)


def normalize_option_chain_df(df: pd.DataFrame, stock_price: float, dte: int, option_type: str) -> pd.DataFrame:
    df = df.copy()

    needed = ["bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "strike"]
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

    df["mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2,
        df["lastPrice"],
    )

    df = df[(df["mid"] > 0) | (df["lastPrice"] > 0)].copy()

    if df.empty:
        return df

    df["spread_pct"] = np.where(
        df["mid"] > 0,
        (df["ask"] - df["bid"]).clip(lower=0) / df["mid"],
        9.99,
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

    df["distance_from_price"] = (df["strike"] - stock_price).abs()
    df["score"] = df.apply(lambda row: score_option_row(row, stock_price, dte), axis=1)

    return df


def row_to_pick(row: pd.Series, scan_direction: str, exp: str, dte: int, filter_name: str) -> dict:
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
        "volume": int(safe_float(row.get("volume"), 0)),
        "open_interest": int(safe_float(row.get("openInterest"), 0)),
        "iv": round(safe_float(row.get("impliedVolatility"), 0.0) * 100, 2),
        "delta_est": round(safe_float(row.get("delta_est"), 0.0), 3),
        "spread_pct": round(safe_float(row.get("spread_pct"), 9.99) * 100, 2),
        "option_stop": round(mid * (1 - OPTION_STOP_PCT), 2),
        "option_target": round(mid * (1 + OPTION_TARGET_PCT), 2),
        "score": round(safe_float(row.get("score"), 0.0), 3),
        "why": build_option_why(row, dte, filter_name),
        "filter_used": filter_name,
    }


def collect_option_picks_for_direction(
    ticker,
    symbol: str,
    scan_direction: str,
    stock_price: float,
    valid_dates
):
    option_side = "calls" if scan_direction == "CALL" else "puts"
    option_type = "call" if scan_direction == "CALL" else "put"

    all_candidates = []

    filter_sets = [
        {"name": "strict", "min_oi": 200, "min_vol": 20, "max_spread": 0.18, "delta_min": 0.35, "delta_max": 0.75},
        {"name": "medium", "min_oi": 25, "min_vol": 0, "max_spread": 0.45, "delta_min": 0.20, "delta_max": 0.90},
        {"name": "loose", "min_oi": 0, "min_vol": 0, "max_spread": 5.00, "delta_min": 0.01, "delta_max": 0.99},
    ]

    for exp, dte in valid_dates:
        try:
            chain = ticker.option_chain(exp)
            df = getattr(chain, option_side, None)

            if df is None or df.empty:
                print(symbol, exp, scan_direction, "empty option chain")
                continue

            df = normalize_option_chain_df(df, stock_price, dte, option_type)

            if df.empty:
                print(symbol, exp, scan_direction, "no contracts with price")
                continue

            print(symbol, exp, scan_direction, "usable rows:", len(df))

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

                    for _, row in fdf.head(5).iterrows():
                        all_candidates.append(row_to_pick(row, scan_direction, exp, dte, filters["name"]))

            fallback_df = df.sort_values(
                by=["score", "openInterest", "volume", "distance_from_price"],
                ascending=[False, False, False, True],
            )

            for _, row in fallback_df.head(3).iterrows():
                all_candidates.append(row_to_pick(row, scan_direction, exp, dte, "fallback"))

        except Exception as e:
            print(f"OPTION PROCESSING ERROR for {symbol} {exp} {scan_direction}:", e)

    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    deduped = []
    seen = set()

    for pick in all_candidates:
        key = pick.get("contract_symbol") or f"{pick['direction']}-{pick['expiration']}-{pick['strike']}"
        if key not in seen:
            deduped.append(pick)
            seen.add(key)

    return deduped[:TOP_OPTION_PICKS]


def get_best_options(symbol: str, signal: dict):
    ticker = yf.Ticker(symbol)
    stock_price = float(signal.get("stock_price") or 0)

    if stock_price <= 0:
        return []

    valid_dates = option_dates_in_range(ticker, MIN_DTE, MAX_DTE)

    print("VALID OPTION DATES:", valid_dates)

    if not valid_dates:
        return []

    all_picks = []

    for scan_direction in ["CALL", "PUT"]:
        picks = collect_option_picks_for_direction(
            ticker=ticker,
            symbol=symbol,
            scan_direction=scan_direction,
            stock_price=stock_price,
            valid_dates=valid_dates,
        )

        all_picks.extend(picks)

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

    for ts, row in zip(idx, df.itertuples()):
        try:
            candles.append(
                {
                    "time": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": round(float(row.Open), 2),
                    "high": round(float(row.High), 2),
                    "low": round(float(row.Low), 2),
                    "close": round(float(row.Close), 2),
                }
            )
        except Exception:
            continue

    return candles[-240:]


def get_news(symbol: str):
    ticker = yf.Ticker(symbol)

    try:
        raw_news = ticker.news
    except Exception as e:
        print(f"NEWS ERROR for {symbol}:", e)
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

            title = (
                item.get("title")
                or content.get("title")
                or "Headline unavailable"
            )

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
                dt = datetime.fromtimestamp(timestamp)
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


def build_scan(symbol: str):
    signal = build_signal(symbol)
    options = get_best_options(symbol, signal)

    try:
        fast, _ = get_price_data(symbol)
        candles = format_candles(fast)
    except Exception as e:
        print("CANDLE ERROR:", e)
        candles = []

    try:
        news = get_news(symbol)
    except Exception as e:
        print("NEWS ERROR:", e)
        news = []

    return {
        "symbol": symbol,
        "version": APP_VERSION,
        "signal": signal,
        "options": options,
        "best_option": options[0] if options else None,
        "candles": candles,
        "news": news,
        "market_status": "OPEN",
        "updated_at": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/version")
def version():
    return jsonify(
        {
            "version": APP_VERSION,
            "min_dte": MIN_DTE,
            "max_dte": MAX_DTE,
            "top_option_picks": TOP_OPTION_PICKS,
        }
    )


@app.route("/debug-options")
def debug_options():
    symbol = normalize_symbol(request.args.get("symbol", DEFAULT_SYMBOL))

    try:
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options)

        result = {
            "symbol": symbol,
            "version": APP_VERSION,
            "min_dte": MIN_DTE,
            "max_dte": MAX_DTE,
            "expiration_count": len(expirations),
            "expirations": expirations[:10],
        }

        if expirations:
            exp = expirations[0]
            chain = ticker.option_chain(exp)

            result["test_expiration"] = exp
            result["calls_count"] = len(chain.calls)
            result["puts_count"] = len(chain.puts)
            result["sample_calls"] = chain.calls.head(3).to_dict("records")
            result["sample_puts"] = chain.puts.head(3).to_dict("records")

        return jsonify(result)

    except Exception as e:
        return jsonify(
            {
                "symbol": symbol,
                "version": APP_VERSION,
                "error": str(e),
            }
        ), 500


@app.route("/scan")
def scan():
    symbol = normalize_symbol(request.args.get("symbol", DEFAULT_SYMBOL))

    try:
        data = build_scan(symbol)

        print("SCAN OK")
        print("Version:", APP_VERSION)
        print("Symbol:", symbol)
        print("Signal:", data["signal"]["direction"])
        print("Score:", data["signal"]["score"])
        print("Options count:", len(data["options"]))
        print("News count:", len(data["news"]))
        print("Candles count:", len(data["candles"]))

        return jsonify({"success": True, "data": data})

    except Exception as e:
        print(f"SCAN ERROR for {symbol}:", str(e))
        return jsonify({"success": False, "error": str(e), "version": APP_VERSION}), 500


@app.route("/quote")
def quote():
    symbol = normalize_symbol(request.args.get("symbol", DEFAULT_SYMBOL))

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d", interval="1d", prepost=True)

        if hist.empty:
            return jsonify({"success": False, "error": "No quote found"}), 404

        price = float(hist["Close"].dropna().iloc[-1])

        return jsonify(
            {
                "success": True,
                "symbol": symbol,
                "price": round(price, 2),
                "version": APP_VERSION,
            }
        )

    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "version": APP_VERSION,
            }
        ), 500


if __name__ == "__main__":
    app.run(debug=True)