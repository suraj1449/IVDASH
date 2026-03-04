import os
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, time as dtime
from flask import Flask, render_template_string, jsonify
from kiteconnect import KiteConnect
from scipy.stats import norm
import json

# ================= CONFIG (from env vars or fallback) ================= #
API_KEY      = os.environ.get("API_KEY", "oxc7tfxwa3n5zvqq")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "nPCK15pKXoHi2wAU6bkKedur1oykpeGd")

INTERVAL     = int(os.environ.get("INTERVAL", "60"))
STRIKE_GAP   = int(os.environ.get("STRIKE_GAP", "50"))
STRIKE_COUNT = int(os.environ.get("STRIKE_COUNT", "4"))
r            = float(os.environ.get("RISK_FREE_RATE", "0.10"))
# ====================================================================== #

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

app = Flask(__name__)

iv_history    = {}
vol_history   = {}
base_iv       = {}
live_snapshot = {}
ltp_snapshot  = {}
time_series   = []
iv_series     = {}
oi_data       = {}

atm_strike_fixed = None
EXPIRY           = None


# ================= AUTO EXPIRY CALCULATION ================= #

def get_nearest_tuesday_expiry():
    now          = datetime.now()
    today        = now.date()
    current_time = now.time()
    weekday      = today.weekday()  # 0=Mon … 6=Sun

    if weekday == 0:
        days_ahead = 1
    elif weekday == 1:
        days_ahead = 7 if current_time >= dtime(15, 30) else 0
    else:
        days_ahead = (1 - weekday) % 7

    expiry_date = today + pd.Timedelta(days=days_ahead)

    if expiry_date.month == 12:
        next_month = expiry_date.replace(year=expiry_date.year + 1, month=1, day=1)
    else:
        next_month = expiry_date.replace(month=expiry_date.month + 1, day=1)

    last_day_of_month = next_month - pd.Timedelta(days=1)
    last_tuesday      = last_day_of_month
    while last_tuesday.weekday() != 1:
        last_tuesday -= pd.Timedelta(days=1)

    is_monthly    = (expiry_date == last_tuesday)
    year_2digit   = expiry_date.year % 100

    if is_monthly:
        month_name = expiry_date.strftime('%b').upper()
        expiry_str = f"{year_2digit:02d}{month_name}"
        expiry_type = "MONTHLY"
    else:
        month_num  = expiry_date.month
        day        = expiry_date.day
        expiry_str = f"{year_2digit:02d}{month_num}{day:02d}"
        expiry_type = "WEEKLY"

    print(f"📅 Auto-selected expiry: {expiry_date.strftime('%Y-%m-%d (%A)')} → {expiry_str} [{expiry_type}]")
    return expiry_str, expiry_date, is_monthly


# ================= TIME TO EXPIRY ================= #

def get_time_to_expiry():
    if EXPIRY is None:
        return 1 / 365.0, 1

    if any(c.isalpha() for c in EXPIRY):
        year_2digit = int(EXPIRY[0:2])
        month_str   = EXPIRY[2:5]
        month_map   = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month       = month_map.get(month_str, 1)
        expiry_year = 2000 + year_2digit

        if month == 12:
            next_month_date = datetime(expiry_year + 1, 1, 1).date()
        else:
            next_month_date = datetime(expiry_year, month + 1, 1).date()

        last_day_of_month = next_month_date - pd.Timedelta(days=1)
        expiry_date       = last_day_of_month
        while expiry_date.weekday() != 1:
            expiry_date -= pd.Timedelta(days=1)
    else:
        year_2digit = int(EXPIRY[0:2])
        month       = int(EXPIRY[2:3])
        day         = int(EXPIRY[3:5])
        expiry_year = 2000 + year_2digit
        expiry_date = datetime(expiry_year, month, day).date()

    today   = datetime.now().date()
    weekday = today.weekday()
    if weekday == 5:
        today -= pd.Timedelta(days=1)
    elif weekday == 6:
        today -= pd.Timedelta(days=2)

    days_remaining = max((expiry_date - today).days, 1)
    return days_remaining / 365.0, days_remaining


# ================= BLACK-SCHOLES IV ================= #

def implied_volatility(option_price, S, K, T, r, option_type="CE"):
    if option_price < 0.05 or T <= 0:
        return np.nan

    sigma      = max(0.01, min((option_price / S) * np.sqrt(2 * np.pi / T), 5.0))
    prev_sigma = sigma

    for _ in range(500):
        try:
            d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2   = d1 - sigma * np.sqrt(T)
            price = (
                S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                if option_type == "CE"
                else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            )
            vega = S * norm.pdf(d1) * np.sqrt(T)
            if vega < 1e-10:
                return np.nan
            diff  = price - option_price
            if abs(diff) < 0.0001:
                break
            sigma -= diff / vega
            sigma  = max(0.01, min(sigma, 5.0))
            if abs(sigma - prev_sigma) < 1e-8:
                break
            prev_sigma = sigma
        except Exception:
            return np.nan

    return round(sigma * 100, 2)


# ================= HELPERS ================= #

def get_atm_strike():
    spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
    return round(spot / STRIKE_GAP) * STRIKE_GAP


def update_base(symbol, iv):
    now = datetime.now().time()
    if dtime(9, 15) <= now <= dtime(10, 0):
        base_iv[symbol] = min(base_iv.get(symbol, iv), iv)


def sma(values):
    if len(values) < 9:
        return np.nan
    return round(pd.Series(values).rolling(9).mean().iloc[-1], 2)


def get_iv_avg(symbol, n):
    hist  = iv_history.get(symbol, [])
    valid = [v for v in hist[-n:] if not np.isnan(v)]
    return round(np.mean(valid), 2) if valid else np.nan


def get_vol_times(symbol, n):
    hist     = vol_history.get(symbol, [])
    sma9_vol = sma(hist)
    if np.isnan(sma9_vol) or sma9_vol == 0:
        return np.nan
    recent = [v for v in hist[-n:] if v > 0]
    return round(np.mean(recent) / sma9_vol, 2) if recent else np.nan


def classify_event(iv, base_val, sma9_val):
    if np.isnan(iv) or np.isnan(base_val):
        return "N/A", "na"
    if iv > 3 + base_val or (not np.isnan(sma9_val) and iv < sma9_val):
        return "EXIT/PANIC", "exit"
    if 1 + base_val <= iv <= 2.5 + base_val and (not np.isnan(sma9_val) and iv > sma9_val):
        return "ENTRY", "entry"
    if 0.6 + base_val <= iv < 1 + base_val:
        return "WATCH", "watch"
    if base_val <= iv <= 0.5 + base_val:
        return "NO TRADE", "no-trade"
    return "N/A", "na"


def smooth_iv_series(values):
    if len(values) < 3:
        return values
    s         = pd.Series(values, dtype=float)
    roll_mean = s.rolling(window=5, min_periods=1, center=True).mean()
    roll_std  = s.rolling(window=5, min_periods=1, center=True).std().fillna(0)
    roll_med  = s.rolling(window=5, min_periods=1, center=True).median()
    is_spike  = (s - roll_mean).abs() > 2.5 * roll_std
    s         = s.where(~is_spike, roll_med)
    s         = s.ewm(span=12, adjust=False).mean()
    result    = []
    for orig, sm in zip(values, s):
        if orig is None or (isinstance(orig, float) and np.isnan(orig)):
            result.append(None)
        else:
            result.append(round(float(sm), 2))
    return result


# ================= BACKGROUND LOOPS ================= #

def fetch_loop():
    global atm_strike_fixed, EXPIRY

    if EXPIRY is None:
        expiry_str, _, _ = get_nearest_tuesday_expiry()
        EXPIRY = expiry_str

    while True:
        try:
            now = datetime.now().time()

            if atm_strike_fixed is None and now >= dtime(9, 15):
                atm_strike_fixed = get_atm_strike()
                print("✅ ATM Fixed:", atm_strike_fixed)

            if atm_strike_fixed is None:
                time.sleep(5)
                continue

            strikes = [
                atm_strike_fixed + i * STRIKE_GAP
                for i in range(-STRIKE_COUNT, STRIKE_COUNT + 1)
            ]

            current_time = datetime.now().strftime("%H:%M")
            time_series.append(current_time)

            spot            = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
            T, days_remaining = get_time_to_expiry()
            print(f"⏱ T = {T:.6f}  |  {days_remaining} days to expiry")

            for strike in strikes:
                for opt_type in ["CE", "PE"]:
                    symbol = f"NFO:NIFTY{EXPIRY}{strike}{opt_type}"
                    try:
                        quote  = kite.quote(symbol)[symbol]
                        ltp    = quote["last_price"]
                        oi     = quote.get("oi", 0)
                        volume = quote.get("volume", 0)
                        iv     = implied_volatility(ltp, spot, strike, T, r, opt_type)

                        iv_history.setdefault(symbol, []).append(iv)
                        iv_history[symbol] = iv_history[symbol][-20:]
                        update_base(symbol, iv)
                        live_snapshot[symbol] = iv
                        ltp_snapshot[symbol]  = ltp

                        vol_history.setdefault(symbol, []).append(volume)
                        vol_history[symbol] = vol_history[symbol][-20:]

                        oi_data.setdefault(symbol, []).append(oi)
                        oi_data[symbol] = oi_data[symbol][-20:]

                        iv_series.setdefault(symbol, []).append(iv)

                    except Exception as e:
                        print(f"Error fetching {symbol}: {e}")
                        iv_series.setdefault(symbol, []).append(np.nan)

            time.sleep(INTERVAL)

        except Exception as e:
            print("Fetch loop error:", e)
            time.sleep(5)


def ltp_loop():
    while True:
        try:
            if atm_strike_fixed is None or EXPIRY is None:
                time.sleep(2)
                continue

            strikes = [
                atm_strike_fixed + i * STRIKE_GAP
                for i in range(-STRIKE_COUNT, STRIKE_COUNT + 1)
            ]
            symbols = [
                f"NFO:NIFTY{EXPIRY}{strike}{opt_type}"
                for strike in strikes
                for opt_type in ["CE", "PE"]
            ]
            ltp_response = kite.ltp(symbols)
            for symbol, data in ltp_response.items():
                ltp_snapshot[symbol] = data["last_price"]

        except Exception as e:
            print("LTP loop error:", e)

        time.sleep(2)


# ================= HTML TEMPLATE ================= #

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IV Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta http-equiv="refresh" content="60">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #fff; padding: 20px; }
        .container { max-width: 1900px; margin: 0 auto; }
        .header { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 20px; border-radius: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 28px; font-weight: 600; }
        .live-indicator { display: inline-block; width: 10px; height: 10px; background: #4CAF50; border-radius: 50%; animation: pulse 2s infinite; margin-right: 10px; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px; }
        .metric-box { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.2); }
        .metric-label { font-size: 12px; opacity: 0.8; margin-bottom: 5px; }
        .metric-value { font-size: 24px; font-weight: 700; }
        .metric-change { font-size: 14px; margin-top: 5px; }
        .positive { color: #4CAF50; } .negative { color: #f44336; } .neutral { color: #FFC107; }
        .card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px; }
        .card h2 { font-size: 18px; margin-bottom: 15px; color: #ffd700; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; background: rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; }
        th { background: rgba(255,255,255,0.2); padding: 12px 8px; text-align: center; font-weight: 600; position: sticky; top: 0; z-index: 10; }
        td { padding: 10px 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1); }
        tr:hover { background: rgba(255,255,255,0.08); }
        .yes { background: rgba(76,175,80,0.3); font-weight: 600; }
        .no  { background: rgba(255,187,107,0.2); }
        .atm-row { background: rgba(255,215,0,0.25); font-weight: 700; }
        .table-container { max-height: 600px; overflow-y: auto; margin-top: 10px; }
        .chart-container { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-top: 10px; }
        .ltp-cell { font-weight: 600; transition: background 0.3s ease; }
        .ltp-flash { background: rgba(255,215,0,0.4) !important; }
        .event-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }
        .event-no-trade { background: rgba(150,150,150,0.35); color: #cccccc; border: 1px solid rgba(180,180,180,0.4); }
        .event-watch    { background: rgba(255,193,7,0.3);   color: #FFD54F;  border: 1px solid rgba(255,193,7,0.5); }
        .event-entry    { background: rgba(76,175,80,0.35);  color: #81C784;  border: 1px solid rgba(76,175,80,0.6); }
        .event-exit     { background: rgba(244,67,54,0.35);  color: #EF9A9A;  border: 1px solid rgba(244,67,54,0.6); }
        .event-na       { background: rgba(100,100,100,0.2); color: #888888;  border: 1px solid rgba(100,100,100,0.3); }
        .vol-high { color: #81C784; font-weight: 700; }
        .vol-low  { color: #EF9A9A; font-weight: 700; }
        .vol-na   { color: #888888; }
        .group-header th { background: rgba(255,255,255,0.12); font-size: 11px; color: #FFD700; padding: 6px 8px; letter-spacing: 0.5px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1><span class="live-indicator"></span>IV Dashboard - Live</h1>
            <div style="font-size:14px;opacity:0.8;margin-top:5px;">
                Nifty Options | Expiry: {{ expiry }} | Last Update: {{ last_update }}
            </div>
        </div>
    </div>
    <div class="metrics">
        <div class="metric-box"><div class="metric-label">Spot Price</div><div class="metric-value">{{ spot }}</div><div class="metric-change positive">LIVE</div></div>
        <div class="metric-box"><div class="metric-label">ATM Strike</div><div class="metric-value">{{ atm }}</div><div class="metric-change neutral">FIXED</div></div>
        <div class="metric-box"><div class="metric-label">Days to Expiry</div><div class="metric-value">{{ days_remaining }}</div><div class="metric-change neutral">Expiry: {{ expiry }}</div></div>
        <div class="metric-box"><div class="metric-label">Data Points</div><div class="metric-value">{{ data_points }}</div><div class="metric-change">Collected</div></div>
        <div class="metric-box"><div class="metric-label">Strike Range</div><div class="metric-value">±{{ strike_count }}</div><div class="metric-change">ATM Coverage</div></div>
    </div>
    <div class="card">
        <h2>📈 Full Day IV Trend - Multi Strike Analysis</h2>
        <div class="chart-container"><div id="iv_chart" style="height:500px;"></div></div>
    </div>
    <div class="card">
        <h2>📊 Complete Strike Analysis Table (ATM ±{{ strike_count }})</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr class="group-header">
                        <th rowspan="2">Strike</th><th rowspan="2">Type</th><th rowspan="2">LTP</th>
                        <th colspan="3" style="border-bottom:1px solid rgba(255,255,255,0.2);">1 Min</th>
                        <th colspan="3" style="border-bottom:1px solid rgba(255,255,255,0.2);">3 Min</th>
                        <th colspan="3" style="border-bottom:1px solid rgba(255,255,255,0.2);">5 Min</th>
                    </tr>
                    <tr class="group-header">
                        <th>IV</th><th>Vol (x SMA9)</th><th>Event</th>
                        <th>IV</th><th>Vol (x SMA9)</th><th>Event</th>
                        <th>IV</th><th>Vol (x SMA9)</th><th>Event</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in rows %}
                    <tr class="{{ row.cls }} {{ 'atm-row' if row.is_atm else '' }}">
                        <td><strong>{{ row.strike }}</strong></td>
                        <td>{{ row.type }}</td>
                        <td class="ltp-cell" data-symbol="{{ row.strike }}{{ row.type }}">{{ row.ltp }}</td>
                        <td>{{ row.iv1 }}%</td><td class="{{ row.vol1_cls }}">{{ row.vol1 }}</td><td><span class="event-badge event-{{ row.event1_cls }}">{{ row.event1 }}</span></td>
                        <td>{{ row.iv3 }}%</td><td class="{{ row.vol3_cls }}">{{ row.vol3 }}</td><td><span class="event-badge event-{{ row.event3_cls }}">{{ row.event3 }}</span></td>
                        <td>{{ row.iv5 }}%</td><td class="{{ row.vol5_cls }}">{{ row.vol5 }}</td><td><span class="event-badge event-{{ row.event5_cls }}">{{ row.event5 }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
<script>
    Plotly.newPlot("iv_chart", {{ datasets | safe }}, {
        title: { text: "Implied Volatility - All Strikes", font: { color: "white" } },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.05)",
        xaxis: { title: "Time", color: "white", gridcolor: "rgba(255,255,255,0.1)" },
        yaxis: { title: "IV (%)", color: "white", gridcolor: "rgba(255,255,255,0.1)", rangemode: "tozero" },
        hovermode: "x unified",
        hoverlabel: { bgcolor: "rgba(0,0,0,0.8)", font: { color: "white", size: 14 }, bordercolor: "rgba(255,255,255,0.3)" },
        legend: { font: { color: "white" } },
        transition: { duration: 500, easing: "cubic-in-out" }
    });

    function fetchLiveLTP() {
        fetch("/ltp").then(r => r.json()).then(data => {
            document.querySelectorAll(".ltp-cell").forEach(cell => {
                const sym = cell.getAttribute("data-symbol");
                if (data[sym] !== undefined && cell.innerText !== data[sym]) {
                    cell.innerText = data[sym];
                    cell.classList.add("ltp-flash");
                    setTimeout(() => cell.classList.remove("ltp-flash"), 400);
                }
            });
        }).catch(err => console.log("LTP fetch error:", err));
    }
    fetchLiveLTP();
    setInterval(fetchLiveLTP, 2000);
</script>
</body>
</html>
"""


# ================= ROUTES ================= #

@app.route("/ltp")
def get_ltp():
    data = {}
    if atm_strike_fixed is None or EXPIRY is None:
        return jsonify(data)
    strikes = [atm_strike_fixed + i * STRIKE_GAP for i in range(-STRIKE_COUNT, STRIKE_COUNT + 1)]
    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            symbol = f"NFO:NIFTY{EXPIRY}{strike}{opt_type}"
            ltp    = ltp_snapshot.get(symbol)
            if ltp is not None:
                data[f"{strike}{opt_type}"] = f"{ltp:,.2f}"
    return jsonify(data)


@app.route("/health")
def health():
    """Health-check endpoint for Render."""
    return jsonify({"status": "ok", "atm": atm_strike_fixed, "expiry": EXPIRY})


@app.route("/")
def dashboard():
    if atm_strike_fixed is None or EXPIRY is None:
        return "<h2 style='color:white;text-align:center;margin-top:50px;'>⏳ Waiting for 9:15 AM market start…</h2>"

    spot_price        = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
    T_now, days_remaining = get_time_to_expiry()

    iv_datasets = []
    for symbol, values in iv_series.items():
        strike_name = symbol.split(EXPIRY)[1]
        iv_datasets.append({
            "x": time_series,
            "y": smooth_iv_series(values),
            "mode": "lines",
            "name": strike_name,
            "line": {"width": 2, "shape": "spline", "smoothing": 1.3}
        })

    strikes = [atm_strike_fixed + i * STRIKE_GAP for i in range(-STRIKE_COUNT, STRIKE_COUNT + 1)]
    rows    = []

    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            symbol   = f"NFO:NIFTY{EXPIRY}{strike}{opt_type}"
            iv       = live_snapshot.get(symbol, np.nan)
            hist     = iv_history.get(symbol, [])
            sma9_val = sma(hist)
            base_val = base_iv.get(symbol, np.nan)

            iv_1m = iv
            iv_3m = get_iv_avg(symbol, 3)
            iv_5m = get_iv_avg(symbol, 5)

            def fmt_vol(v):
                if np.isnan(v): return "N/A", "vol-na"
                return f"{v:.2f}x", "vol-high" if v >= 1.0 else "vol-low"

            vol1_str, vol1_cls = fmt_vol(get_vol_times(symbol, 1))
            vol3_str, vol3_cls = fmt_vol(get_vol_times(symbol, 3))
            vol5_str, vol5_cls = fmt_vol(get_vol_times(symbol, 5))

            event1, event1_cls = classify_event(iv_1m, base_val, sma9_val)
            event3, event3_cls = classify_event(iv_3m, base_val, sma9_val)
            event5, event5_cls = classify_event(iv_5m, base_val, sma9_val)

            entry = "YES" if (not np.isnan(iv) and not np.isnan(sma9_val) and iv > sma9_val and iv > base_val) else "NO"

            rows.append({
                "strike": strike, "type": opt_type,
                "ltp": f"{ltp_snapshot.get(symbol, 0):,.2f}",
                "iv1": f"{iv_1m:.2f}" if not np.isnan(iv_1m) else "N/A",
                "iv3": f"{iv_3m:.2f}" if not np.isnan(iv_3m) else "N/A",
                "iv5": f"{iv_5m:.2f}" if not np.isnan(iv_5m) else "N/A",
                "vol1": vol1_str, "vol1_cls": vol1_cls,
                "vol3": vol3_str, "vol3_cls": vol3_cls,
                "vol5": vol5_str, "vol5_cls": vol5_cls,
                "event1": event1, "event1_cls": event1_cls,
                "event3": event3, "event3_cls": event3_cls,
                "event5": event5, "event5_cls": event5_cls,
                "cls": "yes" if entry == "YES" else "no",
                "is_atm": strike == atm_strike_fixed,
            })

    return render_template_string(
        HTML, rows=rows,
        times=json.dumps(time_series),
        datasets=json.dumps(iv_datasets),
        expiry=EXPIRY,
        last_update=datetime.now().strftime("%H:%M:%S"),
        spot=f"{spot_price:,.2f}",
        atm=atm_strike_fixed,
        data_points=len(time_series),
        strike_count=STRIKE_COUNT,
        days_remaining=days_remaining,
    )


# ================= ENTRYPOINT ================= #

# Start background threads at import time so gunicorn workers pick them up
_fetch_thread = threading.Thread(target=fetch_loop, daemon=True)
_ltp_thread   = threading.Thread(target=ltp_loop,   daemon=True)
_fetch_thread.start()
_ltp_thread.start()

if __name__ == "__main__":
    print("✅ Starting IV Dashboard on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
