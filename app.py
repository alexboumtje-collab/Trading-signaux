"""
📱 Signaux Trading – Application Streamlit mobile-first
Stratégie : confluence d'indicateurs (EMA, RSI, MACD, Supertrend, Bollinger)
⚠️ Outil pédagogique : ne constitue pas un conseil en investissement.
"""

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    import ccxt
except ImportError:  # l'app reste utilisable avec Yahoo Finance seul
    ccxt = None
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# CONFIG PAGE + CSS MOBILE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Signaux Trading",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding: 0.8rem 0.6rem 3rem 0.6rem; max-width: 720px;}
header[data-testid="stHeader"] {height: 2.2rem;}
h1 {font-size: 1.4rem !important; margin-bottom: 0.2rem;}
.sig-card {border-radius: 18px; padding: 18px 12px; text-align: center; color: #fff;
           margin: 8px 0 10px 0; box-shadow: 0 4px 14px rgba(0,0,0,.35);}
.sig-buy  {background: linear-gradient(135deg,#0f9d58,#0b6e3d);}
.sig-sell {background: linear-gradient(135deg,#e53935,#8e1b1b);}
.sig-wait {background: linear-gradient(135deg,#546e7a,#2f3e46);}
.sig-title {font-size: 2.6rem; font-weight: 800; letter-spacing: 2px; line-height: 1.1;}
.sig-sub   {font-size: 1rem; opacity: .92; margin-top: 4px;}
.grid {display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;}
.cell {background: #1c2128; border-radius: 12px; padding: 10px 12px; border-left: 5px solid #607d8b;}
.cell .lbl {font-size: .75rem; color: #9aa4af; text-transform: uppercase;}
.cell .val {font-size: 1.15rem; font-weight: 700; color: #fff;}
.c-entry {border-left-color:#42a5f5;} .c-sl {border-left-color:#ef5350;}
.c-tp {border-left-color:#26a69a;}   .c-rr {border-left-color:#ffca28;}
.why {background:#161b22; border-radius:12px; padding:10px 14px; font-size:.92rem; line-height:1.45;}
.score {display:flex; gap:8px; margin-top:6px;}
.score div {flex:1; background:#1c2128; border-radius:10px; padding:6px; text-align:center; font-size:.85rem;}
.warn {font-size:.72rem; color:#8b949e; margin-top:14px;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# PARAMÈTRES
# ----------------------------------------------------------------------------
PRESETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Or (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq (^IXIC)": "^IXIC",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Autre (saisie libre)": None,
}

# interval yfinance, période, rééchantillonnage éventuel
TIMEFRAMES = {
    "5 min": ("5m", "10d", None),
    "15 min": ("15m", "30d", None),
    "1 heure": ("1h", "180d", None),
    "4 heures": ("1h", "360d", "4h"),
    "1 jour": ("1d", "3y", None),
}

# profil -> (confluence min, multiplicateur ATR du SL, RR TP1, RR TP2)
RISK_PROFILES = {
    "Prudent": (4, 1.5, 2.0, 3.0),
    "Équilibré": (3, 1.2, 2.0, 3.0),
    "Agressif": (3, 1.0, 2.0, 4.0),
}

st.title("📈 Signaux Trading")

# unité de temps supérieure utilisée pour le filtre de tendance
HTF_RULE = {"5 min": "1h", "15 min": "1h", "1 heure": "4h", "4 heures": "1D", "1 jour": "1W"}
CCXT_TF = {"5 min": "5m", "15 min": "15m", "1 heure": "1h", "4 heures": "4h", "1 jour": "1d"}
CCXT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
              "Autre (saisie libre)"]
CCXT_EXCHANGES = ["kraken", "kucoin", "coinbase", "okx", "bybit", "binance"]

with st.expander("⚙️ Paramètres", expanded=True):
    source = st.radio("Source de données", ["Yahoo Finance", "Crypto (ccxt)"], horizontal=True)
    if source == "Yahoo Finance":
        choice = st.selectbox("Actif", list(PRESETS.keys()))
        if PRESETS[choice] is None:
            ticker = st.text_input("Ticker Yahoo Finance", value="NVDA").strip().upper()
        else:
            ticker = PRESETS[choice]
    else:
        exchange_id = st.selectbox(
            "Exchange", CCXT_EXCHANGES,
            help="Binance et Bybit bloquent les serveurs américains (dont Streamlit Cloud) : "
                 "préférez Kraken ou KuCoin.")
        pair = st.selectbox("Paire", CCXT_PAIRS)
        if pair.startswith("Autre"):
            ticker = st.text_input("Paire (ex : LTC/USDT)", value="LTC/USDT").strip().upper()
        else:
            ticker = pair
    c1, c2 = st.columns(2)
    tf_label = c1.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=2)
    risk_label = c2.selectbox("Risque", list(RISK_PROFILES.keys()), index=1)
    sl_method = st.radio(
        "Stop-Loss basé sur", ["Swing Low/High", "ATR"], horizontal=True
    )
    min_stop = st.slider("Stop minimum (× ATR)", 0.5, 4.0, 1.5, step=0.25,
                         help="Distance minimale entre l'entrée et le Stop-Loss. Un stop trop "
                              "serré se fait toucher par le bruit et les frais l'écrasent.")
    use_htf = st.checkbox("Filtre de tendance (unité de temps supérieure)", value=True,
                          help="N'autorise un achat que si la tendance de fond est haussière, "
                               "une vente que si elle est baissière.")
    n_bars = st.slider("Bougies affichées", 60, 300, 120, step=10)
    c3, c4 = st.columns(2)
    capital = c3.number_input("Capital ($)", min_value=0.0, value=1000.0, step=100.0)
    risk_pct = c4.number_input("Risque / trade (%)", 0.1, 10.0, 1.0, step=0.1)

if st.button("🔄 Actualiser les données", width="stretch"):
    st.cache_data.clear()

min_conf, sl_mult, rr1, rr2 = RISK_PROFILES[risk_label]

# ----------------------------------------------------------------------------
# DONNÉES
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_data(symbol: str, interval: str, period: str, resample: str | None):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if resample:
        df = (
            df.resample(resample)
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna()
        )
    return df


@st.cache_data(ttl=30, show_spinner=False)
def load_data_ccxt(exchange_name: str, symbol: str, timeframe: str):
    if ccxt is None:
        raise RuntimeError("Le paquet ccxt n'est pas installé.")
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=700)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df.dropna()


# ----------------------------------------------------------------------------
# INDICATEURS
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=10):
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend(df, period=10, factor=3.0):
    a = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = (hl2 + factor * a).to_numpy()
    lower = (hl2 - factor * a).to_numpy()
    close = df["Close"].to_numpy()
    n = len(df)
    fu, fl = upper.copy(), lower.copy()
    direction = np.ones(n)
    line = np.full(n, np.nan)
    valid = np.where(~np.isnan(upper))[0]
    if len(valid) == 0:
        return pd.Series(line, df.index), pd.Series(direction, df.index)
    s = valid[0]
    line[s] = lower[s]
    for i in range(s + 1, n):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.Series(line, df.index), pd.Series(direction, df.index)


def add_indicators(df, htf_rule):
    df = df.copy()
    c = df["Close"]
    df["EMA20"], df["EMA50"], df["EMA200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["RSI"] = rsi(c, 14)
    fast, slow = ema(c, 12), ema(c, 26)
    df["MACD"] = fast - slow
    df["MACD_SIG"] = ema(df["MACD"], 9)
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["BB_MID"], df["BB_UP"], df["BB_LOW"] = mid, mid + 2 * sd, mid - 2 * sd
    df["ATR"] = atr(df, 10)
    df["ST"], df["ST_DIR"] = supertrend(df, 10, 3.0)
    # tendance de l'unité supérieure, décalée d'une barre (bougie supérieure déjà clôturée)
    htf = df["Close"].resample(htf_rule).last().dropna()
    f20 = htf.ewm(span=20, adjust=False, min_periods=20).mean()
    f50 = htf.ewm(span=50, adjust=False, min_periods=50).mean()
    trend = pd.Series(np.where(f20 > f50, 1.0, np.where(f20 < f50, -1.0, 0.0)),
                      index=htf.index).shift(1)
    df["HTF"] = trend.reindex(df.index, method="ffill").fillna(0.0)
    return df


# ----------------------------------------------------------------------------
# STRATÉGIE : CONFLUENCE
# ----------------------------------------------------------------------------
def htf_label(v):
    return "haussière ↑" if v == 1 else "baissière ↓" if v == -1 else "neutre →"


def evaluate(df):
    r, p = df.iloc[-1], df.iloc[-2]
    buy = {
        "Tendance haussière (EMA20 > EMA50 et prix > EMA200)":
            r.EMA20 > r.EMA50 and r.Close > r.EMA200,
        "Supertrend orienté à la hausse": r.ST_DIR == 1,
        "MACD au-dessus de sa ligne signal et histogramme croissant":
            r.MACD > r.MACD_SIG and r.MACD_HIST > p.MACD_HIST,
        f"RSI en zone haussière non surachetée ({r.RSI:.0f})": 50 <= r.RSI < 70,
        "Prix au-dessus de la moyenne de Bollinger, sous la bande haute":
            r.BB_MID < r.Close < r.BB_UP,
    }
    sell = {
        "Tendance baissière (EMA20 < EMA50 et prix < EMA200)":
            r.EMA20 < r.EMA50 and r.Close < r.EMA200,
        "Supertrend orienté à la baisse": r.ST_DIR == -1,
        "MACD sous sa ligne signal et histogramme décroissant":
            r.MACD < r.MACD_SIG and r.MACD_HIST < p.MACD_HIST,
        f"RSI en zone baissière non survendue ({r.RSI:.0f})": 30 < r.RSI <= 50,
        "Prix sous la moyenne de Bollinger, au-dessus de la bande basse":
            r.BB_LOW < r.Close < r.BB_MID,
    }
    buy = {k: bool(v) for k, v in buy.items()}
    sell = {k: bool(v) for k, v in sell.items()}
    nb, ns = sum(buy.values()), sum(sell.values())

    cand = None
    if nb >= min_conf and nb > ns:
        cand = "BUY"
    elif ns >= min_conf and ns > nb:
        cand = "SELL"
    blocked = False
    if cand and use_htf and ((cand == "BUY" and r.HTF != 1) or (cand == "SELL" and r.HTF != -1)):
        cand, blocked = None, True
    return cand, buy, sell, nb, ns, blocked


def calc_sl(side, entry, a, swing, use_swing, mult, mstop):
    """Stop-Loss : swing (si cohérent) sinon ATR, jamais plus proche que mstop × ATR."""
    if side == 1:
        sl_sw = swing - 0.1 * a
        sl = sl_sw if (use_swing and 0.5 * a <= entry - sl_sw <= 3.5 * a) else entry - mult * a
        return min(sl, entry - mstop * a)
    sl_sw = swing + 0.1 * a
    sl = sl_sw if (use_swing and 0.5 * a <= sl_sw - entry <= 3.5 * a) else entry + mult * a
    return max(sl, entry + mstop * a)


def compute_levels(df, side):
    r = df.iloc[-1]
    entry, a = float(r.Close), float(r.ATR)
    recent = df.tail(10)
    use_sw = sl_method.startswith("Swing")
    if side == "BUY":
        sl = calc_sl(1, entry, a, float(recent["Low"].min()), use_sw, sl_mult, min_stop)
        risk = entry - sl
        tp1, tp2 = entry + rr1 * risk, entry + rr2 * risk
    else:
        sl = calc_sl(-1, entry, a, float(recent["High"].max()), use_sw, sl_mult, min_stop)
        risk = sl - entry
        tp1, tp2 = entry - rr1 * risk, entry - rr2 * risk
    return entry, sl, tp1, tp2, risk


def fmt(x):
    ax = abs(x)
    d = 5 if ax < 10 else 4 if ax < 100 else 2
    return f"{x:,.{d}f}"


# ----------------------------------------------------------------------------
# GRAPHIQUE (optimisé mobile)
# ----------------------------------------------------------------------------
def build_chart(df, n, levels=None):
    d = df.tail(n)
    fmt_x = "%d/%m/%y" if tf_label == "1 jour" else "%d/%m %H:%M"
    x = d.index.strftime(fmt_x)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.56, 0.2, 0.24],
    )
    fig.add_trace(go.Candlestick(
        x=x, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Prix",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False), row=1, col=1)

    for col, color in [("EMA20", "#ffca28"), ("EMA50", "#42a5f5"), ("EMA200", "#ab47bc")]:
        fig.add_trace(go.Scatter(x=x, y=d[col], name=col, mode="lines",
                                 line=dict(color=color, width=1.3)), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=d.BB_UP, mode="lines", name="BB",
                             line=dict(color="rgba(160,160,160,.5)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.BB_LOW, mode="lines", showlegend=False,
                             line=dict(color="rgba(160,160,160,.5)", width=1),
                             fill="tonexty", fillcolor="rgba(160,160,160,.07)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=d.ST.where(d.ST_DIR == 1), mode="lines", name="Supertrend",
                             line=dict(color="#00e676", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.ST.where(d.ST_DIR == -1), mode="lines", showlegend=False,
                             line=dict(color="#ff5252", width=2)), row=1, col=1)

    if levels:
        entry, sl, tp1, tp2 = levels
        for y, txt, col in [(entry, "Entrée", "#42a5f5"), (sl, "SL", "#ef5350"),
                            (tp1, "TP1", "#26a69a"), (tp2, "TP2", "#00e676")]:
            fig.add_hline(y=y, line_dash="dash", line_color=col, line_width=1.2,
                          annotation_text=txt, annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=col,
                          row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=x, y=d.RSI, mode="lines", name="RSI 14",
                             line=dict(color="#ffa726", width=1.5)), row=2, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)

    # MACD
    colors = np.where(d.MACD_HIST >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=d.MACD_HIST, marker_color=colors, name="Histo",
                         showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.MACD, mode="lines", name="MACD",
                             line=dict(color="#42a5f5", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d.MACD_SIG, mode="lines", name="Signal",
                             line=dict(color="#ffca28", width=1.3)), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=780, dragmode="pan",
        margin=dict(l=0, r=0, t=8, b=0), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(type="category", nticks=4, tickangle=0, showgrid=False,
                     rangeslider_visible=False)
    fig.update_yaxes(side="right", showgrid=True, gridcolor="rgba(255,255,255,.06)",
                     tickfont=dict(size=10))
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ----------------------------------------------------------------------------
# EXÉCUTION
# ----------------------------------------------------------------------------
interval, period, resample = TIMEFRAMES[tf_label]
with st.spinner("Chargement des données…"):
    try:
        if source == "Yahoo Finance":
            raw = load_data(ticker, interval, period, resample)
        else:
            raw = load_data_ccxt(exchange_id, ticker, CCXT_TF[tf_label])
    except Exception as e:  # réseau, quota Yahoo, ticker invalide…
        raw = pd.DataFrame()
        st.error(f"Erreur de récupération : {e}")

if raw.empty or len(raw) < 60:
    st.warning("Données insuffisantes ou symbole introuvable. Vérifiez le format "
               "(Yahoo : BTC-USD · ccxt : BTC/USDT), changez d'exchange ou réessayez plus tard.")
    st.stop()

df = add_indicators(raw, HTF_RULE[tf_label]).dropna(subset=["EMA50", "RSI", "BB_UP", "ATR", "MACD_SIG"])
if len(df) < 5:
    st.warning("Pas assez de bougies pour calculer les indicateurs.")
    st.stop()

last = df.iloc[-1]
side, buy_c, sell_c, nb, ns, blocked = evaluate(df)
chg = (df.Close.iloc[-1] / df.Close.iloc[-2] - 1) * 100
st.caption(
    f"**{ticker}** · {tf_label} · Dernier prix : **{fmt(last.Close)}** "
    f"({chg:+.2f}%) · {df.index[-1].strftime('%d/%m %H:%M')}"
)

levels = None
if side:
    entry, sl, tp1, tp2, risk = compute_levels(df, side)
    levels = (entry, sl, tp1, tp2)
    is_buy = side == "BUY"
    st.markdown(
        f"""<div class="sig-card {'sig-buy' if is_buy else 'sig-sell'}">
<div class="sig-title">{'🟢 ACHAT' if is_buy else '🔴 VENTE'}</div>
<div class="sig-sub">Confluence : {nb if is_buy else ns}/5 · Profil {risk_label}<br>Tendance supérieure : {htf_label(last.HTF)}</div>
</div>""",
        unsafe_allow_html=True,
    )
    size_txt = "—"
    if risk > 0 and capital > 0:
        units = capital * risk_pct / 100 / risk
        size_txt = f"{units:,.4f} unités"
    st.markdown(
        f"""<div class="grid">
<div class="cell c-entry"><div class="lbl">Entrée conseillée</div><div class="val">{fmt(entry)}</div></div>
<div class="cell c-sl"><div class="lbl">Stop-Loss</div><div class="val">{fmt(sl)}</div></div>
<div class="cell c-tp"><div class="lbl">Take-Profit 1 (1:{rr1:g})</div><div class="val">{fmt(tp1)}</div></div>
<div class="cell c-tp"><div class="lbl">Take-Profit 2 (1:{rr2:g})</div><div class="val">{fmt(tp2)}</div></div>
<div class="cell c-rr"><div class="lbl">Risque / unité</div><div class="val">{fmt(risk)}</div></div>
<div class="cell c-rr"><div class="lbl">Taille ({risk_pct:g}% du capital)</div><div class="val">{size_txt}</div></div>
</div>""",
        unsafe_allow_html=True,
    )
    valid = [k for k, v in (buy_c if is_buy else sell_c).items() if v]
    why = "<br>".join(f"✅ {v}" for v in valid)
    sl_txt = f"{risk / last.ATR:.1f}× l'ATR de l'entrée"
    st.markdown(
        f"""<div class="why"><b>Pourquoi ce signal ?</b><br>{why}<br>
<b>Gestion :</b> SL placé à {sl_txt} ; TP1 = {rr1:g}× le risque, TP2 = {rr2:g}×.
Sécurisez une partie au TP1 et passez le SL à l'entrée.</div>""",
        unsafe_allow_html=True,
    )
else:
    wait_msg = ("Signal bloqué : il va contre la tendance de l'unité supérieure" if blocked
                else f"Aucune confluence suffisante (minimum {min_conf}/5 requis)")
    st.markdown(
        f"""<div class="sig-card sig-wait">
<div class="sig-title">⏸ ATTENTE</div>
<div class="sig-sub">{wait_msg}</div>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="score"><div>🟢 Achat<br><b>{nb}/5</b></div>
<div>🔴 Vente<br><b>{ns}/5</b></div></div>""",
        unsafe_allow_html=True,
    )

with st.expander("🔍 Détail des conditions"):
    st.write(f"Tendance de l'unité supérieure : **{htf_label(last.HTF)}**"
             + (" (filtre actif)" if use_htf else " (filtre désactivé)"))
    st.markdown("**Achat**")
    for k, v in buy_c.items():
        st.write(("✅ " if v else "⬜ ") + k)
    st.markdown("**Vente**")
    for k, v in sell_c.items():
        st.write(("✅ " if v else "⬜ ") + k)

st.plotly_chart(
    build_chart(df, n_bars, levels),
    width="stretch",
    config={"scrollZoom": True, "displayModeBar": False, "responsive": True,
            "doubleClick": "reset"},
)

# ----------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------
def score_series(d):
    buy = (
        ((d.EMA20 > d.EMA50) & (d.Close > d.EMA200)).astype(int)
        + (d.ST_DIR == 1).astype(int)
        + ((d.MACD > d.MACD_SIG) & (d.MACD_HIST > d.MACD_HIST.shift())).astype(int)
        + ((d.RSI >= 50) & (d.RSI < 70)).astype(int)
        + ((d.Close > d.BB_MID) & (d.Close < d.BB_UP)).astype(int)
    )
    sell = (
        ((d.EMA20 < d.EMA50) & (d.Close < d.EMA200)).astype(int)
        + (d.ST_DIR == -1).astype(int)
        + ((d.MACD < d.MACD_SIG) & (d.MACD_HIST < d.MACD_HIST.shift())).astype(int)
        + ((d.RSI > 30) & (d.RSI <= 50)).astype(int)
        + ((d.Close < d.BB_MID) & (d.Close > d.BB_LOW)).astype(int)
    )
    return buy.to_numpy(), sell.to_numpy()


def prepare_bt(d):
    b, s = score_series(d)
    return {
        "b": b, "s": s, "htf": d.HTF.to_numpy(), "index": d.index,
        "close": d.Close.to_numpy(), "high": d.High.to_numpy(), "low": d.Low.to_numpy(),
        "atr": d.ATR.to_numpy(),
        "sw_low": d.Low.rolling(10).min().to_numpy(),
        "sw_high": d.High.rolling(10).max().to_numpy(),
    }


def run_backtest(P, start, end, conf, mult, use_swing, mstop, htf_on, target_rr, max_hold, fee_pct):
    """Un seul trade à la fois. Sortie au SL, à la cible, ou à la clôture après max_hold.
    Si SL et cible sont touchés dans la même bougie, le SL est retenu (hypothèse prudente)."""
    b, s, htf = P["b"], P["s"], P["htf"]
    close, high, low, a = P["close"], P["high"], P["low"], P["atr"]
    trades = []
    i = start
    while i < end - 1:
        side = 1 if (b[i] >= conf and b[i] > s[i]) else -1 if (s[i] >= conf and s[i] > b[i]) else 0
        if side != 0 and htf_on and htf[i] != side:
            side = 0
        if side == 0 or np.isnan(a[i]):
            i += 1
            continue
        entry = close[i]
        sl = calc_sl(side, entry, a[i], P["sw_low"][i] if side == 1 else P["sw_high"][i],
                     use_swing, mult, mstop)
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1
            continue
        tp = entry + side * target_rr * risk
        last_bar = min(i + max_hold, end - 1)
        exit_j, r_mult, why = last_bar, side * (close[last_bar] - entry) / risk, "Temps"
        for j in range(i + 1, last_bar + 1):
            hit_sl = low[j] <= sl if side == 1 else high[j] >= sl
            hit_tp = high[j] >= tp if side == 1 else low[j] <= tp
            if hit_sl:
                exit_j, r_mult, why = j, -1.0, "SL"
                break
            if hit_tp:
                exit_j, r_mult, why = j, float(target_rr), "TP"
                break
        r_net = r_mult - (fee_pct / 100 * entry / risk)
        trades.append({"Date": P["index"][i], "Sens": "ACHAT" if side == 1 else "VENTE",
                       "Sortie": why, "R net": round(float(r_net), 2)})
        i = exit_j + 1
    return pd.DataFrame(trades)


def summarize(tr):
    if tr is None or tr.empty:
        return {"n": 0, "win": 0.0, "exp": 0.0, "pf": 0.0, "total": 0.0, "dd": 0.0}
    r = tr["R net"].to_numpy()
    gains, pertes = r[r > 0].sum(), -r[r <= 0].sum()
    eq = np.concatenate([[0.0], np.cumsum(r)])
    return {"n": len(r), "win": (r > 0).mean() * 100, "exp": r.mean(),
            "pf": gains / pertes if pertes > 0 else float("inf"),
            "total": r.sum(), "dd": (np.maximum.accumulate(eq) - eq).max()}


def fmt_pf(x):
    return "∞" if x == float("inf") else f"{x:.2f}"


with st.expander("🧪 Backtest de la stratégie"):
    st.caption("Rejoue les règles de confluence sur l'historique chargé ci-dessus, avec "
               "vos réglages (profil, Stop-Loss, stop minimum, filtre de tendance). "
               "Résultats en multiples de R (1R = risque pris sur le trade).")
    bc1, bc2 = st.columns(2)
    target_name = bc1.selectbox("Objectif de sortie", ["TP1", "TP2"])
    fee = bc2.number_input("Frais A/R (%)", 0.0, 2.0, 0.1, step=0.05)
    hold = st.slider("Durée max d'un trade (bougies)", 10, 200, 50)
    if st.button("▶️ Lancer le backtest", width="stretch"):
        tgt = rr1 if target_name == "TP1" else rr2
        P = prepare_bt(df)
        start = min(100, len(df) // 3)
        tr = run_backtest(P, start, len(df), min_conf, sl_mult, sl_method.startswith("Swing"),
                          min_stop, use_htf, tgt, hold, fee)
        if tr.empty:
            st.info("Aucun trade déclenché sur cette période avec ces réglages.")
        else:
            m = summarize(tr)
            equity = tr["R net"].cumsum()
            st.markdown(
                f"""<div class="grid">
<div class="cell c-entry"><div class="lbl">Trades</div><div class="val">{m['n']}</div></div>
<div class="cell c-tp"><div class="lbl">Taux de réussite</div><div class="val">{m['win']:.0f}%</div></div>
<div class="cell c-rr"><div class="lbl">Espérance / trade</div><div class="val">{m['exp']:+.2f} R</div></div>
<div class="cell c-rr"><div class="lbl">Profit factor</div><div class="val">{fmt_pf(m['pf'])}</div></div>
<div class="cell c-tp"><div class="lbl">Résultat total</div><div class="val">{m['total']:+.1f} R</div></div>
<div class="cell c-sl"><div class="lbl">Drawdown max</div><div class="val">-{m['dd']:.1f} R</div></div>
</div>""",
                unsafe_allow_html=True,
            )
            st.caption(f"Pour 1:{tgt:g}, le seuil de rentabilité est ≈ {100 / (1 + tgt):.0f}% de réussite "
                       f"(avant frais). Période testée : {df.index[start].strftime('%d/%m/%y')} "
                       f"→ {df.index[-1].strftime('%d/%m/%y')}.")
            fig_eq = go.Figure(go.Scatter(x=list(range(1, len(equity) + 1)), y=equity,
                                          mode="lines", line=dict(color="#42a5f5", width=2),
                                          fill="tozeroy", fillcolor="rgba(66,165,245,.12)"))
            fig_eq.update_layout(template="plotly_dark", height=240, dragmode="pan",
                                 margin=dict(l=0, r=0, t=24, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                 title=dict(text="Courbe de capital (en R)", font=dict(size=13)),
                                 xaxis_title="N° du trade")
            fig_eq.update_yaxes(side="right")
            st.plotly_chart(fig_eq, width="stretch",
                            config={"scrollZoom": True, "displayModeBar": False})
            show = tr.tail(15).copy()
            show["Date"] = show["Date"].dt.strftime("%d/%m %H:%M")
            st.dataframe(show.iloc[::-1], width="stretch", hide_index=True)
            st.caption("⚠️ Test sur historique, sans slippage : un bon résultat passé "
                       "ne garantit rien. Peu de trades = résultat peu fiable.")

with st.expander("🔬 Trouver le meilleur réglage"):
    st.caption("Teste 64 combinaisons sur les **70 % premiers** de l'historique, puis vérifie "
               "les 5 meilleures sur les **30 % restants**, que la recherche n'a jamais vus. "
               "C'est le test anti-« sur-ajustement » : un réglage n'est crédible que s'il "
               "gagne aussi sur ces données-là.")
    min_tr = st.slider("Trades minimum (partie 70 %)", 10, 60, 20)
    if st.button("🚀 Lancer la recherche", width="stretch"):
        P = prepare_bt(df)
        n = len(df)
        start = min(100, n // 3)
        cut = start + int((n - start) * 0.7)
        rows = []
        for conf, use_sw, ms, rr, htf_on in itertools.product(
                [3, 4], [True, False], [1.0, 1.5, 2.0, 3.0], [2.0, 3.0], [True, False]):
            trn = summarize(run_backtest(P, start, cut, conf, sl_mult, use_sw, ms, htf_on,
                                         rr, hold, fee))
            if trn["n"] >= min_tr:
                rows.append((conf, use_sw, ms, rr, htf_on, trn))
        if not rows:
            st.info("Pas assez de trades pour comparer : baissez le minimum de trades ou "
                    "choisissez une unité de temps plus courte.")
        else:
            rows.sort(key=lambda x: x[5]["exp"], reverse=True)
            ok = 0
            for k, (conf, use_sw, ms, rr, htf_on, trn) in enumerate(rows[:5], 1):
                tst = summarize(run_backtest(P, cut, n, conf, sl_mult, use_sw, ms, htf_on,
                                             rr, hold, fee))
                if tst["n"] < 8:
                    verdict = "❓ Trop peu de trades en test"
                elif tst["exp"] > 0 and tst["pf"] >= 1.2:
                    verdict, ok = "✅ Résiste au test", ok + 1
                else:
                    verdict = "❌ Ne résiste pas au test"
                st.markdown(
                    f"""<div class="why"><b>#{k} · {verdict}</b><br>
Profil {'Prudent' if conf == 4 else 'Équilibré'} · SL {'Swing' if use_sw else 'ATR'} · Stop min {ms:g}×ATR · Cible 1:{rr:g} · Filtre tendance {'oui' if htf_on else 'non'}<br>
Apprentissage : {trn['n']} trades · {trn['exp']:+.2f} R/trade<br>
<b>Test (jamais vu)</b> : {tst['n']} trades · {tst['exp']:+.2f} R/trade · PF {fmt_pf(tst['pf'])}</div>""",
                    unsafe_allow_html=True,
                )
                st.write("")
            if ok == 0:
                st.warning("Aucun réglage ne tient sur les données non vues. Ne tradez pas cette "
                           "configuration : essayez un autre actif ou une autre unité de temps.")
            else:
                st.success(f"{ok} réglage(s) sur 5 résistent. Recopiez-les dans ⚙️ Paramètres "
                           "(profil, Stop-Loss, stop minimum, filtre), relancez le backtest "
                           "complet, puis confirmez en démo. Un seul test réussi ne prouve rien.")

st.markdown(
    '<div class="warn">⚠️ Outil éducatif, sans garantie de résultat. Aucun signal '
    "n'est une recommandation d'investissement. Données Yahoo Finance, parfois "
    "différées. Testez toujours sur compte démo.</div>",
    unsafe_allow_html=True,
)
