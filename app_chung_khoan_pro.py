import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (V4.0 Pro - Chuẩn KRX)", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Xu Hướng (ADX + SuperTrend + Stochastic + Phân Kỳ) & R:R (V4.0 Pro)")
st.markdown("Nguyên tắc kỷ luật: **Chỉ kích hoạt MUA khi SuperTrend Xanh & ADX ≥ 22**, không dính **Phân kỳ âm**, ưu tiên **Dòng tiền xác nhận (OBV/MFI)**, **Điểm mua sát nền** và **R:R ≥ 1.5**.")

if st.button("🔄 Làm mới dữ liệu trực tiếp (Xóa Cache & Cập nhật giá mới nhất)"):
    st.cache_data.clear()
    st.rerun()

if "nhat_ky_lenh" not in st.session_state:
    st.session_state.nhat_ky_lenh = [
        {"Ngày vào": "2026-11-25", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ (đ)": 13300, "Chốt lời (đ)": 14650, "Giá đóng tay (đ)": 0.0, "Ngày đóng": "", "Ghi chú": "Mẫu Bùng nổ Vol"}
    ]

# --- HÀM CHUẨN HÓA BƯỚC GIÁ SÀN VIỆT NAM (KRX) ---
def round_vn_tick(price: float, exchange: str = "HOSE") -> float:
    if price <= 0:
        return 0.0
    ex = exchange.upper()
    if ex in ["HNX", "UPCOM"]:
        tick = 100.0
    else:  # HOSE
        if price < 10000:
            tick = 10.0
        elif price < 50000:
            tick = 50.0
        else:
            tick = 100.0
    return round(round(price / tick) * tick, 0)

# --- TẢI DỮ LIỆU GIÁ LỊCH SỬ (ĐÃ SỬA ĐỦ ENDPOINT API 3 SÀN) ---
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
    now_vn = datetime.utcnow() + timedelta(hours=7)
    today_str = now_vn.strftime('%Y-%m-%d')
    end_ts = int(now_vn.timestamp())
    start_ts = int((now_vn - timedelta(days=days)).timestamp())
    sym = symbol.upper().strip()
    df = pd.DataFrame()

    # 1. Nguồn 1: Vietcap (VCI) - Đầy đủ 100% HOSE, HNX, UPCOM, VNINDEX
    try:
        url_vci = "https://vietcap.com.vn"
        payload_vci = {
            "timeFrame": "ONE_DAY",
            "symbols": [sym],
            "to": end_ts,
            "countBack": min(int(days * 0.75), 1000)
        }
        headers_vci = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Content-Type': 'application/json',
            'Origin': 'https://vietcap.com.vn',
            'Referer': 'https://vietcap.com.vn/'
        }
        res_vci = requests.post(url_vci, json=payload_vci, headers=headers_vci, timeout=8).json()
        if isinstance(res_vci, list) and len(res_vci) > 0:
            d = res_vci[0]
            if isinstance(d, dict) and d.get('t') and len(d.get('t')) >= 30:
                df = pd.DataFrame({
                    'time': pd.to_datetime(pd.to_numeric(d.get('t')), unit='s').strftime('%Y-%m-%d'),
                    'open': pd.to_numeric(d.get('o'), errors='coerce'),
                    'high': pd.to_numeric(d.get('h'), errors='coerce'),
                    'low': pd.to_numeric(d.get('l'), errors='coerce'),
                    'close': pd.to_numeric(d.get('c'), errors='coerce'),
                    'volume': pd.to_numeric(d.get('v'), errors='coerce')
                })
                if sym != "VNINDEX":
                    for col in ['open', 'high', 'low', 'close']:
                        if df[col].iloc[-1] < 1000:
                            df[col] = df[col] * 1000
                df = df.dropna().sort_values('time').reset_index(drop=True)
    except Exception:
        pass

    # 2. Nguồn 2: DNSE / EntradeX API
    if df.empty:
        try:
            url_dnse = "https://entrade.com.vn"
            params_dnse = {"symbol": sym, "resolution": "1D", "from": start_ts, "to": end_ts}
            res_dnse = requests.get(url_dnse, params=params_dnse, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8).json()
            if isinstance(res_dnse, dict) and res_dnse.get('t') and len(res_dnse.get('t')) >= 30:
                df = pd.DataFrame({
                    'time': pd.to_datetime(pd.to_numeric(res_dnse.get('t')), unit='s').strftime('%Y-%m-%d'),
                    'open': pd.to_numeric(res_dnse.get('o'), errors='coerce'),
                    'high': pd.to_numeric(res_dnse.get('h'), errors='coerce'),
                    'low': pd.to_numeric(res_dnse.get('l'), errors='coerce'),
                    'close': pd.to_numeric(res_dnse.get('c'), errors='coerce'),
                    'volume': pd.to_numeric(res_dnse.get('v'), errors='coerce')
                })
                if sym != "VNINDEX":
                    for col in ['open', 'high', 'low', 'close']:
                        if df[col].iloc[-1] < 1000:
                            df[col] = df[col] * 1000
                df = df.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass

    # 3. Nguồn 3: Simplize API
    if df.empty and sym != "VNINDEX":
        try:
            url_sim = f"https://simplize.vn{sym}?page=0&size={min(int(days * 0.75), 1000)}"
            res_sim = requests.get(url_sim, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8).json()
            items = res_sim.get('data', [])
            if isinstance(items, list) and len(items) >= 30:
                raw_df = pd.DataFrame(items)
                df = pd.DataFrame({
                    'time': pd.to_datetime(raw_df.get('date'), unit='s').dt.strftime('%Y-%m-%d'),
                    'open': pd.to_numeric(raw_df.get('priceOpen'), errors='coerce'),
                    'high': pd.to_numeric(raw_df.get('priceHigh'), errors='coerce'),
                    'low': pd.to_numeric(raw_df.get('priceLow'), errors='coerce'),
                    'close': pd.to_numeric(raw_df.get('priceClose'), errors='coerce'),
                    'volume': pd.to_numeric(raw_df.get('totalVolume'), errors='coerce')
                })
                for col in ['open', 'high', 'low', 'close']:
                    if df[col].iloc[-1] < 1000:
                        df[col] = df[col] * 1000
                df = df.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass

    # 4. Nguồn 4: Yahoo Finance (.VN - Dự phòng sàn HOSE)
    if df.empty:
        try:
            start_date = (now_vn - timedelta(days=days)).strftime('%Y-%m-%d')
            ticker_vn = "^VNINDEX" if sym == "VNINDEX" else (sym if sym.endswith(".VN") else f"{sym}.VN")
            df_yf = yf.download(ticker_vn, start=start_date, end=today_str, progress=False, auto_adjust=True)
            if df_yf is not None and not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                df_yf = df_yf.reset_index()
                df_yf = df_yf.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
                df_yf.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                df_yf['time'] = pd.to_datetime(df_yf['time']).dt.strftime('%Y-%m-%d')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_yf[col] = pd.to_numeric(df_yf[col], errors='coerce')
                df = df_yf.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass

    return df

# --- LẤY CHỈ SỐ CƠ BẢN (ROE, P/E, P/B, D/E) & SÀN NIÊM YẾT ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_ratios(symbol: str) -> dict:
    sym = symbol.upper().strip()
    info = {"roe": None, "pe": None, "pb": None, "de": None, "exchange": "HOSE"}
    try:
        url_tcbs = f"https://tcbs.com.vn{sym}/overview"
        res = requests.get(url_tcbs, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        if isinstance(res, dict):
            ex_raw = str(res.get("exchange", "HOSE")).upper()
            if "HNX" in ex_raw:
                info["exchange"] = "HNX"
            elif "UPCOM" in ex_raw:
                info["exchange"] = "UPCOM"
            if res.get("roe") is not None:
                info["roe"] = round(float(res["roe"]) * 100, 1)
            if res.get("pe") is not None:
                info["pe"] = round(float(res["pe"]), 1)
            if res.get("pb") is not None:
                info["pb"] = round(float(res["pb"]), 2)
            if res.get("debtOnEquity") is not None:
                info["de"] = round(float(res["debtOnEquity"]), 2)
    except Exception:
        pass
    return info

# --- TÍNH HỆ SỐ BETA (CAPM) SO VỚI VNINDEX ---
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_stock_beta(df_stock: pd.DataFrame) -> float:
    try:
        df_mkt = get_stock_data("VNINDEX", days=400)
        if df_mkt.empty or len(df_mkt) < 60:
            return 1.0
        merged = pd.merge(df_stock[['time', 'close']], df_mkt[['time', 'close']], on='time', suffixes=('_s', '_m'))
        if len(merged) < 60:
            return 1.0
        ret_s = merged['close_s'].pct_change().dropna()
        ret_m = merged['close_m'].pct_change().dropna()
        cov = np.cov(ret_s, ret_m)[0, 1]
        var_m = np.var(ret_m)
        return round(float(cov / var_m), 2) if var_m > 0 else 1.0
    except Exception:
        return 1.0

@st.cache_data(ttl=600, show_spinner=False)
def get_dynamic_top_tickers(fallback_list):
    try:
        url = "https://tcbs.com.vn"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=6).json()
        items = res.get('data', [])
        tickers = [x['ticker'] for x in items if x.get('ticker') and len(x['ticker']) == 3]
        if len(tickers) >= 30:
            return tickers
    except Exception:
        pass
    return list(fallback_list)

# --- TÍNH CHỈ BÁO KỸ THUẬT + OBV + NHẬN DIỆN PHÂN KỲ RSI ---

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['STD20'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
    df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = (df['volume'] / df['Vol_MA20'].replace(0, np.nan)) * 100
    # OBV (On-Balance Volume) & OBV_MA20 (Dòng tiền tích lũy)
    obv_dir = np.sign(df['close'].diff()).fillna(0)
    df['OBV'] = (obv_dir * df['volume']).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
    # ATR (14) chuẩn Wilder
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    # ADX (14) & +DI / -DI
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_14 = df['ATR'].replace(0, np.nan)
    df['Plus_DI'] = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_14)
    df['Minus_DI'] = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_14)
    di_sum = (df['Plus_DI'] + df['Minus_DI']).replace(0, np.nan)
    dx = 100 * ((df['Plus_DI'] - df['Minus_DI']).abs() / di_sum)
    df['ADX'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    # SuperTrend (10, 3.0)
    st_atr = df['TR'].ewm(alpha=1/10, min_periods=1, adjust=False).mean()
    hl2 = (df['high'] + df['low']) / 2
    basic_ub = (hl2 + 3.0 * st_atr).fillna(df['close']).values
    basic_lb = (hl2 - 3.0 * st_atr).fillna(df['close']).values
    close_vals = df['close'].values
    n = len(df)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    st_dir = np.ones(n, dtype=int)
    for i in range(n):
    if i == 0:
    final_ub[i] = basic_ub[i]
    final_lb[i] = basic_lb[i]
    supertrend[i] = basic_lb[i]
    continue
    final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_vals[i-1] > final_ub[i-1]) else final_ub[i-1]
    final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_vals[i-1] < final_lb[i-1]) else final_lb[i-1]
    if st_dir[i-1] == 1:
    if close_vals[i] < final_lb[i]:
    st_dir[i] = -1
    supertrend[i] = final_ub[i]
    else:
    st_dir[i] = 1
    supertrend[i] = final_lb[i]
    else:
    if close_vals[i] > final_ub[i]:
    st_dir[i] = 1
    supertrend[i] = final_lb[i]
    else:
    st_dir[i] = -1
    supertrend[i] = final_ub[i]
    df['SuperTrend'] = supertrend
    df['ST_Dir'] = st_dir
    # Stochastic Oscillator (14, 3, 3)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    denom_stoch = (high_14 - low_14).replace(0, np.nan)
    fast_k = (100 * ((df['close'] - low_14) / denom_stoch)).fillna(50.0)
    df['Stoch_K'] = fast_k.rolling(window=3, min_periods=1).mean()
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean()
    # Ichimoku Cloud (9, 26, 52)
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['Span_A_Raw'] = (df['Tenkan'] + df['Kijun']) / 2
    df['Span_B_Raw'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
    df['Senkou_A'] = df['Span_A_Raw'].shift(26)
    df['Senkou_B'] = df['Span_B_Raw'].shift(26)
    # Hộp Sideway & Đỉnh/Đáy 20 phiên
    df['High_20'] = df['high'].rolling(window=20).max()
    df['Low_20'] = df['low'].rolling(window=20).min()
    df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20'].replace(0, np.nan)) * 100
    # RSI (14) chuẩn Wilder
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_vals = np.where(
    (avg_gain == 0) & (avg_loss == 0), 50.0,
    np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss.replace(0, np.nan)))))
    )
    df['RSI'] = pd.Series(rsi_vals, index=df.index)
    # MFI (14)
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    pos_mf = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0.0), index=df.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0.0), index=df.index)
    pos_sum = pos_mf.rolling(window=14).sum()
    neg_sum = neg_mf.rolling(window=14).sum()
    mfi_vals = np.where(
    (pos_sum == 0) & (neg_sum == 0), 50.0,
    np.where(neg_sum == 0, 100.0,
    np.where(pos_sum == 0, 0.0, 100.0 - (100.0 / (1.0 + (pos_sum / neg_sum.replace(0, np.nan))))))
    )
    df['MFI'] = pd.Series(mfi_vals, index=df.index)
    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

--- HÀM QUÉT PHÂN KỲ ÂM / DƯƠNG RSI TRONG 15 PHIÊN ---

def detect_rsi_divergence(df_clean: pd.DataFrame, lookback: int = 15):
    if len(df_clean) < lookback + 5:
    return False, False
    sub = df_clean.tail(lookback)
    curr_p = float(sub['close'].iloc[-1])
    curr_rsi = float(sub['RSI'].iloc[-1])
    past = sub.iloc[:-3]
    max_past_p = float(past['close'].max())
    max_past_rsi = float(past['RSI'].max())
    min_past_p = float(past['close'].min())
    min_past_rsi = float(past['RSI'].min())
    # Phân kỳ âm: Giá vượt đỉnh cũ nhưng RSI thấp hơn đỉnh cũ đáng kể và đang ở vùng cao (>60)
    bearish_div = (curr_p >= max_past_p * 0.995) and (curr_rsi < max_past_rsi - 5.0) and (max_past_rsi > 62.0)
    # Phân kỳ dương: Giá thủng/bằng đáy cũ nhưng RSI tạo đáy sau cao hơn đáy trước từ vùng quá bán (<40)
    bullish_div = (curr_p <= min_past_p * 1.005) and (curr_rsi > min_past_rsi + 5.0) and (min_past_rsi < 38.0)
    return bearish_div, bullish_div

--- ĐÁNH GIÁ TÍN HIỆU TỔNG HỢP (TA + FA + KRX + PHÂN KỲ + OBV) ---

def evaluate_stock_signals(df_clean: pd.DataFrame, ai_pct_change: float = 0.0, ai_is_reliable: bool = False, fund_info: dict = None):
    if fund_info is None:
    fund_info = {"roe": None, "pe": None, "pb": None, "de": None, "exchange": "HOSE"}
    exchange = fund_info.get("exchange", "HOSE")
    last = df_clean.iloc[-1]
    prev = df_clean.iloc[-2]
    curr_p = float(last['close'])
    prev_p = float(prev['close'])
    vol_ratio = float(last['Vol_Ratio'])
    atr_val = float(last['ATR'])
    rsi_val = float(last['RSI'])
    mfi_val = float(last['MFI'])
    ma20_val = float(last['MA20'])
    ma50_val = float(last['MA50'])
    kijun_val = float(last['Kijun'])
    bb_upper = float(last['BB_Upper'])
    high_20 = float(last['High_20'])
    low_20 = float(last['Low_20'])
    cloud_top = max(float(last['Senkou_A']), float(last['Senkou_B']))
    cloud_bot = min(float(last['Senkou_A']), float(last['Senkou_B']))
    range_20 = float(last['Range_20_Pct'])
    adx_val = float(last['ADX'])
    plus_di = float(last['Plus_DI'])
    minus_di = float(last['Minus_DI'])
    st_val = float(last['SuperTrend'])
    st_dir = int(last['ST_Dir'])
    stoch_k = float(last['Stoch_K'])
    stoch_d = float(last['Stoch_D'])
    prev_k = float(prev['Stoch_K'])
    prev_d = float(prev['Stoch_D'])
    obv_bull = float(last['OBV']) > float(last['OBV_MA20'])
    bearish_div, bullish_div = detect_rsi_divergence(df_clean)
    # 1. Tính Điểm Mua Hợp Lý, Cắt Lỗ, Chốt Lời theo Bước Giá Chuẩn KRX
    dist_ma20_pct = ((curr_p - ma20_val) / ma20_val) * 100
    is_good_entry = (-1.0 <= dist_ma20_pct <= 6.0)
    is_overextended = (dist_ma20_pct > 6.5)
    supports = [ma20_val, kijun_val]
    if st_dir == 1:
    supports.append(st_val)
    valid_supports = [s for s in supports if s < curr_p]
    support_level = max(valid_supports) if valid_supports else (curr_p - 1.5 * atr_val)
    raw_sl = max(support_level - 0.5 * atr_val, curr_p - 2.5 * atr_val)
    if raw_sl >= curr_p * 0.985:
    raw_sl = curr_p - 1.5 * atr_val
    raw_tp = max(bb_upper, high_20, curr_p + 3.0 * atr_val)
    stop_loss = round_vn_tick(raw_sl, exchange)
    take_profit = round_vn_tick(raw_tp, exchange)
    risk_per_share = max(curr_p - stop_loss, 10.0)
    reward_per_share = max(take_profit - curr_p, 0.0)
    rr_ratio = round(reward_per_share / risk_per_share, 2)
    # 2. Điều kiện khóa cứng "Chỉ mua khi SuperTrend Xanh + ADX >= 22 + Không Phân Kỳ Âm"
    is_st_green = (st_dir == 1)
    is_adx_strong = (adx_val >= 22.0) and (plus_di > minus_di)
    mandatory_buy_gate = is_st_green and is_adx_strong and (not bearish_div)
    is_uptrend = (curr_p > ma20_val) and (curr_p > cloud_top) and is_st_green and (ma20_val >= ma50_val * 0.98)
    is_sideway = ((range_20 < 9.5) and (abs(dist_ma20_pct) < 2.5)) or (adx_val < 18.0 and abs(dist_ma20_pct) < 3.0)
    is_bull_trap = (curr_p > prev_p * 1.02) and (vol_ratio < 75.0) and (rsi_val > 65)
    is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= 150.0) and is_uptrend and mandatory_buy_gate and (not is_bull_trap)
    is_stoch_timing = (stoch_k > stoch_d) and (prev_k <= prev_d or stoch_k < 35.0) and (stoch_k < 55.0)
    stoch_note = f" + Timing Stoch đẹp ({stoch_k:.0f})" if (is_stoch_timing and is_uptrend) else ""
    obv_note = " + OBV gom hàng" if obv_bull else ""
    div_note = " | ⚠️ PHÂN KỲ ÂM RSI" if bearish_div else (" | ✨ Phân kỳ dương RSI" if bullish_div else "")
    # Cảnh báo cơ bản (FA) nếu ROE âm hoặc Nợ/VCSH quá cao
    fa_warn = ""
    roe_v = fund_info.get("roe")
    de_v = fund_info.get("de")
    if roe_v is not None and roe_v < 0:
    fa_warn += f" | ⚠️ Lỗ (ROE {roe_v}%)"
    if de_v is not None and de_v > 2.5:
    fa_warn += f" | ⚠️ Nợ cao (D/E {de_v})"
    # 3. Chấm điểm 5 tiêu chí Pro
    buy_score = (
    (1 if (is_uptrend and last['MACD'] > last['Signal_Line']) else 0) +
    (1 if mandatory_buy_gate else 0) +
    (1 if (vol_ratio >= 110.0 and curr_p >= prev_p and mfi_val >= 45 and obv_bull) else 0) +
    (1 if is_good_entry else 0) +
    (1 if rr_ratio >= 1.5 else 0)
    )
    sell_score = (
    (1 if last['MACD'] < last['Signal_Line'] else 0) +
    (1 if curr_p < ma20_val else 0) +
    (1 if (curr_p < cloud_bot or st_dir == -1) else 0) +
    (1 if bearish_div else 0) +
    (2 if (rsi_val > 76 and mfi_val > 80 and not is_uptrend) else (1 if (rsi_val > 80 or mfi_val > 85) else 0))
    )
    ai_note = ""
    if ai_is_reliable:
    if ai_pct_change > 0.8 and buy_score >= 4:
    ai_note = f" | AI tin cậy ủng hộ ({ai_pct_change:+.1f}%)"
    elif ai_pct_change < -0.8:
    sell_score += 1
    ai_note = f" | AI cảnh báo chỉnh ({ai_pct_change:+.1f}%)"
    # 4. Kết luận
    krx_order_tip = " [Gợi ý KRX: Lệnh MTL]" if exchange == "HOSE" else " [Gợi ý KRX: Lệnh MTL/MAK]"
    if is_bull_trap:
    trang_thai = "🚨 CẢNH BÁO BẪY GIÁ (BULL TRAP)"
    ly_do = f"Giá tăng nhưng Vol yếu ({vol_ratio:.0f}% TB20), RSI cao ({rsi_val:.0f}) -> Tuyệt đối không mua đuổi{fa_warn}"
    elif is_vol_breakout and is_good_entry and rr_ratio >= 1.5 and sell_score <= 1:
    trang_thai = "🔥 BÙNG NỔ VOL + ĐIỂM MUA CHUẨN"
    ly_do = f"Đạt chuẩn ST Xanh + ADX {adx_val:.0f} (+DI>{minus_di:.0f}), Vol {vol_ratio:.0f}%, sát MA20 (+{dist_ma20_pct:.1f}%), R:R {rr_ratio}:1{stoch_note}{obv_note}{krx_order_tip}{ai_note}{fa_warn}"
    elif bearish_div and is_uptrend:
    trang_thai = "⚠️ PHÂN KỲ ÂM (CHẶN MUA MỚI)"
    ly_do = f"Giá tạo đỉnh mới nhưng RSI suy yếu (Phân kỳ âm) -> Rủi ro đảo chiều, ưu tiên chốt lời từng phần{fa_warn}"
    elif is_overextended and is_uptrend:
    trang_thai = "⏳ QUA ĐIỂM MUA (CHỜ CHỈNH)"
    ly_do = f"Uptrend khỏe (ADX {adx_val:.0f}, ST Xanh) nhưng giá chạy xa MA20 (+{dist_ma20_pct:.1f}%), R:R thấp ({rr_ratio}:1) -> Chờ kéo ngược{fa_warn}"
    elif is_sideway:
    trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
    ly_do = f"Xu hướng yếu (ADX {adx_val:.0f}), biên độ 20P {range_20:.1f}% ({low_20:,.0f}–{high_20:,.0f}đ){obv_note}{div_note}{fa_warn}"
    elif sell_score >= 3:
    trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
    ly_do = f"Gãy hỗ trợ MA20/SuperTrend Đỏ hoặc quá mua/phân kỳ (RSI {rsi_val:.0f} / MFI {mfi_val:.0f}){div_note}{ai_note}{fa_warn}"
    elif mandatory_buy_gate and buy_score >= 4 and rr_ratio >= 1.5 and sell_score <= 1:
    trang_thai = "🟢 TÍN HIỆU MUA ĐẸP (R:R TỐT)"
    ly_do = f"Khóa chuẩn ST Xanh + ADX {adx_val:.0f}, dòng tiền xác nhận, sát nền (+{dist_ma20_pct:.1f}%), R:R {rr_ratio}:1{stoch_note}{obv_note}{ai_note}{fa_warn}"
    elif buy_score >= 3 and sell_score <= 2:
    thieu_dk = "Phân kỳ âm RSI" if bearish_div else ("ADX < 22 hoặc -DI > +DI" if not is_adx_strong else ("SuperTrend Đỏ" if not is_st_green else "chưa đạt R:R/Dòng tiền"))
    trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
    ly_do = f"Giữ xu hướng nhưng chưa đủ điều kiện mở mua mới ({thieu_dk}), ADX {adx_val:.0f}, R:R {rr_ratio}:1{stoch_note}{obv_note}{ai_note}{fa_warn}"
    else:
    trang_thai = "⚪ ĐỨNG NGOÀI QUAN SÁT"
    ly_do = f"Không đạt điều kiện khóa (ST {'Xanh' if is_st_green else 'Đỏ'}, ADX {adx_val:.0f}), giá dưới MA20/Mây{div_note}{fa_warn}"
    return {
    "trang_thai": trang_thai,
    "ly_do": ly_do,
    "buy_score": buy_score,
    "sell_score": sell_score,
    "cat_lo": stop_loss,
    "chot_loi": take_profit,
    "rr_ratio": rr_ratio,
    "dist_ma20_pct": dist_ma20_pct,
    "cloud_top": cloud_top,
    "cloud_bot": cloud_bot,
    "adx": adx_val,
    "plus_di": plus_di,
    "minus_di": minus_di,
    "st_val": st_val,
    "st_dir": st_dir,
    "stoch_k": stoch_k,
    "stoch_d": stoch_d,
    "obv_bull": obv_bull,
    "bearish_div": bearish_div,
    "exchange": exchange
    }

--- BACKTEST CHUẨN LUẬT THANH TOÁN T+2,5 ---

def run_strategy_backtest(df_clean: pd.DataFrame):
df_bt = df_clean.tail(150).copy().reset_index(drop=True)
in_pos = False
entry_p = 0.0
sl_p = 0.0
tp_p = 0.0
days_held = 0
trades = []
equity_curve = [1.0]
for i in range(1, len(df_bt)):
row = df_bt.iloc[i]
prev = df_bt.iloc[i-1]
curr_eq = equity_curve[-1]
if not in_pos:
dist_ma20 = ((float(prev['close']) - float(prev['MA20'])) / float(prev['MA20'])) * 100
cond_trend = (prev['MACD'] > prev['Signal_Line']) and (prev['ST_Dir'] == 1) and (prev['ADX'] >= 22.0) and (prev['Plus_DI'] > prev['Minus_DI'])
if cond_trend and (0 <= dist_ma20 <= 6.0) and (prev['Vol_Ratio'] >= 100):
in_pos = True
days_held = 0
entry_p = float(row['open'])
sl_p = entry_p - 2.0 * float(prev['ATR'])
tp_p = entry_p + 3.0 * float(prev['ATR'])
equity_curve.append(curr_eq)
else:
days_held += 1
daily_ret = (float(row['close']) - float(prev['close'])) / float(prev['close'])
equity_curve.append(curr_eq * (1 + daily_ret))
# Luật T+2,5: Cổ phiếu mua ngày T (days_held=0) chỉ được phép bán từ chiều ngày T+2 (days_held >= 2)
if days_held >= 2:
if float(row['low']) <= sl_p:
trades.append(((sl_p - entry_p) / entry_p) - 0.003)
in_pos = False
elif float(row['high']) >= tp_p:
trades.append(((tp_p - entry_p) / entry_p) - 0.003)
in_pos = False
elif (row['ST_Dir'] == -1) or ((row['MACD'] < row['Signal_Line']) and (row['close'] < row['MA20'])):
trades.append(((float(row['close']) - entry_p) / entry_p) - 0.003)
in_pos = False
eq_series = pd.Series(equity_curve)
rolling_max = eq_series.cummax()
max_dd = float(((eq_series - rolling_max) / rolling_max).min()) * 100
net_profit = (equity_curve[-1] - 1.0) * 100
if trades:
wins = [t for t in trades if t > 0]
losses = [abs(t) for t in trades if t <= 0]
win_rate = (len(wins) / len(trades)) * 100
profit_factor = (sum(wins) / sum(losses)) if sum(losses) > 0 else (9.99 if sum(wins) > 0 else 0.0)
else:
win_rate = 0.0
profit_factor = 0.0
return {"total_trades": len(trades), "win_rate": win_rate, "net_profit": net_profit, "profit_factor": profit_factor, "max_dd": max_dd}
def analyze_one_stock(symbol: str, days_history: int, epochs: int, nav_von: float, risk_pct: float):
df = get_stock_data(symbol, days=days_history)
if df.empty or len(df) < 90:
st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol} (tải được {len(df)} phiên). Vui lòng kiểm tra lại mã cổ phiếu hoặc mạng!")
return
df = add_technical_indicators(df)
df_clean = df.dropna().copy()
if len(df_clean) < 45:
st.error(f"Dữ liệu sau khi tính chỉ báo quá ít cho mã {symbol} (còn {len(df_clean)} phiên).")
return
fund_info = get_fundamental_ratios(symbol)
beta_val = calculate_stock_beta(df_clean)
# Huấn luyện mô hình AI MLPRegressor
features = ('close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD', 'ADX', 'OBV')
raw_values = df_clean.loc[:, features].values
split_idx = int(len(raw_values) * 0.8)
scaler_eval = MinMaxScaler(feature_range=(0, 1))
scaler_eval.fit(raw_values[:split_idx])
scaled_eval = scaler_eval.transform(raw_values)
look_back = min(30, max(10, len(df_clean) // 4))
X_all, y_all = [], []
for i in range(look_back, len(scaled_eval)):
X_all.append(scaled_eval[i-look_back:i].flatten())
y_all.append(scaled_eval[i, 0])
X_all, y_all = np.array(X_all), np.array(y_all)
train_samples = split_idx - look_back
X_train, y_train = X_all[:train_samples], y_all[:train_samples]
X_test, y_test = X_all[train_samples:], y_all[train_samples:]
model_eval = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
model_eval.fit(X_train, y_train)
mape_err = 99.0
if len(X_test) > 0:
test_preds = model_eval.predict(X_test)
dummy_test = np.zeros((len(test_preds), len(features)))
dummy_test[:, 0] = test_preds
inv_preds = scaler_eval.inverse_transform(dummy_test)[:, 0]
actual_test = raw_values[split_idx:, 0]
mape_err = mean_absolute_percentage_error(actual_test, inv_preds) * 100
scaler_full = MinMaxScaler(feature_range=(0, 1))
scaled_full = scaler_full.fit_transform(raw_values)
X_full = np.array([scaled_full[i-look_back:i].flatten() for i in range(look_back, len(scaled_full))])
y_full = np.array([scaled_full[i, 0] for i in range(look_back, len(scaled_full))])
model_full = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
model_full.fit(X_full, y_full)
last_30_days = scaled_full[-look_back:].flatten().reshape(1, -1)
pred_scaled = model_full.predict(last_30_days).item()
dummy_next = np.zeros((1, len(features)))
dummy_next.put(0, pred_scaled)
predicted_price = scaler_full.inverse_transform(dummy_next).item(0)
current_price = float(df_clean['close'].iloc[-1])
pct_change = ((predicted_price - current_price) / current_price) * 100
ai_is_reliable = (mape_err < 3.0)
vol_ratio = float(df_clean['Vol_Ratio'].iloc[-1])
latest_ma20 = round_vn_tick(float(df_clean['MA20'].iloc[-1]), fund_info["exchange"])
sig = evaluate_stock_signals(df_clean, ai_pct_change=pct_change, ai_is_reliable=ai_is_reliable, fund_info=fund_info)
bt = run_strategy_backtest(df_clean)
gia_cat_lo = sig["cat_lo"]
gia_chot_loi = sig["chot_loi"]
pct_cat_lo = ((gia_cat_lo - current_price) / current_price) * 100
pct_chot_loi = ((gia_chot_loi - current_price) / current_price) * 100
# Tính khối lượng đi vốn chuẩn lô chẵn 100 CP
max_risk_vnd = nav_von * (risk_pct / 100.0)
risk_1_cp = max(current_price - gia_cat_lo, 10.0)
raw_shares = int(max_risk_vnd / risk_1_cp)
lot_shares = (raw_shares // 100) * 100
# Giới hạn không vượt quá tổng NAV hiện có
max_affordable_lots = (int(nav_von / current_price) // 100) * 100
recommended_shares = min(lot_shares, max_affordable_lots)
capital_needed = recommended_shares * current_price
st.divider()
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(f"Giá hiện tại ({fund_info['exchange']})", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}% TB20")
st_status_txt = "🟢 XANH (Tăng)" if sig["st_dir"] == 1 else "🔴 ĐỎ (Giảm)"
m2.metric("SuperTrend (10,3)", st_status_txt, f"Mốc ST: {round_vn_tick(sig['st_val'], fund_info['exchange']):,.0f} đ", delta_color="off")
adx_strength = "Mạnh (≥22)" if sig["adx"] >= 22 else "Yếu (<22)"
m3.metric("ADX & DI (14)", f"ADX: {sig['adx']:.1f} ({adx_strength})", f"+DI: {sig['plus_di']:.1f} | -DI: {sig['minus_di']:.1f}", delta_color="off")
m4.metric("Tỷ lệ R:R & Điểm", f"{sig['rr_ratio']} : 1", f"Điểm mua: {sig['buy_score']}/5", delta_color="off")
roe_str = f"{fund_info['roe']}%" if fund_info['roe'] is not None else "N/A"
pe_str = f"{fund_info['pe']}" if fund_info['pe'] is not None else "N/A"
m5.metric("Cơ bản & Beta (CAPM)", f"ROE: {roe_str} | P/E: {pe_str}", f"Beta (β): {beta_val} vs VNIndex", delta_color="off")
ai_badge = "Tin cậy" if ai_is_reliable else "Phụ trợ"
m6.metric("AI Dự đoán", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}% (MAPE {mape_err:.1f}% - {ai_badge})")
st.subheader(f"🤖 Kết Luận Đồng Nhất: {sig['trang_thai']}")
if "BẪY GIÁ" in sig["trang_thai"] or "CẢNH BÁO BÁN" in sig["trang_thai"] or "PHÂN KỲ ÂM" in sig["trang_thai"]:
st.error(f"Phân tích chi tiết: {sig['ly_do']}")
elif "BÙNG NỔ" in sig["trang_thai"] or "MUA ĐẸP" in sig["trang_thai"]:
st.success(f"Phân tích chi tiết: {sig['ly_do']}")
elif "SIDEWAY" in sig["trang_thai"] or "QUA ĐIỂM MUA" in sig["trang_thai"]:
st.warning(f"Phân tích chi tiết: {sig['ly_do']}")
else:
st.info(f"Phân tích chi tiết: {sig['ly_do']}")
st.markdown("#### 🎯 Kế hoạch giao dịch, Bước giá chuẩn KRX & Phân bổ vốn (Position Sizing):")
p1, p2, p3, p4 = st.columns(4)
p1.info(f"Vùng mua hợp lý (Sát MA20):\n\n**{latest_ma20:,.0f} đ – {round_vn_tick(latest_ma20 * 1.05, fund_info['exchange']):,.0f} đ**")
p2.success(f"Mục tiêu Chốt lời (Bước giá {fund_info['exchange']}):\n\n**{gia_chot_loi:,.0f} đ ({pct_chot_loi:+.1f}%)")
p3.error(f"Mốc Cắt lỗ chuẩn (Bước giá {fund_info['exchange']}):\n\nThủng {gia_cat_lo:,.0f} đ ({pct_cat_lo:+.1f}%)")
p4.warning(f"Khối lượng đi vốn tối đa (Rủi ro {risk_pct}%):\n\n{recommended_shares:,} CP** (~{capital_needed/1e6:,.1f} triệu đ)")
st.markdown("#### 📊 Kiểm thử định lượng chiến lược trên 150 phiên gần nhất (Khóa ST Xanh + ADX ≥ 22, chuẩn luật T+2,5 & 0,3% phí):")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Số lệnh / Tỷ lệ thắng", f"{bt['total_trades']} lệnh", f"Win Rate: {bt['win_rate']:.1f}%")
b2.metric("Lợi nhuận ròng chiến lược", f"{bt['net_profit']:+.2f}%")
b3.metric("Hệ số Profit Factor", f"{bt['profit_factor']:.2f}", "Tốt (>1.5)" if bt['profit_factor'] > 1.5 else "Trung bình", delta_color="off")
b4.metric("Sụt giảm tối đa (Max DD)", f"{bt['max_dd']:.2f}%", delta_color="inverse")
ngay_moi_nhat = df_clean['time'].iloc[-1]
st.subheader(f"Biểu đồ kỹ thuật {symbol} — SuperTrend & Mây Ichimoku (Đến: {ngay_moi_nhat})")
df_plot = df_clean.tail(100).copy().reset_index(drop=True)
future_dates = [f"+{k}P" for k in range(1, 27)]
future_span_a = df['Span_A_Raw'].tail(26).tolist()
future_span_b = df['Span_B_Raw'].tail(26).tolist()
x_cloud = df_plot['time'].tolist() + future_dates
y_span_a = df_plot['Senkou_A'].tolist() + future_span_a
y_span_b = df_plot['Senkou_B'].tolist() + future_span_b
st_bull = np.where(df_plot['ST_Dir'] == 1, df_plot['SuperTrend'], np.nan)
st_bear = np.where(df_plot['ST_Dir'] == -1, df_plot['SuperTrend'], np.nan)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20 (Nền hỗ trợ)'))
fig.add_trace(go.Scatter(x=df_plot['time'], y=st_bull, line=dict(color='#00C853', width=2.2), name='SuperTrend (Tăng)'))
fig.add_trace(go.Scatter(x=df_plot['time'], y=st_bear, line=dict(color='#FF1744', width=2.2), name='SuperTrend (Giảm)'))
fig.add_trace(go.Scatter(x=x_cloud, y=y_span_a, line=dict(color='rgba(0, 200, 100, 0.45)', width=1), name='Senkou A (Kéo 26P)'))
fig.add_trace(go.Scatter(x=x_cloud, y=y_span_b, line=dict(color='rgba(255, 80, 80, 0.45)', width=1), fill='tonexty', fillcolor='rgba(144, 238, 144, 0.18)', name='Mây Ichimoku'))
fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Cắt lỗ hỗ trợ")
fig.update_xaxes(type='category', tickangle=-45, nticks=42)
fig.update_layout(xaxis_rangeslider_visible=False, height=560, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)
def scan_single_stock(ma: str):
try:
df_ma = get_stock_data(ma, days=320)
if len(df_ma) < 90:
return None
df_ma = add_technical_indicators(df_ma).dropna()
if len(df_ma) < 10:
return None
fund_info = get_fundamental_ratios(ma)
last = df_ma.iloc[-1]
curr_p = float(last['close'])
sig = evaluate_stock_signals(df_ma, ai_pct_change=0.0, ai_is_reliable=False, fund_info=fund_info)
st_txt = "🟢 Xanh" if sig["st_dir"] == 1 else "🔴 Đỏ"
roe_txt = f"{fund_info['roe']}%" if fund_info['roe'] is not None else "-"
pe_txt = f"{fund_info['pe']}" if fund_info['pe'] is not None else "-"
return {
"Mã CP": ma,
"Sàn": fund_info["exchange"],
"Giá hiện tại (đ)": round(curr_p, 0),
"Cắt lỗ (đ)": round(sig["cat_lo"], 0),
"Chốt lời (đ)": round(sig["chot_loi"], 0),
"Tỷ lệ R:R": f"{sig['rr_ratio']} : 1",
"SuperTrend": st_txt,
"ADX (+DI/-DI)": f"{sig['adx']:.0f} ({sig['plus_di']:.0f}/{sig['minus_di']:.0f})",
"ROE / P/E": f"{roe_txt} / {pe_txt}",
"Cách nền MA20": f"{sig['dist_ma20_pct']:+.1f}%",
"Tỷ lệ Vol (%)": f"{float(last['Vol_Ratio']):.0f}%",
"Điểm Số": int(sig['buy_score']),
"Điểm Mua": f"{sig['buy_score']}/5",
"Khuyến nghị": sig["trang_thai"],
"Phân tích & Lý do chi tiết": sig["ly_do"]
}
except Exception:
return None
def check_journal_trade_history(ma_cp: str, ngay_vao: str, gia_vao: float, cat_lo: float, chot_loi: float):
df_hist = get_stock_data(ma_cp, days=90)
if df_hist.empty:
return gia_vao, "⏳ ĐANG NẮM GIỮ", "-"
df_after = df_hist[df_hist['time'] >= ngay_vao].copy()
gia_tt = float(df_hist['close'].iloc[-1])
for _, r in df_after.iterrows():
h_val = float(r['high'])
l_val = float(r['low'])
ngay_p = r['time']
hit_tp = h_val >= chot_loi
hit_sl = l_val <= cat_lo
if hit_tp and hit_sl:
return float(r['close']), "⚠️ KHÔNG XÁC ĐỊNH (Chạm cả 2 trong phiên)", ngay_p
elif hit_tp:
return chot_loi, "🎉 ĐẠT CHỐT LỜI", ngay_p
elif hit_sl:
return cat_lo, "🛑 CHẠM CẮT LỖ", ngay_p
return gia_tt, "⏳ ĐANG NẮM GIỮ", "-"
DEFAULT_TOP_125 = (
'MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS', 'EVS', 'TVS',
'ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB', 'BAB', 'NVB',
'CEO', 'DIG', 'DXG', 'DXS', 'HDC', 'HDG', 'HUT', 'IDC', 'IJC', 'KBC', 'KDH', 'L14', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VPI', 'VRE',
'HPG', 'HSG', 'NKG', 'VGS', 'TVN', 'SMC', 'TLH',
'BSR', 'GAS', 'OIL', 'PLX', 'POW', 'PVD', 'PVS', 'PVC', 'PVT', 'GEG', 'NT2', 'PC1', 'REE', 'BCG', 'GEX',
'FPT', 'CMG', 'ELC', 'ITD', 'CTR', 'VGI', 'FOX',
'DGC', 'DCM', 'DPM', 'CSV', 'LAS', 'BFC', 'GVR', 'DPR', 'PHR', 'DRC',
'MWG', 'MSN', 'PNJ', 'DGW', 'FRT', 'PET', 'VNM', 'SAB', 'KDC', 'MCH', 'PAN', 'TAR', 'LTG', 'DBC', 'BAF', 'HAG', 'ANV', 'VHC', 'IDI', 'CMX',
'GMD', 'HAH', 'VSC', 'VOS', 'VJC', 'HVN', 'ACV', 'CTD', 'VCG', 'HHV', 'LCG', 'FCN', 'CII', 'KSB', 'HT1', 'BCC', 'TNG', 'VGT', 'MSH', 'GIL', 'VCS', 'PTB', 'RAL'
)
DANH_MUC_NHOM = {
"🌐 Tự động tải Top 150 Cổ phiếu Sôi động nhất Thị trường (Động)": "DYNAMIC",
"🏛️ Nhóm Cổ phiếu Chứng khoán": ('MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS', 'EVS', 'TVS', 'APS', 'ART', 'IVS', 'PSI', 'VDS'),
"🏦 Nhóm Cổ phiếu Ngân hàng": ('ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB', 'BAB', 'NVB', 'ABB', 'BVB', 'KLB', 'PGB', 'VAB', 'VBB'),
"🏗️ Nhóm Bất động sản & Khu công nghiệp": ('CEO', 'DIG', 'DXG', 'DXS', 'HDC', 'HDG', 'HUT', 'IDC', 'IJC', 'KBC', 'KDH', 'L14', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VPI', 'VRE', 'NTL', 'NBB', 'CRE', 'HPX', 'LDG', 'TIP', 'PHR', 'GVR'),
"🔥 Nhóm Sàn HNX Tiêu biểu": ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'BVS', 'LAS', 'PVC', 'DTD', 'TIG', 'L14', 'NTP', 'PLC', 'TVD', 'NBC', 'DDG', 'NDN', 'MST', 'AMV', 'TAR', 'VC3', 'NRC'),
"⚙️ Nhóm Thép - Dầu khí - Hóa chất - Công nghệ": ('HPG', 'HSG', 'NKG', 'VGS', 'BSR', 'GAS', 'OIL', 'PLX', 'POW', 'PVD', 'PVS', 'PVC', 'PVT', 'FPT', 'CMG', 'ELC', 'CTR', 'VGI', 'DGC', 'DCM', 'DPM', 'CSV', 'GMD', 'HAH', 'MWG', 'FRT', 'DGW', 'PNJ', 'VHC', 'ANV', 'DBC', 'HAG')
}
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích & Backtest Chuyên Sâu", "🔍 Quét Đa Trạng Thái & Tỷ Lệ R:R", "📝 Nhật Ký Thực Chiến (High/Low)"])
with tab1:
col1, col2, col3, col4, col5, col6 = st.columns((1.2, 1.1, 1.1, 1.3, 1.0, 1.3))
symbol = col1.text_input("Mã CP (HOSE/HNX/UPCOM):", value="MBS").upper().strip()
days_history = col2.selectbox("Dữ liệu học (ngày):", (300, 500, 730, 1000), index=2)
epochs = col3.selectbox("Vòng học AI:", (80, 150, 250), index=1)
nav_von = col4.number_input("Tổng vốn NAV (VNĐ):", min_value=10000000, value=100000000, step=10000000)
risk_pct = col5.number_input("Rủi ro/lệnh (%):", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
col6.write("")
col6.write("")
run_btn = col6.button("🚀 Chạy Phân Tích", use_container_width=True, type="primary")
if run_btn and symbol:
with st.spinner(f"Đang phân tích chuyên sâu (TA + FA + CAPM + KRX) & chạy Backtest T+2,5 cho mã {symbol}..."):
analyze_one_stock(symbol, days_history, epochs, nav_von, risk_pct)
with tab2:
st.subheader("Quét Xu Hướng Tăng (ST Xanh + ADX ≥ 22), Dòng Tiền (OBV/MFI), Phân Kỳ & Cơ Bản (ROE/PE)")
c_filter1, c_filter2, c_filter3 = st.columns((2, 1.5, 1.5))
nhom_chon = c_filter1.selectbox("Chọn nhóm cổ phiếu / chế độ quét:", list(DANH_MUC_NHOM.keys()))
ma_rieng = c_filter2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="VD: BSR, KBC, VGC").upper().strip()
che_do_loc = c_filter3.selectbox("Bộ lọc hiển thị trạng thái:", (
"📋 Hiển thị TẤT CẢ (Mua + Bán + Chờ chỉnh + Đứng ngoài)",
"🟢 Chỉ hiện Tín hiệu MUA ĐẸP & Bùng nổ Vol (ST Xanh + ADX ≥ 22)",
"🔴 Chỉ hiện Cảnh báo BÁN / Chốt lời / Phân kỳ âm / Bẫy giá",
"⏳ Chỉ hiện Qua điểm mua (Chờ chỉnh) & Sideway"
))
scan_btn = st.button("⚡ Bắt Đầu Quét Siêu Tốc Ngay", type="primary")
if scan_btn:
nhom_val = DANH_MUC_NHOM[nhom_chon]
ds_quet = get_dynamic_top_tickers(DEFAULT_TOP_125) if nhom_val == "DYNAMIC" else list(nhom_val)
if ma_rieng:
for m in ma_rieng.split(','):
m_clean = m.strip()
if m_clean and m_clean not in ds_quet:
ds_quet.insert(0, m_clean)
progress_bar = st.progress(0)
status_text = st.empty()
ket_qua_loc = []
tong_so = len(ds_quet)
da_xong = 0
with ThreadPoolExecutor(max_workers=8) as executor:
future_to_ma = {executor.submit(scan_single_stock, ma): ma for ma in ds_quet}
for future in as_completed(future_to_ma):
da_xong += 1
progress_bar.progress(da_xong / tong_so)
status_text.text(f"⚡ Đang quét song song: {da_xong}/{tong_so} mã cổ phiếu...")
res = future.result()
if res is not None:
ket_qua_loc.append(res)
status_text.text(f"✅ Hoàn tất quét {tong_so} mã cổ phiếu!")
if ket_qua_loc:
df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Điểm Số", "Khuyến nghị"], ascending=[False, True]).drop(columns=["Điểm Số"])
if "Chỉ hiện Tín hiệu MUA" in che_do_loc:
df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP")]
elif "Chỉ hiện Cảnh báo BÁN" in che_do_loc:
df_res = df_res[df_res["Khuyến nghị"].str.contains("CẢNH BÁO BÁN|BẪY GIÁ|PHÂN KỲ ÂM")]
elif "Chỉ hiện Qua điểm mua" in che_do_loc:
df_res = df_res[df_res["Khuyến nghị"].str.contains("QUA ĐIỂM MUA|SIDEWAY|ĐỨNG NGOÀI|NẮM GIỮ")]
st.write(f"Hiển thị {len(df_res)} cổ phiếu theo bộ lọc đã chọn:")
st.dataframe(df_res, use_container_width=True, hide_index=True)
else:
st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
with tab3:
st.subheader("📝 Nhật Ký Thực Chiến (Cho phép Nhập & Chỉnh Sửa Trực Tiếp)")
st.write("Bạn có thể nhập thông số khớp lệnh thực tế bên dưới, hoặc bấm đúp chuột trực tiếp vào từng ô trên bảng để chỉnh sửa Giá vào, Cắt lỗ, Chốt lời, Ngày đóng và Ghi chú.")
with st.form("form_them_ma"):
fc1, fc2, fc3, fc4, fc5 = st.columns((1.2, 1.3, 1.3, 1.3, 2))
ma_luu = fc1.text_input("Mã CP ():", placeholder="VD: POW, HAG").upper().strip()
gia_vao_tay = fc2.number_input("Giá vào (đ) (0=Tự lấy):", min_value=0, value=0, step=50)
cat_lo_tay = fc3.number_input("Cắt lỗ (đ) (0=Tự tính):", min_value=0, value=0, step=50)
chot_loi_tay = fc4.number_input("Chốt lời (đ) (0=Tự tính):", min_value=0, value=0, step=50)
ghi_chu = fc5.text_input("Ghi chú:", value="Điểm mua chuẩn R:R >= 1.5")
submit_luu = st.form_submit_button("➕ Thêm lệnh vào Nhật ký", type="primary")
if submit_luu and ma_luu:
info_ma = scan_single_stock(ma_luu)
now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d')
if info_ma or gia_vao_tay > 0:
ex_ma = info_ma["Sàn"] if info_ma else "HOSE"
g_vao = float(gia_vao_tay) if gia_vao_tay > 0 else float(info_ma["Giá hiện tại (đ)"])
c_lo = float(cat_lo_tay) if cat_lo_tay > 0 else (float(info_ma["Cắt lỗ (đ)"]) if info_ma else round_vn_tick(g_vao * 0.93, ex_ma))
c_loi = float(chot_loi_tay) if chot_loi_tay > 0 else (float(info_ma["Chốt lời (đ)"]) if info_ma else round_vn_tick(g_vao * 1.10, ex_ma))
st.session_state.nhat_ky_lenh.append({
"Ngày vào": now_str,
"Mã CP": ma_luu,
"Giá vào (đ)": round(g_vao, 0),
"Cắt lỗ (đ)": round(c_lo, 0),
"Chốt lời (đ)": round(c_loi, 0),
"Giá đóng tay (đ)": 0.0,
"Ngày đóng": "",
"Ghi chú": ghi_chu
})
st.success(f"Đã lưu lệnh {ma_luu} (Giá vào: {g_vao:,.0f} đ) vào nhật ký!")
else:
st.error(f"Không lấy được dữ liệu của mã {ma_luu}. Bạn hãy tự nhập Giá vào (đ) ở ô bên cạnh nhé!")
if st.session_state.nhat_ky_lenh:
st.markdown("#### ✏️ Bảng điều chỉnh thông số lệnh (Bấm đúp vào ô bất kỳ để sửa, chọn dòng và nhấn Delete để xóa):")
df_edit_input = pd.DataFrame(st.session_state.nhat_ky_lenh)
for col_req in ["Ngày vào", "Mã CP", "Giá vào (đ)", "Cắt lỗ (đ)", "Chốt lời (đ)", "Giá đóng tay (đ)", "Ngày đóng", "Ghi chú"]:
if col_req not in df_edit_input.columns:
df_edit_input[col_req] = 0.0 if "(đ)" in col_req else ""
edited_df = st.data_editor(
df_edit_input,
num_rows="dynamic",
use_container_width=True,
key="editor_nhat_ky",
column_config={
"Giá vào (đ)": st.column_config.NumberColumn(format="%d"),
"Cắt lỗ (đ)": st.column_config.NumberColumn(format="%d"),
"Chốt lời (đ)": st.column_config.NumberColumn(format="%d"),
"Giá đóng tay (đ)": st.column_config.NumberColumn(help="Nhập giá bán thực tế nếu bạn đã chủ động chốt lời/cắt lỗ tay (để 0 nếu hệ thống tự theo dõi)", format="%d")
}
)
st.session_state.nhat_ky_lenh = edited_df.to_dict('records')
st.markdown("#### 📊 Kết quả chấm điểm & Lãi/Lỗ thực tế (Đã trừ 0,3% thuế phí):")
bang_cham_diem = []
so_lenh_thang = 0
tong_lenh_xac_dinh = 0
for item in st.session_state.nhat_ky_lenh:
ma_cp = str(item.get("Mã CP", "")).upper().strip()
if not ma_cp:
continue
ngay_vao = str(item.get("Ngày vào", "")).strip()
gia_vao = float(item.get("Giá vào (đ)", 0) or 0)
cat_lo = float(item.get("Cắt lỗ (đ)", 0) or 0)
chot_loi = float(item.get("Chốt lời (đ)", 0) or 0)
gia_dong_tay = float(item.get("Giá đóng tay (đ)", 0) or 0)
ngay_dong_tay = str(item.get("Ngày đóng", "")).strip()
if gia_vao <= 0:
continue
if gia_dong_tay > 0:
gia_khop = gia_dong_tay
ngay_dong = ngay_dong_tay if ngay_dong_tay else (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d')
ket_qua = "✅ ĐÃ CHỐT TAY (LÃI)" if ((gia_khop - gia_vao) / gia_vao * 100 - 0.3) > 0 else "🔻 ĐÃ CẮT TAY (LỖ)"
else:
gia_khop, ket_qua, ngay_dong = check_journal_trade_history(ma_cp, ngay_vao, gia_vao, cat_lo, chot_loi)
if ngay_dong_tay:
ngay_dong = ngay_dong_tay
lai_lo_rong = ((gia_khop - gia_vao) / gia_vao) * 100 - 0.3
if "ĐẠT CHỐT LỜI" in ket_qua or "CHỐT TAY" in ket_qua:
so_lenh_thang += 1
tong_lenh_xac_dinh += 1
elif "CHẠM CẮT LỖ" in ket_qua or "CẮT TAY" in ket_qua:
tong_lenh_xac_dinh += 1
bang_cham_diem.append({
"Ngày vào": ngay_vao,
"Mã CP": ma_cp,
"Giá vào (đ)": round(gia_vao, 0),
"Giá TT / Khớp (đ)": round(gia_khop, 0),
"Cắt lỗ (đ)": round(cat_lo, 0),
"Chốt lời (đ)": round(chot_loi, 0),
"Ngày đóng": ngay_dong,
"Lãi/Lỗ ròng (%)": f"{lai_lo_rong:+.2f}%",
"Trạng thái": ket_qua,
"Ghi chú": item.get("Ghi chú", "")
})
if bang_cham_diem:
df_nk = pd.DataFrame(bang_cham_diem)
st.dataframe(df_nk, use_container_width=True, hide_index=True)
if tong_lenh_xac_dinh > 0:
st.metric("Tỷ lệ thắng thực chiến (các lệnh đã đóng)", f"{so_lenh_thang}/{tong_lenh_xac_dinh} ({so_lenh_thang/tong_lenh_xac_dinh100:.0f}%)")
c_btn1, c_btn2 = st.columns(2)
csv_data = df_nk.to_csv(index=False).encode('utf-8-sig')
c_btn1.download_button("📥 Tải Nhật Ký về máy (CSV)", data=csv_data, file_name="nhat_ky_thuc_chien.csv", mime="text/csv")
if c_btn2.button("🗑️ Xóa toàn bộ nhật ký làm lại"):
st.session_state.nhat_ky_lenh = []
st.rerun()
