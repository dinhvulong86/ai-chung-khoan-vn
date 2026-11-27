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

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (V3.3 Pro + Thực Chiến)", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Xu Hướng (ADX + SuperTrend + Stochastic) & R:R (V3.3 Pro)")
st.markdown("**Nguyên tắc kỷ luật**: Chỉ kích hoạt **MUA** khi **SuperTrend Xanh & ADX ≥ 22**, ưu tiên Dòng tiền, Điểm mua sát nền (Stochastic) và **R:R ≥ 1.5** + **Bước giá chuẩn + Quản trị vốn**.")

if st.button("🔄 Làm mới dữ liệu trực tiếp (Xóa Cache & Cập nhật giá mới nhất)"):
    st.cache_data.clear()
    st.rerun()

if "nhat_ky_lenh" not in st.session_state:
    st.session_state.nhat_ky_lenh = [
        {"Ngày vào": "2026-11-25", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ (đ)": 13314, "Chốt lời (đ)": 14654, "Giá đóng tay (đ)": 0.0, "Ngày đóng": "", "Ghi chú": "Mẫu Bùng nổ Vol"}
    ]

# ====================== HÀM THỰC CHIẾN (BỔ SUNG) ======================
def get_price_step(price: float, exchange: str = "HOSE") -> int:
    exchange = exchange.upper()
    if exchange in ["HOSE", "HSX"]:
        if price < 10000:
            return 10
        elif price < 50000:
            return 50
        else:
            return 100
    else:  # HNX / UPCOM
        if price < 10000:
            return 10
        else:
            return 100

def round_price(price: float, exchange: str = "HOSE", direction: str = "nearest") -> float:
    step = get_price_step(price, exchange)
    if direction == "down":
        return np.floor(price / step) * step
    elif direction == "up":
        return np.ceil(price / step) * step
    else:
        return round(price / step) * step

def detect_exchange(symbol: str) -> str:
    hnx_typical = {"MBS", "SHS", "CEO", "PVS", "IDC", "HUT", "TNG", "VCS", "BVS", "LAS", "PVC", "DTD", "TIG", "NTP", "PLC", "TVD", "NBC", "DDG", "NDN", "MST", "AMV", "TAR", "VC3", "NRC", "APS", "ART", "IVS", "PSI", "VDS"}
    if symbol.upper() in hnx_typical:
        return "HNX"
    return "HOSE"

def calculate_position_size(capital: float, risk_percent: float, entry: float, stop_loss: float) -> int:
    if entry <= stop_loss or capital <= 0 or risk_percent <= 0:
        return 0
    risk_amount = capital * (risk_percent / 100.0)
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return 0
    shares = risk_amount / risk_per_share
    lot = int(shares // 100) * 100
    return max(lot, 0)

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 25) -> str:
    if len(df) < lookback + 5:
        return "Không rõ"
    recent = df.tail(lookback).copy()
    # Bullish divergence
    price_low_idx = recent['low'].idxmin()
    rsi_at_low = recent.loc[price_low_idx, 'RSI']
    later = recent.loc[price_low_idx:]
    if len(later) >= 4:
        if later['low'].iloc[-1] < recent.loc[price_low_idx, 'low'] and later['RSI'].iloc[-1] > rsi_at_low + 3:
            return "Phân kỳ Dương (Bullish)"
    # Bearish divergence
    price_high_idx = recent['high'].idxmax()
    rsi_at_high = recent.loc[price_high_idx, 'RSI']
    later_h = recent.loc[price_high_idx:]
    if len(later_h) >= 4:
        if later_h['high'].iloc[-1] > recent.loc[price_high_idx, 'high'] and later_h['RSI'].iloc[-1] < rsi_at_high - 3:
            return "Phân kỳ Âm (Bearish)"
    return "Không có phân kỳ rõ"

# ====================== LẤY DỮ LIỆU ======================
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
    now_vn = datetime.utcnow() + timedelta(hours=7)
    today_str = now_vn.strftime('%Y-%m-%d')
    end_ts = int(now_vn.timestamp())
    start_ts = int((now_vn - timedelta(days=days)).timestamp())
    sym = symbol.upper().strip()
    df = pd.DataFrame()

    # 1. Vietcap
    try:
        url_vci = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
        payload = {"timeFrame": "ONE_DAY", "symbols": [sym], "to": end_ts, "countBack": min(int(days * 0.75), 1000)}
        headers = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json', 'Origin': 'https://trading.vietcap.com.vn'}
        res = requests.post(url_vci, json=payload, headers=headers, timeout=8).json()
        if isinstance(res, list) and len(res) > 0:
            d = res[0]
            if isinstance(d, dict) and d.get('t') and len(d.get('t')) >= 30:
                df = pd.DataFrame({
                    'time': pd.to_datetime(pd.to_numeric(d.get('t')), unit='s').strftime('%Y-%m-%d'),
                    'open': pd.to_numeric(d.get('o'), errors='coerce'),
                    'high': pd.to_numeric(d.get('h'), errors='coerce'),
                    'low': pd.to_numeric(d.get('l'), errors='coerce'),
                    'close': pd.to_numeric(d.get('c'), errors='coerce'),
                    'volume': pd.to_numeric(d.get('v'), errors='coerce')
                })
                for col in ['open', 'high', 'low', 'close']:
                    if df[col].iloc[-1] < 1000:
                        df[col] = df[col] * 1000
                df = df.dropna().sort_values('time').reset_index(drop=True)
    except Exception:
        pass

    # 2. DNSE / Entrade
    if df.empty:
        try:
            url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?symbol={sym}&resolution=1D&from={start_ts}&to={end_ts}"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8).json()
            if isinstance(res, dict) and res.get('t') and len(res.get('t')) >= 30:
                df = pd.DataFrame({
                    'time': pd.to_datetime(pd.to_numeric(res.get('t')), unit='s').strftime('%Y-%m-%d'),
                    'open': pd.to_numeric(res.get('o'), errors='coerce'),
                    'high': pd.to_numeric(res.get('h'), errors='coerce'),
                    'low': pd.to_numeric(res.get('l'), errors='coerce'),
                    'close': pd.to_numeric(res.get('c'), errors='coerce'),
                    'volume': pd.to_numeric(res.get('v'), errors='coerce')
                })
                for col in ['open', 'high', 'low', 'close']:
                    if df[col].iloc[-1] < 1000:
                        df[col] = df[col] * 1000
                df = df.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass

    # 3. yfinance fallback
    if df.empty:
        try:
            start_date = (now_vn - timedelta(days=days)).strftime('%Y-%m-%d')
            ticker_vn = f"{sym}.VN"
            df_yf = yf.download(ticker_vn, start=start_date, end=today_str, progress=False, auto_adjust=True)
            if df_yf is not None and not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                df_yf = df_yf.reset_index()
                df_yf = df_yf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df_yf.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                df_yf['time'] = pd.to_datetime(df_yf['time']).dt.strftime('%Y-%m-%d')
                df = df_yf.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass

    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_dynamic_top_tickers(fallback_list):
    try:
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/second-gainer?exchange=HOSE&limit=100"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=6).json()
        items = res.get('data', [])
        tickers = [x['ticker'] for x in items if x.get('ticker') and len(x['ticker']) == 3]
        if len(tickers) >= 20:
            return tickers
    except Exception:
        pass
    return list(fallback_list)

# ====================== CHỈ BÁO KỸ THUẬT ======================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['STD20'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
    df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = (df['volume'] / df['Vol_MA20'].replace(0, np.nan)) * 100

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # ADX
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

    # SuperTrend
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

    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    fast_k = (100 * ((df['close'] - low_14) / denom)).fillna(50.0)
    df['Stoch_K'] = fast_k.rolling(window=3, min_periods=1).mean()
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean()

    # Ichimoku
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['Span_A_Raw'] = (df['Tenkan'] + df['Kijun']) / 2
    df['Span_B_Raw'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
    df['Senkou_A'] = df['Span_A_Raw'].shift(26)
    df['Senkou_B'] = df['Span_B_Raw'].shift(26)

    df['High_20'] = df['high'].rolling(20).max()
    df['Low_20'] = df['low'].rolling(20).min()
    df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20'].replace(0, np.nan)) * 100

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MFI
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    pos_mf = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0.0), index=df.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0.0), index=df.index)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum()
    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # OBV (bổ sung)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    return df

# ====================== ĐÁNH GIÁ TÍN HIỆU ======================
def evaluate_stock_signals(df_clean: pd.DataFrame, exchange: str = "HOSE"):
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
    cloud_top = max(float(last['Senkou_A']), float(last['Senkou_B']))
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

    dist_ma20_pct = ((curr_p - ma20_val) / ma20_val) * 100
    is_good_entry = (-1.0 <= dist_ma20_pct <= 6.0)
    is_overextended = (dist_ma20_pct > 6.5)

    # Hỗ trợ
    supports = [ma20_val, kijun_val]
    if st_dir == 1:
        supports.append(st_val)
    valid_supports = [s for s in supports if s < curr_p]
    support_level = max(valid_supports) if valid_supports else (curr_p - 1.5 * atr_val)
    stop_loss_raw = max(support_level - 0.5 * atr_val, curr_p - 2.5 * atr_val)
    if stop_loss_raw >= curr_p * 0.985:
        stop_loss_raw = curr_p - 1.5 * atr_val
    take_profit_raw = max(bb_upper, high_20, curr_p + 3.0 * atr_val)

    # === LÀM TRÒN BƯỚC GIÁ ===
    stop_loss = round_price(stop_loss_raw, exchange, direction="down")
    take_profit = round_price(take_profit_raw, exchange, direction="up")
    curr_p_rounded = round_price(curr_p, exchange)

    risk_per_share = max(curr_p - stop_loss, 1.0)
    reward_per_share = max(take_profit - curr_p, 0.0)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    # Khóa cứng
    is_st_green = (st_dir == 1)
    is_adx_strong = (adx_val >= 22.0) and (plus_di > minus_di)
    mandatory_buy_gate = is_st_green and is_adx_strong

    is_uptrend = (curr_p > ma20_val) and (curr_p > cloud_top) and (ma20_val >= ma50_val * 0.98)
    is_sideway = ((range_20 < 9.5) and (abs(dist_ma20_pct) < 2.5)) or (adx_val < 18.0)
    is_bull_trap = (curr_p > prev_p * 1.02) and (vol_ratio < 80) and (rsi_val > 72)

    # Stochastic timing
    stoch_oversold_cross = (prev_k < prev_d) and (stoch_k > stoch_d) and (stoch_k < 35)
    stoch_overbought = stoch_k > 80

    divergence = detect_rsi_divergence(df_clean)

    # Điểm số
    buy_score = 0.0
    if mandatory_buy_gate:
        buy_score += 2.0
    if is_uptrend:
        buy_score += 1.0
    if is_good_entry:
        buy_score += 1.0
    if rr_ratio >= 1.5:
        buy_score += 1.0
    if vol_ratio >= 130:
        buy_score += 0.5
    if mfi_val >= 55:
        buy_score += 0.5
    if stoch_oversold_cross:
        buy_score += 0.5
    if "Dương" in divergence:
        buy_score += 0.5

    # Khuyến nghị
    if is_bull_trap:
        trang_thai = "⚠️ CẢNH BÁO BẪY GIÁ (Bull Trap)"
        ly_do = "Giá tăng mạnh nhưng volume yếu + RSI cao"
    elif not mandatory_buy_gate:
        if st_dir == -1:
            trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
            ly_do = "SuperTrend đang Đỏ"
        else:
            trang_thai = "⏳ ĐỨNG NGOÀI (Chưa đủ ADX ≥ 22)"
            ly_do = f"ADX hiện tại {adx_val:.1f} < 22 hoặc -DI chiếm ưu thế"
    elif is_sideway:
        trang_thai = "⏳ SIDEWAY / ĐỨNG NGOÀI"
        ly_do = f"Biên độ hẹp hoặc ADX thấp ({adx_val:.0f})"
    elif is_overextended:
        trang_thai = "⏳ QUA ĐIỂM MUA (Chờ chỉnh)"
        ly_do = f"Cách MA20 {dist_ma20_pct:+.1f}% → chờ về gần nền"
    elif buy_score >= 4.5 and rr_ratio >= 1.5:
        if vol_ratio >= 160:
            trang_thai = "🚀 BÙNG NỔ VOL - MUA ĐẸP"
        else:
            trang_thai = "🟢 MUA ĐẸP (R:R tốt)"
        ly_do = f"ST Xanh + ADX {adx_val:.0f} | Stochastic timing | R:R {rr_ratio}"
    else:
        trang_thai = "⏳ ĐỨNG NGOÀI / CHỜ TÍN HIỆU"
        ly_do = "Chưa đủ điểm hoặc R:R chưa đạt"

    return {
        "trang_thai": trang_thai,
        "ly_do": ly_do,
        "buy_score": round(buy_score, 1),
        "rr_ratio": rr_ratio,
        "cat_lo": stop_loss,
        "chot_loi": take_profit,
        "gia_hien_tai": curr_p_rounded,
        "dist_ma20_pct": dist_ma20_pct,
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "st_dir": st_dir,
        "stoch_k": stoch_k,
        "vol_ratio": vol_ratio,
        "divergence": divergence,
        "exchange": exchange
    }

# ====================== PHÂN TÍCH 1 MÃ ======================
def analyze_one_stock(symbol: str, days_history: int = 500, capital: float = 100_000_000, risk_pct: float = 1.0):
    df = get_stock_data(symbol, days=days_history)
    if len(df) < 90:
        st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol}")
        return

    df = add_technical_indicators(df).dropna()
    exchange = detect_exchange(symbol)
    sig = evaluate_stock_signals(df, exchange)

    so_luong = calculate_position_size(capital, risk_pct, sig["gia_hien_tai"], sig["cat_lo"])
    gia_tri = so_luong * sig["gia_hien_tai"]
    rui_ro_tien = so_luong * (sig["gia_hien_tai"] - sig["cat_lo"])

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá hiện tại", f"{sig['gia_hien_tai']:,.0f} đ")
    c2.metric("Cắt lỗ (đã làm tròn)", f"{sig['cat_lo']:,.0f} đ")
    c3.metric("Chốt lời (đã làm tròn)", f"{sig['chot_loi']:,.0f} đ")
    c4.metric("R:R", f"{sig['rr_ratio']} : 1")

    st.success(f"**Khuyến nghị: {sig['trang_thai']}**")
    st.write(f"**Lý do chi tiết:** {sig['ly_do']}")
    st.write(f"**ADX:** {sig['adx']:.1f} (+DI {sig['plus_di']:.0f} / -DI {sig['minus_di']:.0f}) | **SuperTrend:** {'🟢 Xanh' if sig['st_dir']==1 else '🔴 Đỏ'} | **Stoch %K:** {sig['stoch_k']:.0f} | **Phân kỳ RSI:** {sig['divergence']}")

    # Quản trị vốn
    st.subheader("📦 Quản trị vốn & Khối lượng đề xuất")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Số lượng CP (lô 100)", f"{so_luong:,}")
    k2.metric("Giá trị lệnh", f"{gia_tri:,.0f} đ")
    k3.metric("Rủi ro tiền", f"{rui_ro_tien:,.0f} đ")
    k4.metric("% Rủi ro", f"{risk_pct}%")

    if so_luong == 0:
        st.warning("⚠️ Rủi ro quá lớn so với vốn hoặc điểm cắt lỗ quá sát → Không nên vào lệnh với mức rủi ro hiện tại.")

    # Biểu đồ
    df_plot = df.tail(150).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'],
                                 low=df_plot['low'], close=df_plot['close'], name="Giá"))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], name="MA20", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['SuperTrend'],
                             name="SuperTrend", line=dict(color="lime" if sig['st_dir']==1 else "red", width=2.2)))
    fig.add_hline(y=sig['chot_loi'], line_dash="dot", line_color="green", annotation_text="Chốt lời")
    fig.add_hline(y=sig['cat_lo'], line_dash="dot", line_color="red", annotation_text="Cắt lỗ")
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, title=f"{symbol} — {sig['trang_thai']}")
    st.plotly_chart(fig, use_container_width=True)

# ====================== QUÉT ======================
def scan_single_stock(ma: str):
    try:
        df_ma = get_stock_data(ma, days=320)
        if len(df_ma) < 90:
            return None
        df_ma = add_technical_indicators(df_ma).dropna()
        if len(df_ma) < 10:
            return None
        exchange = detect_exchange(ma)
        sig = evaluate_stock_signals(df_ma, exchange)
        st_txt = "🟢 Xanh" if sig["st_dir"] == 1 else "🔴 Đỏ"
        return {
            "Mã CP": ma,
            "Giá (đ)": round(sig["gia_hien_tai"], 0),
            "Cắt lỗ (đ)": round(sig["cat_lo"], 0),
            "Chốt lời (đ)": round(sig["chot_loi"], 0),
            "R:R": f"{sig['rr_ratio']} : 1",
            "SuperTrend": st_txt,
            "ADX": f"{sig['adx']:.0f}",
            "Stoch %K": f"{sig['stoch_k']:.0f}",
            "Cách MA20": f"{sig['dist_ma20_pct']:+.1f}%",
            "Vol %": f"{sig['vol_ratio']:.0f}%",
            "Phân kỳ": sig["divergence"],
            "Điểm": sig["buy_score"],
            "Khuyến nghị": sig["trang_thai"]
        }
    except Exception:
        return None

# ====================== GIAO DIỆN ======================
DEFAULT_TOP_125 = (
    'MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS',
    'ACB', 'BID', 'CTG', 'HDB', 'MBB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB',
    'CEO', 'DIG', 'DXG', 'HDC', 'HDG', 'KBC', 'KDH', 'NLG', 'NVL', 'PDR', 'VHM', 'VIC', 'VRE',
    'HPG', 'HSG', 'NKG', 'GAS', 'PLX', 'POW', 'PVD', 'FPT', 'CMG', 'MWG', 'PNJ', 'VNM', 'HAG'
)

DANH_MUC_NHOM = {
    "🌐 Top động": "DYNAMIC",
    "🏛️ Chứng khoán": ('MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'ORS', 'VIX'),
    "🏦 Ngân hàng": ('ACB', 'MBB', 'TCB', 'VPB', 'VIB', 'TPB', 'HDB', 'STB', 'CTG', 'BID', 'VCB'),
    "🏗️ Bất động sản": ('VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'DXG', 'KDH', 'DIG', 'NLG', 'CEO', 'KBC'),
    "🔥 HNX tiêu biểu": ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'BVS', 'LAS'),
    "⚙️ Thép - Dầu khí - Công nghệ": ('HPG', 'HSG', 'NKG', 'GAS', 'PLX', 'PVD', 'FPT', 'CMG', 'MWG', 'PNJ')
}

tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Thực Chiến", "🔍 Quét Đa Trạng Thái", "📝 Nhật Ký"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.text_input("Mã cổ phiếu", value="HPG").upper().strip()
    capital = col2.number_input("Vốn khả dụng (đ)", min_value=10_000_000, value=100_000_000, step=5_000_000)
    risk_pct = col3.number_input("% Rủi ro / lệnh", min_value=0.5, max_value=3.0, value=1.0, step=0.25)
    days_history = col4.selectbox("Số ngày lịch sử", (300, 500, 730), index=1)

    if st.button("🚀 Chạy Phân Tích Ngay", type="primary", use_container_width=True):
        with st.spinner(f"Đang phân tích {symbol}..."):
            analyze_one_stock(symbol, days_history, capital, risk_pct)

with tab2:
    st.subheader("Quét theo nhóm (chỉ hiện mã đạt điều kiện ST Xanh + ADX ≥ 22)")
    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    nhom_chon = c1.selectbox("Chọn nhóm", list(DANH_MUC_NHOM.keys()))
    ma_rieng = c2.text_input("Thêm mã riêng (cách nhau dấu phẩy)", placeholder="BSR, KBC...")
    che_do = c3.selectbox("Bộ lọc", [
        "📋 Tất cả",
        "🟢 Chỉ MUA ĐẸP / Bùng nổ Vol",
        "🔴 Chỉ Cảnh báo BÁN",
        "⏳ Chỉ Đứng ngoài / Chờ chỉnh"
    ])

    if st.button("⚡ Bắt Đầu Quét", type="primary"):
        nhom_val = DANH_MUC_NHOM[nhom_chon]
        ds_quet = get_dynamic_top_tickers(DEFAULT_TOP_125) if nhom_val == "DYNAMIC" else list(nhom_val)
        if ma_rieng:
            for m in ma_rieng.split(","):
                m = m.strip().upper()
                if m and m not in ds_quet:
                    ds_quet.insert(0, m)

        progress = st.progress(0)
        status = st.empty()
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(scan_single_stock, ma): ma for ma in ds_quet}
            done = 0
            for fut in as_completed(futures):
                done += 1
                progress.progress(done / len(ds_quet))
                status.text(f"Đang quét {done}/{len(ds_quet)}...")
                res = fut.result()
                if res:
                    results.append(res)

        status.text(f"✅ Hoàn tất {len(ds_quet)} mã")
        if results:
            df_res = pd.DataFrame(results).sort_values("Điểm", ascending=False)
            if "MUA ĐẸP" in che_do:
                df_res = df_res[df_res["Khuyến nghị"].str.contains("MUA ĐẸP|BÙNG NỔ")]
            elif "Cảnh báo BÁN" in che_do:
                df_res = df_res[df_res["Khuyến nghị"].str.contains("CẢNH BÁO BÁN|BẪY GIÁ")]
            elif "Đứng ngoài" in che_do:
                df_res = df_res[df_res["Khuyến nghị"].str.contains("ĐỨNG NGOÀI|SIDEWAY|QUA ĐIỂM|CHỜ")]
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning("Không lấy được dữ liệu.")

with tab3:
    st.subheader("📝 Nhật Ký Thực Chiến")
    with st.form("form_them"):
        f1, f2, f3, f4, f5 = st.columns([1.2, 1.3, 1.3, 1.3, 2])
        ma_luu = f1.text_input("Mã CP").upper().strip()
        gia_vao = f2.number_input("Giá vào (0 = tự lấy)", min_value=0, value=0, step=50)
        cat_lo = f3.number_input("Cắt lỗ (0 = tự tính)", min_value=0, value=0, step=50)
        chot_loi = f4.number_input("Chốt lời (0 = tự tính)", min_value=0, value=0, step=50)
        ghi_chu = f5.text_input("Ghi chú", value="R:R ≥ 1.5")
        if st.form_submit_button("➕ Thêm lệnh"):
            if ma_luu:
                info = scan_single_stock(ma_luu)
                now_str = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d")
                g_vao = float(gia_vao) if gia_vao > 0 else (float(info["Giá (đ)"]) if info else 0)
                c_lo = float(cat_lo) if cat_lo > 0 else (float(info["Cắt lỗ (đ)"]) if info else round(g_vao * 0.94))
                c_loi = float(chot_loi) if chot_loi > 0 else (float(info["Chốt lời (đ)"]) if info else round(g_vao * 1.08))
                st.session_state.nhat_ky_lenh.append({
                    "Ngày vào": now_str, "Mã CP": ma_luu,
                    "Giá vào (đ)": g_vao, "Cắt lỗ (đ)": c_lo, "Chốt lời (đ)": c_loi,
                    "Giá đóng tay (đ)": 0.0, "Ngày đóng": "", "Ghi chú": ghi_chu
                })
                st.success(f"Đã thêm {ma_luu}")

    if st.session_state.nhat_ky_lenh:
        df_edit = pd.DataFrame(st.session_state.nhat_ky_lenh)
        edited = st.data_editor(df_edit, num_rows="dynamic", use_container_width=True, key="editor")
        st.session_state.nhat_ky_lenh = edited.to_dict("records")
