raw_code = r"""
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
____st.cache_data.clear()
____st.rerun()

if "nhat_ky_lenh" not in st.session_state:
____st.session_state.nhat_ky_lenh = [
________{"Ngày vào": "2026-11-25", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ (đ)": 13300, "Chốt lời (đ)": 14650, "Giá đóng tay (đ)": 0.0, "Ngày đóng": "", "Ghi chú": "Mẫu Bùng nổ Vol"}
____]

def round_vn_tick(price: float, exchange: str = "HOSE") -> float:
____if price <= 0:
________return 0.0
____ex = exchange.upper()
____if ex in ["HNX", "UPCOM"]:
________tick = 100.0
____else:
________if price < 10000:
____________tick = 10.0
________elif price < 50000:
____________tick = 50.0
________else:
____________tick = 100.0
____return round(round(price / tick) * tick, 0)

@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
____now_vn = datetime.utcnow() + timedelta(hours=7)
____today_str = now_vn.strftime('%Y-%m-%d')
____end_ts = int(now_vn.timestamp())
____start_ts = int((now_vn - timedelta(days=days)).timestamp())
____sym = symbol.upper().strip()
____df = pd.DataFrame()

____try:
________url_vci = "https://vietcap.com.vn"
________payload_vci = {
____________"timeFrame": "ONE_DAY",
____________"symbols": [sym],
____________"to": end_ts,
____________"countBack": min(int(days * 0.75), 1000)
________}
________headers_vci = {
____________'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
____________'Content-Type': 'application/json',
____________'Origin': 'https://vietcap.com.vn',
____________'Referer': 'https://vietcap.com.vn/'
________}
________res_vci = requests.post(url_vci, json=payload_vci, headers=headers_vci, timeout=8).json()
________if isinstance(res_vci, list) and len(res_vci) > 0:
____________d = res_vci[0]
____________if isinstance(d, dict) and d.get('t') and len(d.get('t')) >= 30:
________________df = pd.DataFrame({
____________________'time': pd.to_datetime(pd.to_numeric(d.get('t')), unit='s').strftime('%Y-%m-%d'),
____________________'open': pd.to_numeric(d.get('o'), errors='coerce'),
____________________'high': pd.to_numeric(d.get('h'), errors='coerce'),
____________________'low': pd.to_numeric(d.get('l'), errors='coerce'),
____________________'close': pd.to_numeric(d.get('c'), errors='coerce'),
____________________'volume': pd.to_numeric(d.get('v'), errors='coerce')
________________})
________________if sym != "VNINDEX":
____________________for col in ['open', 'high', 'low', 'close']:
________________________if df[col].iloc[-1] < 1000:
____________________________df[col] = df[col] * 1000
________________df = df.dropna().sort_values('time').reset_index(drop=True)
____except Exception:
________pass

____if df.empty:
________try:
____________url_dnse = "https://entrade.com.vn"
____________params_dnse = {"symbol": sym, "resolution": "1D", "from": start_ts, "to": end_ts}
____________res_dnse = requests.get(url_dnse, params=params_dnse, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8).json()
____________if isinstance(res_dnse, dict) and res_dnse.get('t') and len(res_dnse.get('t')) >= 30:
________________df = pd.DataFrame({
____________________'time': pd.to_datetime(pd.to_numeric(res_dnse.get('t')), unit='s').strftime('%Y-%m-%d'),
____________________'open': pd.to_numeric(res_dnse.get('o'), errors='coerce'),
____________________'high': pd.to_numeric(res_dnse.get('h'), errors='coerce'),
____________________'low': pd.to_numeric(res_dnse.get('l'), errors='coerce'),
____________________'close': pd.to_numeric(res_dnse.get('c'), errors='coerce'),
____________________'volume': pd.to_numeric(res_dnse.get('v'), errors='coerce')
________________})
________________if sym != "VNINDEX":
____________________for col in ['open', 'high', 'low', 'close']:
________________________if df[col].iloc[-1] < 1000:
____________________________df[col] = df[col] * 1000
________________df = df.dropna().sort_values('time').reset_index(drop=True)
________except Exception:
____________pass

____if df.empty and sym != "VNINDEX":
________try:
____________url_sim = f"https://simplize.vn{sym}?page=0&size={min(int(days * 0.75), 1000)}"
____________res_sim = requests.get(url_sim, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8).json()
____________items = res_sim.get('data', [])
____________if isinstance(items, list) and len(items) >= 30:
________________raw_df = pd.DataFrame(items)
________________df = pd.DataFrame({
____________________'time': pd.to_datetime(raw_df.get('date'), unit='s').dt.strftime('%Y-%m-%d'),
____________________'open': pd.to_numeric(raw_df.get('priceOpen'), errors='coerce'),
____________________'high': pd.to_numeric(raw_df.get('priceHigh'), errors='coerce'),
____________________'low': pd.to_numeric(raw_df.get('priceLow'), errors='coerce'),
____________________'close': pd.to_numeric(raw_df.get('priceClose'), errors='coerce'),
____________________'volume': pd.to_numeric(raw_df.get('totalVolume'), errors='coerce')
________________})
________________for col in ['open', 'high', 'low', 'close']:
____________________if df[col].iloc[-1] < 1000:
________________________df[col] = df[col] * 1000
________________df = df.dropna().sort_values('time').reset_index(drop=True)
________except Exception:
____________pass

____if df.empty:
________try:
____________start_date = (now_vn - timedelta(days=days)).strftime('%Y-%m-%d')
____________ticker_vn = "^VNINDEX" if sym == "VNINDEX" else (sym if sym.endswith(".VN") else f"{sym}.VN")
____________df_yf = yf.download(ticker_vn, start=start_date, end=today_str, progress=False, auto_adjust=True)
____________if df_yf is not None and not df_yf.empty:
________________if isinstance(df_yf.columns, pd.MultiIndex):
____________________df_yf.columns = df_yf.columns.get_level_values(0)
________________df_yf = df_yf.reset_index()
________________df_yf = df_yf.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
________________df_yf.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
________________df_yf['time'] = pd.to_datetime(df_yf['time']).dt.strftime('%Y-%m-%d')
________________for col in ['open', 'high', 'low', 'close', 'volume']:
____________________df_yf[col] = pd.to_numeric(df_yf[col], errors='coerce')
________________df = df_yf.dropna().sort_values('time').reset_index(drop=True)
________except Exception:
____________pass

____return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_ratios(symbol: str) -> dict:
____sym = symbol.upper().strip()
____info = {"roe": None, "pe": None, "pb": None, "de": None, "exchange": "HOSE"}
____try:
________url_tcbs = f"https://tcbs.com.vn{sym}/overview"
________res = requests.get(url_tcbs, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
________if isinstance(res, dict):
____________ex_raw = str(res.get("exchange", "HOSE")).upper()
____________if "HNX" in ex_raw:
________________info["exchange"] = "HNX"
____________elif "UPCOM" in ex_raw:
________________info["exchange"] = "UPCOM"
____________if res.get("roe") is not None:
________________info["roe"] = round(float(res["roe"]) * 100, 1)
____________if res.get("pe") is not None:
________________info["pe"] = round(float(res["pe"]), 1)
____________if res.get("pb") is not None:
________________info["pb"] = round(float(res["pb"]), 2)
____________if res.get("debtOnEquity") is not None:
________________info["de"] = round(float(res["debtOnEquity"]), 2)
____except Exception:
________pass
____return info

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_stock_beta(df_stock: pd.DataFrame) -> float:
____try:
________df_mkt = get_stock_data("VNINDEX", days=400)
________if df_mkt.empty or len(df_mkt) < 60:
____________return 1.0
________merged = pd.merge(df_stock[['time', 'close']], df_mkt[['time', 'close']], on='time', suffixes=('_s', '_m'))
________if len(merged) < 60:
____________return 1.0
________ret_s = merged['close_s'].pct_change().dropna()
________ret_m = merged['close_m'].pct_change().dropna()
________cov = np.cov(ret_s, ret_m)[0, 1]
________var_m = np.var(ret_m)
________return round(float(cov / var_m), 2) if var_m > 0 else 1.0
____except Exception:
________return 1.0

@st.cache_data(ttl=600, show_spinner=False)
def get_dynamic_top_tickers(fallback_list):
____try:
________url = "https://tcbs.com.vn"
________headers = {'User-Agent': 'Mozilla/5.0'}
________res = requests.get(url, headers=headers, timeout=6).json()
________items = res.get('data', [])
________tickers = [x['ticker'] for x in items if x.get('ticker') and len(x['ticker']) == 3]
________if len(tickers) >= 30:
____________return tickers
____except Exception:
________pass
____return list(fallback_list)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
____df = df.copy()
____df['MA20'] = df['close'].rolling(window=20).mean()
____df['MA50'] = df['close'].rolling(window=50).mean()
____df['STD20'] = df['close'].rolling(window=20).std()
____df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
____df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
____df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
____df['Vol_Ratio'] = (df['volume'] / df['Vol_MA20'].replace(0, np.nan)) * 100
____obv_dir = np.sign(df['close'].diff()).fillna(0)
____df['OBV'] = (obv_dir * df['volume']).cumsum()
____df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
____tr1 = df['high'] - df['low']
____tr2 = (df['high'] - df['close'].shift(1)).abs()
____tr3 = (df['low'] - df['close'].shift(1)).abs()
____df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
____df['ATR'] = df['TR'].ewm(alpha=1/14, min_periods=14, adjust=False).mean()
____up_move = df['high'] - df['high'].shift(1)
____down_move = df['low'].shift(1) - df['low']
____plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
____minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
____atr_14 = df['ATR'].replace(0, np.nan)
____df['Plus_DI'] = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_14)
____df['Minus_DI'] = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_14)
____di_sum = (df['Plus_DI'] + df['Minus_DI']).replace(0, np.nan)
____dx = 100 * ((df['Plus_DI'] - df['Minus_DI']).abs() / di_sum)
____df['ADX'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
____st_atr = df['TR'].ewm(alpha=1/10, min_periods=1, adjust=False).mean()
____hl2 = (df['high'] + df['low']) / 2
____basic_ub = (hl2 + 3.0 * st_atr).fillna(df['close']).values
____basic_lb = (hl2 - 3.0 * st_atr).fillna(df['close']).values
____close_vals = df['close'].values
____n = len(df)
____final_ub = np.zeros(n)
____final_lb = np.zeros(n)
____supertrend = np.zeros(n)
____st_dir = np.ones(n, dtype=int)
____for i in range(n):
________if i == 0:
____________final_ub[i] = basic_ub[i]
____________final_lb[i] = basic_lb[i]
____________supertrend[i] = basic_lb[i]
____________continue
________final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_vals[i-1] > final_ub[i-1]) else final_ub[i-1]
________final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_vals[i-1] < final_lb[i-1]) else final_lb[i-1]
________if st_dir[i-1] == 1:
____________if close_vals[i] < final_lb[i]:
________________st_dir[i] = -1
________________supertrend[i] = final_ub[i]
____________else:
________________st_dir[i] = 1
________________supertrend[i] = final_lb[i]
________else:
____________if close_vals[i] > final_ub[i]:
________________st_dir[i] = 1
________________supertrend[i] = final_lb[i]
____________else:
________________st_dir[i] = -1
________________supertrend[i] = final_ub[i]
____df['SuperTrend'] = supertrend
____df['ST_Dir'] = st_dir
____low_14 = df['low'].rolling(window=14).min()
____high_14 = df['high'].rolling(window=14).max()
____denom_stoch = (high_14 - low_14).replace(0, np.nan)
____fast_k = (100 * ((df['close'] - low_14) / denom_stoch)).fillna(50.0)
____df['Stoch_K'] = fast_k.rolling(window=3, min_periods=1).mean()
____df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean()
____df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
____df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
____df['Span_A_Raw'] = (df['Tenkan'] + df['Kijun']) / 2
____df['Span_B_Raw'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
____df['Senkou_A'] = df['Span_A_Raw'].shift(26)
____df['Senkou_B'] = df['Span_B_Raw'].shift(26)
____df['High_20'] = df['high'].rolling(window=20).max()
____df['Low_20'] = df['low'].rolling(window=20).min()
____df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20'].replace(0, np.nan)) * 100
____delta = df['close'].diff()
____gain = delta.where(delta > 0, 0.0)
____loss = -delta.where(delta < 0, 0.0)
____avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
____avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
____rsi_vals = np.where(
________(avg_gain == 0) & (avg_loss == 0), 50.0,
________np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss.replace(0, np.nan)))))
____)
____df['RSI'] = pd.Series(rsi_vals, index=df.index)
____tp = (df['high'] + df['low'] + df['close']) / 3
____raw_mf = tp * df['volume']
____pos_mf = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0.0), index=df.index)
____neg_mf = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0.0), index=df.index)
____pos_sum = pos_mf.rolling(window=14).sum()
____neg_sum = neg_mf.rolling(window=14).sum()
____mfi_vals = np.where(
________(pos_sum == 0) & (neg_sum == 0), 50.0,
________np.where(neg_sum == 0, 100.0,
________np.where(pos_sum == 0, 0.0, 100.0 - (100.0 / (1.0 + (pos_sum / neg_sum.replace(0, np.nan))))))
____)
____df['MFI'] = pd.Series(mfi_vals, index=df.index)
____exp1 = df['close'].ewm(span=12, adjust=False).mean()
____exp2 = df['close'].ewm(span=26, adjust=False).mean()
____df['MACD'] = exp1 - exp2
____df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
____return df
def detect_rsi_divergence(df_clean: pd.DataFrame, lookback: int = 15):
____if len(df_clean) < lookback + 5:
________return False, False
____sub = df_clean.tail(lookback)
____curr_p = float(sub['close'].iloc[-1])
____curr_rsi = float(sub['RSI'].iloc[-1])
____past = sub.iloc[:-3]
____max_past_p = float(past['close'].max())
____max_past_rsi = float(past['RSI'].max())
____min_past_p = float(past['close'].min())
____min_past_rsi = float(past['RSI'].min())

____bearish_div = (curr_p >= max_past_p * 0.995) and (curr_rsi < max_past_rsi - 5.0) and (max_past_rsi > 62.0)
____bullish_div = (curr_p <= min_past_p * 1.005) and (curr_rsi > min_past_rsi + 5.0) and (min_past_rsi < 38.0)
____return bearish_div, bullish_div

def evaluate_stock_signals(df_clean: pd.DataFrame, ai_pct_change: float = 0.0, ai_is_reliable: bool = False, fund_info: dict = None):
____if fund_info is None:
________fund_info = {"roe": None, "pe": None, "pb": None, "de": None, "exchange": "HOSE"}
____exchange = fund_info.get("exchange", "HOSE")

____last = df_clean.iloc[-1]
____prev = df_clean.iloc[-2]
____curr_p = float(last['close'])
____prev_p = float(prev['close'])
____vol_ratio = float(last['Vol_Ratio'])
____atr_val = float(last['ATR'])
____rsi_val = float(last['RSI'])
____mfi_val = float(last['MFI'])
____ma20_val = float(last['MA20'])
____ma50_val = float(last['MA50'])
____kijun_val = float(last['Kijun'])
____bb_upper = float(last['BB_Upper'])
____high_20 = float(last['High_20'])
____low_20 = float(last['Low_20'])
____cloud_top = max(float(last['Senkou_A']), float(last['Senkou_B']))
____cloud_bot = min(float(last['Senkou_A']), float(last['Senkou_B']))
____range_20 = float(last['Range_20_Pct'])
____adx_val = float(last['ADX'])
____plus_di = float(last['Plus_DI'])
____minus_di = float(last['Minus_DI'])
____st_val = float(last['SuperTrend'])
____st_dir = int(last['ST_Dir'])
____stoch_k = float(last['Stoch_K'])
____stoch_d = float(last['Stoch_D'])
____prev_k = float(prev['Stoch_K'])
____prev_d = float(prev['Stoch_D'])
____obv_bull = float(last['OBV']) > float(last['OBV_MA20'])

____bearish_div, bullish_div = detect_rsi_divergence(df_clean)

____dist_ma20_pct = ((curr_p - ma20_val) / ma20_val) * 100
____is_good_entry = (-1.0 <= dist_ma20_pct <= 6.0)
____is_overextended = (dist_ma20_pct > 6.5)

____supports = [ma20_val, kijun_val]
____if st_dir == 1:
________supports.append(st_val)
____valid_supports = [s for s in supports if s < curr_p]
____support_level = max(valid_supports) if valid_supports else (curr_p - 1.5 * atr_val)

____raw_sl = max(support_level - 0.5 * atr_val, curr_p - 2.5 * atr_val)
____if raw_sl >= curr_p * 0.985:
________raw_sl = curr_p - 1.5 * atr_val
____raw_tp = max(bb_upper, high_20, curr_p + 3.0 * atr_val)

____stop_loss = round_vn_tick(raw_sl, exchange)
____take_profit = round_vn_tick(raw_tp, exchange)

____risk_per_share = max(curr_p - stop_loss, 10.0)
____reward_per_share = max(take_profit - curr_p, 0.0)
____rr_ratio = round(reward_per_share / risk_per_share, 2)

____is_st_green = (st_dir == 1)
____is_adx_strong = (adx_val >= 22.0) and (plus_di > minus_di)
____mandatory_buy_gate = is_st_green and is_adx_strong and (not bearish_div)

____is_uptrend = (curr_p > ma20_val) and (curr_p > cloud_top) and is_st_green and (ma20_val >= ma50_val * 0.98)
____is_sideway = ((range_20 < 9.5) and (abs(dist_ma20_pct) < 2.5)) or (adx_val < 18.0 and abs(dist_ma20_pct) < 3.0)
____is_bull_trap = (curr_p > prev_p * 1.02) and (vol_ratio < 75.0) and (rsi_val > 65)
____is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= 150.0) and is_uptrend and mandatory_buy_gate and (not is_bull_trap)

____is_stoch_timing = (stoch_k > stoch_d) and (prev_k <= prev_d or stoch_k < 35.0) and (stoch_k < 55.0)
____stoch_note = f" + Timing Stoch đẹp ({stoch_k:.0f})" if (is_stoch_timing and is_uptrend) else ""
____obv_note = " + OBV gom hàng" if obv_bull else ""
____div_note = " | ⚠️ PHÂN KỲ ÂM RSI" if bearish_div else (" | ✨ Phân kỳ dương RSI" if bullish_div else "")

____fa_warn = ""
____roe_v = fund_info.get("roe")
____de_v = fund_info.get("de")
____if roe_v is not None and roe_v < 0:
________fa_warn += f" | ⚠️ Lỗ (ROE {roe_v}%)"
____if de_v is not None and de_v > 2.5:
________fa_warn += f" | ⚠️ Nợ cao (D/E {de_v})"

____buy_score = (
________(1 if (is_uptrend and last['MACD'] > last['Signal_Line']) else 0) +
________(1 if mandatory_buy_gate else 0) +
________(1 if (vol_ratio >= 110.0 and curr_p >= prev_p and mfi_val >= 45 and obv_bull) else 0) +
________(1 if is_good_entry else 0) +
________(1 if rr_ratio >= 1.5 else 0)
____)
____sell_score = (
________(1 if last['MACD'] < last['Signal_Line'] else 0) +
________(1 if curr_p < ma20_val else 0) +
________(1 if (curr_p < cloud_bot or st_dir == -1) else 0) +
________(1 if bearish_div else 0) +
________(2 if (rsi_val > 76 and mfi_val > 80 and not is_uptrend) else (1 if (rsi_val > 80 or mfi_val > 85) else 0))
____)
____ai_note = ""
____if ai_is_reliable:
________if ai_pct_change > 0.8 and buy_score >= 4:
____________ai_note = f" | AI tin cậy ủng hộ ({ai_pct_change:+.1f}%)"
________elif ai_pct_change < -0.8:
____________sell_score += 1
____________ai_note = f" | AI cảnh báo chỉnh ({ai_pct_change:+.1f}%)"

____krx_order_tip = " [Gợi ý KRX: Lệnh MTL]" if exchange == "HOSE" else " [Gợi ý KRX: Lệnh MTL/MAK]"
____if is_bull_trap:
________trang_thai = "🚨 CẢNH BÁO BẪY GIÁ (BULL TRAP)"
________ly_do = f"Giá tăng nhưng Vol yếu ({vol_ratio:.0f}% TB20), RSI cao ({rsi_val:.0f}) -> Tuyệt đối không mua đuổi{fa_warn}"
____elif is_vol_breakout and is_good_entry and rr_ratio >= 1.5 and sell_score <= 1:
________trang_thai = "🔥 BÙNG NỔ VOL + ĐIỂM MUA CHUẨN"
________ly_do = f"Đạt chuẩn ST Xanh + ADX {adx_val:.0f} (+DI>{minus_di:.0f}), Vol {vol_ratio:.0f}%, sát MA20 (+{dist_ma20_pct:.1f}%), R:R {rr_ratio}:1{stoch_note}{obv_note}{krx_order_tip}{ai_note}{fa_warn}"
____elif bearish_div and is_uptrend:
________trang_thai = "⚠️ PHÂN KỲ ÂM (CHẶN MUA MỚI)"
________ly_do = f"Giá tạo đỉnh mới nhưng RSI suy yếu (Phân kỳ âm) -> Rủi ro đảo chiều, ưu tiên chốt lời từng phần{fa_warn}"
____elif is_overextended and is_uptrend:
________trang_thai = "⏳ QUA ĐIỂM MUA (CHỜ CHỈNH)"
________ly_do = f"Uptrend khỏe (ADX {adx_val:.0f}, ST Xanh) nhưng giá chạy xa MA20 (+{dist_ma20_pct:.1f}%), R:R thấp ({rr_ratio}:1) -> Chờ kéo ngược{fa_warn}"
____elif is_sideway:
________trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
________ly_do = f"Xu hướng yếu (ADX {adx_val:.0f}), biên độ 20P {range_20:.1f}% ({low_20:,.0f}–{high_20:,.0f}đ){obv_note}{div_note}{fa_warn}"
____elif sell_score >= 3:
________trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
________ly_do = f"Gãy hỗ trợ MA20/SuperTrend Đỏ hoặc quá mua/phân kỳ (RSI {rsi_val:.0f} / MFI {mfi_val:.0f}){div_note}{ai_note}{fa_warn}"
____elif mandatory_buy_gate and buy_score >= 4 and rr_ratio >= 1.5 and sell_score <= 1:
________trang_thai = "🟢 TÍN HIỆU MUA ĐẸP (R:R TỐT)"
________ly_do = f"Khóa chuẩn ST Xanh + ADX {adx_val:.0f}, dòng tiền xác nhận, sát nền (+{dist_ma20_pct:.1f}%), R:R {rr_ratio}:1{stoch_note}{obv_note}{ai_note}{fa_warn}"
____elif buy_score >= 3 and sell_score <= 2:
________thieu_dk = "Phân kỳ âm RSI" if bearish_div else ("ADX < 22 hoặc -DI > +DI" if not is_adx_strong else ("SuperTrend Đỏ" if not is_st_green else "chưa đạt R:R/Dòng tiền"))
________trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
________ly_do = f"Giữ xu hướng nhưng chưa đủ điều kiện mở mua mới ({thieu_dk}), ADX {adx_val:.0f}, R:R {rr_ratio}:1{stoch_note}{obv_note}{ai_note}{fa_warn}"
____else:
________trang_thai = "⚪ ĐỨNG NGOÀI QUAN SÁT"
________ly_do = f"Không đạt điều kiện khóa (ST {'Xanh' if is_st_green else 'Đỏ'}, ADX {adx_val:.0f}), giá dưới MA20/Mây{div_note}{fa_warn}"

____return {
________"trang_thai": trang_thai,
________"ly_do": ly_do,
________"buy_score": buy_score,
________"sell_score": sell_score,
________"cat_lo": stop_loss,
________"chot_loi": take_profit,
________"rr_ratio": rr_ratio,
________"dist_ma20_pct": dist_ma20_pct,
________"cloud_top": cloud_top,
________"cloud_bot": cloud_bot,
________"adx": adx_val,
________"plus_di": plus_di,
________"minus_di": minus_di,
________"st_val": st_val,
________"st_dir": st_dir,
________"stoch_k": stoch_k,
________"stoch_d": stoch_d,
________"obv_bull": obv_bull,
________"bearish_div": bearish_div,
________"exchange": exchange
____}

def run_strategy_backtest(df_clean: pd.DataFrame):
____df_bt = df_clean.tail(150).copy().reset_index(drop=True)
____in_pos = False
____entry_p = 0.0
____sl_p = 0.0
____tp_p = 0.0
____days_held = 0
____trades = []
____hold_days_list = []
____equity_curve = [1.0]

____for i in range(1, len(df_bt)):
________row = df_bt.iloc[i]
________prev = df_bt.iloc[i-1]
________curr_eq = equity_curve[-1]

________if not in_pos:
____________dist_ma20 = ((float(prev['close']) - float(prev['MA20'])) / float(prev['MA20'])) * 100
____________cond_trend = (prev['MACD'] > prev['Signal_Line']) and (prev['ST_Dir'] == 1) and (prev['ADX'] >= 22.0) and (prev['Plus_DI'] > prev['Minus_DI'])
____________if cond_trend and (0 <= dist_ma20 <= 6.0) and (prev['Vol_Ratio'] >= 100):
________________in_pos = True
________________days_held = 0
________________entry_p = float(row['open'])
________________sl_p = entry_p - 2.0 * float(prev['ATR'])
________________tp_p = entry_p + 3.0 * float(prev['ATR'])
____________equity_curve.append(curr_eq)
________else:
____________days_held += 1
____________daily_ret = (float(row['close']) - float(prev['close'])) / float(prev['close'])
____________equity_curve.append(curr_eq * (1 + daily_ret))

____________if days_held >= 2:
________________if float(row['low']) <= sl_p:
____________________trades.append(((sl_p - entry_p) / entry_p) - 0.003)
____________________hold_days_list.append(days_held)
____________________in_pos = False
________________elif float(row['high']) >= tp_p:
____________________trades.append(((tp_p - entry_p) / entry_p) - 0.003)
____________________hold_days_list.append(days_held)
____________________in_pos = False
________________elif (row['ST_Dir'] == -1) or ((row['MACD'] < row['Signal_Line']) and (row['close'] < row['MA20'])):
____________________trades.append(((float(row['close']) - entry_p) / entry_p) - 0.003)
____________________hold_days_list.append(days_held)
____________________in_pos = False

____eq_series = pd.Series(equity_curve)
____rolling_max = eq_series.cummax()
____max_dd = float(((eq_series - rolling_max) / rolling_max).min()) * 100
____net_profit = (equity_curve[-1] - 1.0) * 100

____if trades:
________wins = [t for t in trades if t > 0]
________losses = [abs(t) for t in trades if t <= 0]
________win_rate = (len(wins) / len(trades)) * 100
________profit_factor = (sum(wins) / sum(losses)) if sum(losses) > 0 else (9.99 if sum(wins) > 0 else 0.0)
________avg_hold_days = round(float(np.mean(hold_days_list)), 1)
____else:
________win_rate = 0.0
________profit_factor = 0.0
________avg_hold_days = 0.0

____return {"total_trades": len(trades), "win_rate": win_rate, "net_profit": net_profit, "profit_factor": profit_factor, "max_dd": max_dd, "avg_hold_days": avg_hold_days}
def analyze_one_stock(symbol: str, days_history: int, epochs: int, nav_von: float, risk_pct: float):
____df = get_stock_data(symbol, days=days_history)
____if df.empty or len(df) < 90:
________st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol} (tải được {len(df)} phiên). Vui lòng kiểm tra lại mã cổ phiếu hoặc mạng!")
________return
____df = add_technical_indicators(df)
____df_clean = df.dropna().copy()
____if len(df_clean) < 45:
________st.error(f"Dữ liệu sau khi tính chỉ báo quá ít cho mã {symbol} (còn {len(df_clean)} phiên).")
________return
____fund_info = get_fundamental_ratios(symbol)
____beta_val = calculate_stock_beta(df_clean)
____features = ('close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD', 'ADX', 'OBV')
____raw_values = df_clean.loc[:, features].values
____split_idx = int(len(raw_values) * 0.8)
____scaler_eval = MinMaxScaler(feature_range=(0, 1))
____scaler_eval.fit(raw_values[:split_idx])
____scaled_eval = scaler_eval.transform(raw_values)
____look_back = min(30, max(10, len(df_clean) // 4))
____X_all, y_all = [], []
____for i in range(look_back, len(scaled_eval)):
________X_all.append(scaled_eval[i-look_back:i].flatten())
________y_all.append(scaled_eval[i, 0])
____X_all, y_all = np.array(X_all), np.array(y_all)
____train_samples = split_idx - look_back
____X_train, y_train = X_all[:train_samples], y_all[:train_samples]
____X_test, y_test = X_all[train_samples:], y_all[train_samples:]
____model_eval = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
____model_eval.fit(X_train, y_train)
____mape_err = 99.0
____if len(X_test) > 0:
________test_preds = model_eval.predict(X_test)
________dummy_test = np.zeros((len(test_preds), len(features)))
________dummy_test[:, 0] = test_preds
________inv_preds = scaler_eval.inverse_transform(dummy_test)[:, 0]
________actual_test = raw_values[split_idx:, 0]
________mape_err = mean_absolute_percentage_error(actual_test, inv_preds) * 100
____scaler_full = MinMaxScaler(feature_range=(0, 1))
____scaled_full = scaler_full.fit_transform(raw_values)
____X_full = np.array([scaled_full[i-look_back:i].flatten() for i in range(look_back, len(scaled_full))])
____y_full = np.array([scaled_full[i, 0] for i in range(look_back, len(scaled_full))])
____model_full = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
____model_full.fit(X_full, y_full)
____last_30_days = scaled_full[-look_back:].flatten().reshape(1, -1)
____pred_scaled = model_full.predict(last_30_days).item()
____dummy_next = np.zeros((1, len(features)))
____dummy_next.put(0, pred_scaled)
____predicted_price = scaler_full.inverse_transform(dummy_next).item(0)
____current_price = float(df_clean['close'].iloc[-1])
____pct_change = ((predicted_price - current_price) / current_price) * 100
____ai_is_reliable = (mape_err < 3.0)
____vol_ratio = float(df_clean['Vol_Ratio'].iloc[-1])
____latest_ma20 = round_vn_tick(float(df_clean['MA20'].iloc[-1]), fund_info["exchange"])
____sig = evaluate_stock_signals(df_clean, ai_pct_change=pct_change, ai_is_reliable=ai_is_reliable, fund_info=fund_info)
____bt = run_strategy_backtest(df_clean)
____gia_cat_lo = sig["cat_lo"]
____gia_chot_loi = sig["chot_loi"]
____pct_cat_lo = ((gia_cat_lo - current_price) / current_price) * 100
____pct_chot_loi = ((gia_chot_loi - current_price) / current_price) * 100
____max_risk_vnd = nav_von * (risk_pct / 100.0)
____risk_1_cp = max(current_price - gia_cat_lo, 10.0)
____raw_shares = int(max_risk_vnd / risk_1_cp)
____lot_shares = (raw_shares // 100) * 100
____max_affordable_lots = (int(nav_von / current_price) // 100) * 100
____recommended_shares = min(lot_shares, max_affordable_lots)
____capital_needed = recommended_shares * current_price
____st.divider()
____m1, m2, m3, m4, m5, m6 = st.columns(6)
____m1.metric(f"Giá hiện tại ({fund_info['exchange']})", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}% TB20")
____st_status_txt = "🟢 XANH (Tăng)" if sig["st_dir"] == 1 else "🔴 ĐỎ (Giảm)"
____m2.metric("SuperTrend (10,3)", st_status_txt, f"Mốc ST: {round_vn_tick(sig['st_val'], fund_info['exchange']):,.0f} đ", delta_color="off")
____adx_strength = "Mạnh (≥22)" if sig["adx"] >= 22 else "Yếu (<22)"
____m3.metric("ADX & DI (14)", f"ADX: {sig['adx']:.1f} ({adx_strength})", f"+DI: {sig['plus_di']:.1f} | -DI: {sig['minus_di']:.1f}", delta_color="off")
____m4.metric("Tỷ lệ R:R & Điểm", f"{sig['rr_ratio']} : 1", f"Điểm mua: {sig['buy_score']}/5", delta_color="off")
____roe_str = f"{fund_info['roe']}%" if fund_info['roe'] is not None else "N/A"
____pe_str = f"{fund_info['pe']}" if fund_info['pe'] is not None else "N/A"
____m5.metric("Cơ bản & Beta (CAPM)", f"ROE: {roe_str} | P/E: {pe_str}", f"Beta (β): {beta_val} vs VNIndex", delta_color="off")
____ai_badge = "Tin cậy" if ai_is_reliable else "Phụ trợ"
____m6.metric("AI Dự đoán", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}% (MAPE {mape_err:.1f}% - {ai_badge})")
____st.subheader(f"🤖 Kết Luận Đồng Nhất: {sig['trang_thai']}")
____if "BẪY GIÁ" in sig["trang_thai"] or "CẢNH BÁO BÁN" in sig["trang_thai"] or "PHÂN KỲ ÂM" in sig["trang_thai"]:
________st.error(f"Phân tích chi tiết: {sig['ly_do']}")
____elif "BÙNG NỔ" in sig["trang_thai"] or "MUA ĐẸP" in sig["trang_thai"]:
________st.success(f"Phân tích chi tiết: {sig['ly_do']}")
____elif "SIDEWAY" in sig["trang_thai"] or "QUA ĐIỂM MUA" in sig["trang_thai"]:
________st.warning(f"Phân tích chi tiết: {sig['ly_do']}")
____else:
________st.info(f"Phân tích chi tiết: {sig['ly_do']}")
____st.markdown("#### 🎯 Kế hoạch giao dịch, Bước giá chuẩn KRX & Phân bổ vốn (Position Sizing):")
____p1, p2, p3, p4 = st.columns(4)
____p1.info(f"Vùng mua hợp lý (Sát MA20):\n\n**{latest_ma20:,.0f} đ – {round_vn_tick(latest_ma20 * 1.05, fund_info['exchange']):,.0f} đ**")
____p2.success(f"Mục tiêu Chốt lời (Bước giá {fund_info['exchange']}):\n\n**{gia_chot_loi:,.0f} đ ({pct_chot_loi:+.1f}%)")
____p3.error(f"Mốc Cắt lỗ chuẩn (Bước giá {fund_info['exchange']}):\n\nThủng {gia_cat_lo:,.0f} đ ({pct_cat_lo:+.1f}%)")
____p4.warning(f"Khối lượng đi vốn tối đa (Rủi ro {risk_pct}%):\n\n{recommended_shares:,} CP** (~{capital_needed/1e6:,.1f} triệu đ)")
____st.markdown("#### 📊 Kiểm thử định lượng chiến lược trên 150 phiên gần nhất (Khóa ST Xanh + ADX ≥ 22, chuẩn luật T+2,5 & 0,3% phí):")
____b1, b2, b3, b4, b5 = st.columns(5)
____b1.metric("Số lệnh / Tỷ lệ thắng", f"{bt['total_trades']} lệnh", f"Win Rate: {bt['win_rate']:.1f}%")
____b2.metric("Giữ lệnh trung bình", f"{bt['avg_hold_days']} phiên", "Tối thiểu T+2,5", delta_color="off")
____b3.metric("Lợi nhuận ròng chiến lược", f"{bt['net_profit']:+.2f}%")
____b4.metric("Hệ số Profit Factor", f"{bt['profit_factor']:.2f}", "Tốt (>1.5)" if bt['profit_factor'] > 1.5 else "Trung bình", delta_color="off")
____b5.metric("Sụt giảm tối đa (Max DD)", f"{bt['max_dd']:.2f}%", delta_color="inverse")
____ngay_moi_nhat = df_clean['time'].iloc[-1]
____st.subheader(f"Biểu đồ kỹ thuật {symbol} — SuperTrend & Mây Ichimoku (Đến: {ngay_moi_nhat})")
____df_plot = df_clean.tail(100).copy().reset_index(drop=True)
____future_dates = [f"+{k}P" for k in range(1, 27)]
____future_span_a = df['Span_A_Raw'].tail(26).tolist()
____future_span_b = df['Span_B_Raw'].tail(26).tolist()
____x_cloud = df_plot['time'].tolist() + future_dates
____y_span_a = df_plot['Senkou_A'].tolist() + future_span_a
____y_span_b = df_plot['Senkou_B'].tolist() + future_span_b
____st_bull = np.where(df_plot['ST_Dir'] == 1, df_plot['SuperTrend'], np.nan)
____st_bear = np.where(df_plot['ST_Dir'] == -1, df_plot['SuperTrend'], np.nan)
____fig = go.Figure()
____fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20 (Nền hỗ trợ)'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=st_bull, line=dict(color='#00C853', width=2.2), name='SuperTrend (Tăng)'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=st_bear, line=dict(color='#FF1744', width=2.2), name='SuperTrend (Giảm)'))
____fig.add_trace(go.Scatter(x=x_cloud, y=y_span_a, line=dict(color='rgba(0, 200, 100, 0.45)', width=1), name='Senkou A (Kéo 26P)'))
____fig.add_trace(go.Scatter(x=x_cloud, y=y_span_b, line=dict(color='rgba(255, 80, 80, 0.45)', width=1), fill='tonexty', fillcolor='rgba(144, 238, 144, 0.18)', name='Mây Ichimoku'))
____fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
____fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Cắt lỗ hỗ trợ")
____fig.update_xaxes(type='category', tickangle=-45, nticks=42)
____fig.update_layout(xaxis_rangeslider_visible=False, height=560, hovermode='x unified')
____st.plotly_chart(fig, use_container_width=True)
def quick_ai_next_session(df_clean: pd.DataFrame) -> tuple:
____try:
________features = ('close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD', 'ADX', 'OBV')
________raw_values = df_clean.loc[:, features].tail(120).values
________if len(raw_values) < 40:
____________return 0.0, "➖ Đi ngang"
________scaler = MinMaxScaler(feature_range=(0, 1))
________scaled = scaler.fit_transform(raw_values)
________look_back = 15
________X = np.array([scaled[i-look_back:i].flatten() for i in range(look_back, len(scaled))])
________y = np.array([scaled[i, 0] for i in range(look_back, len(scaled))])
________model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=60, random_state=42)
________model.fit(X, y)
________last_window = scaled[-look_back:].flatten().reshape(1, -1)
________pred_scaled = model.predict(last_window).item()
________dummy = np.zeros((1, len(features)))
________dummy.put(0, pred_scaled)
________pred_price = scaler.inverse_transform(dummy).item(0)
________curr_price = float(df_clean['close'].iloc[-1])
________pct = ((pred_price - curr_price) / curr_price) * 100
________if pct >= 0.5:
____________return pct, f"📈 Tăng ({pct:+.1f}%)"
________elif pct <= -0.5:
____________return pct, f"📉 Giảm ({pct:+.1f}%)"
________else:
____________return pct, f"➖ Đi ngang ({pct:+.1f}%)"
____except Exception:
________return 0.0, "➖ N/A"

def scan_single_stock(ma: str):
____try:
________df_ma = get_stock_data(ma, days=320)
________if len(df_ma) < 90:
____________return None
________df_ma = add_technical_indicators(df_ma).dropna()
________if len(df_ma) < 45:
____________return None
________fund_info = get_fundamental_ratios(ma)
________last = df_ma.iloc[-1]
________curr_p = float(last['close'])

________# 1. Chạy Backtest 150 phiên & AI dự đoán phiên tới
________bt = run_strategy_backtest(df_ma)
________ai_pct, ai_next_label = quick_ai_next_session(df_ma)

________# Đánh giá độ hợp sóng của mã (Lọc nhiễu sóng)
________is_backtest_good = (bt['profit_factor'] > 1.5) and (bt['net_profit'] > 0)
________bt_badge = f"⭐ Chuẩn sóng (PF {bt['profit_factor']:.1f} | {bt['net_profit']:+.1f}%)" if is_backtest_good else f"⚠️ Nhiễu sóng (PF {bt['profit_factor']:.1f} | {bt['net_profit']:+.1f}%)"

________sig = evaluate_stock_signals(df_ma, ai_pct_change=ai_pct, ai_is_reliable=True, fund_info=fund_info)
________st_txt = "🟢 Xanh" if sig["st_dir"] == 1 else "🔴 Đỏ"
________roe_txt = f"{fund_info['roe']}%" if fund_info['roe'] is not None else "-"
________pe_txt = f"{fund_info['pe']}" if fund_info['pe'] is not None else "-"

________return {
____________"Mã CP": ma,
____________"Sàn": fund_info["exchange"],
____________"Giá hiện tại (đ)": round(curr_p, 0),
____________"Cắt lỗ (đ)": round(sig["cat_lo"], 0),
____________"Chốt lời (đ)": round(sig["chot_loi"], 0),
____________"Tỷ lệ R:R": f"{sig['rr_ratio']} : 1",
____________"SuperTrend": st_txt,
____________"ADX (+DI/-DI)": f"{sig['adx']:.0f} ({sig['plus_di']:.0f}/{sig['minus_di']:.0f})",
____________"Dự đoán Phiên tới": ai_next_label,
____________"Lịch sử Sóng (150P)": bt_badge,
____________"Is_BT_Good": is_backtest_good,
____________"Cách nền MA20": f"{sig['dist_ma20_pct']:+.1f}%",
____________"Tỷ lệ Vol (%)": f"{float(last['Vol_Ratio']):.0f}%",
____________"Điểm Số": int(sig['buy_score']),
____________"Điểm Mua": f"{sig['buy_score']}/5",
____________"Khuyến nghị": sig["trang_thai"],
____________"Phân tích & Lý do chi tiết": sig["ly_do"]
________}
____except Exception:
________return None
def check_journal_trade_history(ma_cp: str, ngay_vao: str, gia_vao: float, cat_lo: float, chot_loi: float):
____df_hist = get_stock_data(ma_cp, days=90)
____if df_hist.empty:
________return gia_vao, "⏳ ĐANG NẮM GIỮ", "-"
____df_after = df_hist[df_hist['time'] >= ngay_vao].copy()
____gia_tt = float(df_hist['close'].iloc[-1])
____for _, r in df_after.iterrows():
________h_val = float(r['high'])
________l_val = float(r['low'])
________ngay_p = r['time']
________hit_tp = h_val >= chot_loi
________hit_sl = l_val <= cat_lo
________if hit_tp and hit_sl:
____________return float(r['close']), "⚠️ KHÔNG XÁC ĐỊNH (Chạm cả 2 trong phiên)", ngay_p
________elif hit_tp:
____________return chot_loi, "🎉 ĐẠT CHỐT LỜI", ngay_p
________elif hit_sl:
____________return cat_lo, "🛑 CHẠM CẮT LỖ", ngay_p
____return gia_tt, "⏳ ĐANG NẮM GIỮ", "-"
DEFAULT_TOP_125 = (
____'MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS', 'EVS', 'TVS',
____'ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB', 'BAB', 'NVB',
____'CEO', 'DIG', 'DXG', 'DXS', 'HDC', 'HDG', 'HUT', 'IDC', 'IJC', 'KBC', 'KDH', 'L14', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VPI', 'VRE',
____'HPG', 'HSG', 'NKG', 'VGS', 'TVN', 'SMC', 'TLH',
____'BSR', 'GAS', 'OIL', 'PLX', 'POW', 'PVD', 'PVS', 'PVC', 'PVT', 'GEG', 'NT2', 'PC1', 'REE', 'BCG', 'GEX',
____'FPT', 'CMG', 'ELC', 'ITD', 'CTR', 'VGI', 'FOX',
____'DGC', 'DCM', 'DPM', 'CSV', 'LAS', 'BFC', 'GVR', 'DPR', 'PHR', 'DRC',
____'MWG', 'MSN', 'PNJ', 'DGW', 'FRT', 'PET', 'VNM', 'SAB', 'KDC', 'MCH', 'PAN', 'TAR', 'LTG', 'DBC', 'BAF', 'HAG', 'ANV', 'VHC', 'IDI', 'CMX',
____'GMD', 'HAH', 'VSC', 'VOS', 'VJC', 'HVN', 'ACV', 'CTD', 'VCG', 'HHV', 'LCG', 'FCN', 'CII', 'KSB', 'HT1', 'BCC', 'TNG', 'VGT', 'MSH', 'GIL', 'VCS', 'PTB', 'RAL'
)
DANH_MUC_NHOM = {
____"🌐 Tự động tải Top 150 Cổ phiếu Sôi động nhất Thị trường (Động)": "DYNAMIC",
____"🏛️ Nhóm Cổ phiếu Chứng khoán": ('MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS', 'EVS', 'TVS', 'APS', 'ART', 'IVS', 'PSI', 'VDS'),
____"🏦 Nhóm Cổ phiếu Ngân hàng": ('ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB', 'BAB', 'NVB', 'ABB', 'BVB', 'KLB', 'PGB', 'VAB', 'VBB'),
____"🏗️ Nhóm Bất động sản & Khu công nghiệp": ('CEO', 'DIG', 'DXG', 'DXS', 'HDC', 'HDG', 'HUT', 'IDC', 'IJC', 'KBC', 'KDH', 'L14', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VPI', 'VRE', 'NTL', 'NBB', 'CRE', 'HPX', 'LDG', 'TIP', 'PHR', 'GVR'),
____"🔥 Nhóm Sàn HNX Tiêu biểu": ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'BVS', 'LAS', 'PVC', 'DTD', 'TIG', 'L14', 'NTP', 'PLC', 'TVD', 'NBC', 'DDG', 'NDN', 'MST', 'AMV', 'TAR', 'VC3', 'NRC'),
____"⚙️ Nhóm Thép - Dầu khí - Hóa chất - Công nghệ": ('HPG', 'HSG', 'NKG', 'VGS', 'BSR', 'GAS', 'OIL', 'PLX', 'POW', 'PVD', 'PVS', 'PVC', 'PVT', 'FPT', 'CMG', 'ELC', 'CTR', 'VGI', 'DGC', 'DCM', 'DPM', 'CSV', 'GMD', 'HAH', 'MWG', 'FRT', 'DGW', 'PNJ', 'VHC', 'ANV', 'DBC', 'HAG')
}
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích & Backtest Chuyên Sâu", "🔍 Quét Đa Trạng Thái & Tỷ Lệ R:R", "📝 Nhật Ký Thực Chiến (High/Low)"])
with tab1:
____col1, col2, col3, col4, col5, col6 = st.columns((1.2, 1.1, 1.1, 1.3, 1.0, 1.3))
____symbol = col1.text_input("Mã CP (HOSE/HNX/UPCOM):", value="MBS").upper().strip()
____days_history = col2.selectbox("Dữ liệu học (ngày):", (300, 500, 730, 1000), index=2)
____epochs = col3.selectbox("Vòng học AI:", (80, 150, 250), index=1)
____nav_von = col4.number_input("Tổng vốn NAV (VNĐ):", min_value=1000000, value=10000000, step=1000000, format="%d")
____risk_pct = col5.number_input("Rủi ro/lệnh (%):", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
____col6.write("")
____col6.write("")
____run_btn = col6.button("🚀 Chạy Phân Tích", use_container_width=True, type="primary")
____if run_btn and symbol:
________with st.spinner(f"Đang phân tích chuyên sâu (TA + FA + CAPM + KRX) & chạy Backtest T+2,5 cho mã {symbol}..."):
____________analyze_one_stock(symbol, days_history, epochs, nav_von, risk_pct)
with tab2:
____st.subheader("Quét Xu Hướng Tăng (ST Xanh + ADX ≥ 22), Dự Đoán Phiên Tới & Lọc Chất Lượng Sóng (PF > 1.5)")
____c_filter1, c_filter2, c_filter3 = st.columns((2, 1.5, 1.5))
____nhom_chon = c_filter1.selectbox("Chọn nhóm cổ phiếu / chế độ quét:", list(DANH_MUC_NHOM.keys()))
____ma_rieng = c_filter2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="VD: BSR, KBC, VGC").upper().strip()
____che_do_loc = c_filter3.selectbox("Bộ lọc hiển thị trạng thái:", (
________"📋 Hiển thị TẤT CẢ (Mua + Bán + Chờ chỉnh + Đứng ngoài)",
________"🟢 Chỉ hiện Tín hiệu MUA ĐẸP & Bùng nổ Vol (ST Xanh + ADX ≥ 22)",
________"🔴 Chỉ hiện Cảnh báo BÁN / Chốt lời / Phân kỳ âm / Bẫy giá",
________"⏳ Chỉ hiện Qua điểm mua (Chờ chỉnh) & Sideway"
____))
____only_good_bt = st.checkbox("🛡️ Chỉ giữ mã có Lịch sử Sóng tốt (Profit Factor > 1.5 & Lợi nhuận ròng 150P dương — Loại bỏ mã nhiễu sóng)", value=False)
____scan_btn = st.button("⚡ Bắt Đầu Quét Siêu Tốc Ngay", type="primary")
____if scan_btn:
________nhom_val = DANH_MUC_NHOM[nhom_chon]
________ds_quet = get_dynamic_top_tickers(DEFAULT_TOP_125) if nhom_val == "DYNAMIC" else list(nhom_val)
________if ma_rieng:
____________for m in ma_rieng.split(','):
________________m_clean = m.strip()
________________if m_clean and m_clean not in ds_quet:
____________________ds_quet.insert(0, m_clean)
________progress_bar = st.progress(0)
________status_text = st.empty()
________ket_qua_loc = []
________tong_so = len(ds_quet)
________da_xong = 0
________with ThreadPoolExecutor(max_workers=8) as executor:
____________future_to_ma = {executor.submit(scan_single_stock, ma): ma for ma in ds_quet}
____________for future in as_completed(future_to_ma):
________________da_xong += 1
________________progress_bar.progress(da_xong / tong_so)
________________status_text.text(f"⚡ Đang quét song song & chạy AI dự đoán: {da_xong}/{tong_so} mã cổ phiếu...")
________________res = future.result()
________________if res is not None:
____________________ket_qua_loc.append(res)
________status_text.text(f"✅ Hoàn tất quét {tong_so} mã cổ phiếu!")
________if ket_qua_loc:
____________df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Is_BT_Good", "Điểm Số", "Khuyến nghị"], ascending=[False, False, True])
____________if only_good_bt:
________________df_res = df_res[df_res["Is_BT_Good"] == True]
____________if "Chỉ hiện Tín hiệu MUA" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP")]
____________elif "Chỉ hiện Cảnh báo BÁN" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("CẢNH BÁO BÁN|BẪY GIÁ|PHÂN KỲ ÂM")]
____________elif "Chỉ hiện Qua điểm mua" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("QUA ĐIỂM MUA|SIDEWAY|ĐỨNG NGOÀI|NẮM GIỮ")]
____________df_res = df_res.drop(columns=["Điểm Số", "Is_BT_Good"])
____________st.write(f"Hiển thị {len(df_res)} cổ phiếu theo bộ lọc đã chọn:")
____________st.dataframe(df_res, use_container_width=True, hide_index=True)
________else:
____________st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
with tab3:
____st.subheader("📝 Nhật Ký Thực Chiến (Cho phép Nhập & Chỉnh Sửa Trực Tiếp)")
____st.write("Bạn có thể nhập thông số khớp lệnh thực tế bên dưới, hoặc bấm đúp chuột trực tiếp vào từng ô trên bảng để chỉnh sửa Giá vào, Cắt lỗ, Chốt lời, Ngày đóng và Ghi chú.")
____with st.form("form_them_ma"):
________fc1, fc2, fc3, fc4, fc5 = st.columns((1.2, 1.3, 1.3, 1.3, 2))
________ma_luu = fc1.text_input("Mã CP ():", placeholder="VD: POW, HAG").upper().strip()
________gia_vao_tay = fc2.number_input("Giá vào (đ) (0=Tự lấy):", min_value=0, value=0, step=50)
________cat_lo_tay = fc3.number_input("Cắt lỗ (đ) (0=Tự tính):", min_value=0, value=0, step=50)
________chot_loi_tay = fc4.number_input("Chốt lời (đ) (0=Tự tính):", min_value=0, value=0, step=50)
________ghi_chu = fc5.text_input("Ghi chú:", value="Điểm mua chuẩn R:R >= 1.5")
________submit_luu = st.form_submit_button("➕ Thêm lệnh vào Nhật ký", type="primary")
____if submit_luu and ma_luu:
________info_ma = scan_single_stock(ma_luu)
________now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d')
________if info_ma or gia_vao_tay > 0:
____________ex_ma = info_ma["Sàn"] if info_ma else "HOSE"
____________g_vao = float(gia_vao_tay) if gia_vao_tay > 0 else float(info_ma["Giá hiện tại (đ)"])
____________c_lo = float(cat_lo_tay) if cat_lo_tay > 0 else (float(info_ma["Cắt lỗ (đ)"]) if info_ma else round_vn_tick(g_vao * 0.93, ex_ma))
____________c_loi = float(chot_loi_tay) if chot_loi_tay > 0 else (float(info_ma["Chốt lời (đ)"]) if info_ma else round_vn_tick(g_vao * 1.10, ex_ma))
____________st.session_state.nhat_ky_lenh.append({
________________"Ngày vào": now_str,
________________"Mã CP": ma_luu,
________________"Giá vào (đ)": round(g_vao, 0),
________________"Cắt lỗ (đ)": round(c_lo, 0),
________________"Chốt lời (đ)": round(c_loi, 0),
________________"Giá đóng tay (đ)": 0.0,
________________"Ngày đóng": "",
________________"Ghi chú": ghi_chu
____________})
____________st.success(f"Đã lưu lệnh {ma_luu} (Giá vào: {g_vao:,.0f} đ) vào nhật ký!")
________else:
____________st.error(f"Không lấy được dữ liệu của mã {ma_luu}. Bạn hãy tự nhập Giá vào (đ) ở ô bên cạnh nhé!")
____if st.session_state.nhat_ky_lenh:
________st.markdown("#### ✏️ Bảng điều chỉnh thông số lệnh (Bấm đúp vào ô bất kỳ để sửa, chọn dòng và nhấn Delete để xóa):")
________df_edit_input = pd.DataFrame(st.session_state.nhat_ky_lenh)
________for col_req in ["Ngày vào", "Mã CP", "Giá vào (đ)", "Cắt lỗ (đ)", "Chốt lời (đ)", "Giá đóng tay (đ)", "Ngày đóng", "Ghi chú"]:
____________if col_req not in df_edit_input.columns:
________________df_edit_input[col_req] = 0.0 if "(đ)" in col_req else ""
________edited_df = st.data_editor(
____________df_edit_input,
____________num_rows="dynamic",
____________use_container_width=True,
____________key="editor_nhat_ky",
____________column_config={
________________"Giá vào (đ)": st.column_config.NumberColumn(format="%d"),
________________"Cắt lỗ (đ)": st.column_config.NumberColumn(format="%d"),
________________"Chốt lời (đ)": st.column_config.NumberColumn(format="%d"),
________________"Giá đóng tay (đ)": st.column_config.NumberColumn(help="Nhập giá bán thực tế nếu bạn đã chủ động chốt lời/cắt lỗ tay (để 0 nếu hệ thống tự theo dõi)", format="%d")
____________}
________)
________st.session_state.nhat_ky_lenh = edited_df.to_dict('records')
________st.markdown("#### 📊 Kết quả chấm điểm & Lãi/Lỗ thực tế (Đã trừ 0,3% thuế phí):")
________bang_cham_diem = []
________so_lenh_thang = 0
________tong_lenh_xac_dinh = 0
________for item in st.session_state.nhat_ky_lenh:
____________ma_cp = str(item.get("Mã CP", "")).upper().strip()
____________if not ma_cp:
________________continue
____________ngay_vao = str(item.get("Ngày vào", "")).strip()
____________gia_vao = float(item.get("Giá vào (đ)", 0) or 0)
____________cat_lo = float(item.get("Cắt lỗ (đ)", 0) or 0)
____________chot_loi = float(item.get("Chốt lời (đ)", 0) or 0)
____________gia_dong_tay = float(item.get("Giá đóng tay (đ)", 0) or 0)
____________ngay_dong_tay = str(item.get("Ngày đóng", "")).strip()
____________if gia_vao <= 0:
________________continue
____________if gia_dong_tay > 0:
________________gia_khop = gia_dong_tay
________________ngay_dong = ngay_dong_tay if ngay_dong_tay else (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d')
________________ket_qua = "✅ ĐÃ CHỐT TAY (LÃI)" if ((gia_khop - gia_vao) / gia_vao * 100 - 0.3) > 0 else "🔻 ĐÃ CẮT TAY (LỖ)"
____________else:
________________gia_khop, ket_qua, ngay_dong = check_journal_trade_history(ma_cp, ngay_vao, gia_vao, cat_lo, chot_loi)
________________if ngay_dong_tay:
____________________ngay_dong = ngay_dong_tay
____________lai_lo_rong = ((gia_khop - gia_vao) / gia_vao) * 100 - 0.3
____________if "ĐẠT CHỐT LỜI" in ket_qua or "CHỐT TAY" in ket_qua:
________________so_lenh_thang += 1
________________tong_lenh_xac_dinh += 1
____________elif "CHẠM CẮT LỖ" in ket_qua or "CẮT TAY" in ket_qua:
________________tong_lenh_xac_dinh += 1
____________bang_cham_diem.append({
________________"Ngày vào": ngay_vao,
________________"Mã CP": ma_cp,
________________"Giá vào (đ)": round(gia_vao, 0),
________________"Giá TT / Khớp (đ)": round(gia_khop, 0),
________________"Cắt lỗ (đ)": round(cat_lo, 0),
________________"Chốt lời (đ)": round(chot_loi, 0),
________________"Ngày đóng": ngay_dong,
________________"Lãi/Lỗ ròng (%)": f"{lai_lo_rong:+.2f}%",
________________"Trạng thái": ket_qua,
________________"Ghi chú": item.get("Ghi chú", "")
____________})
________if bang_cham_diem:
____________df_nk = pd.DataFrame(bang_cham_diem)
____________st.dataframe(df_nk, use_container_width=True, hide_index=True)
____________if tong_lenh_xac_dinh > 0:
________________st.metric("Tỷ lệ thắng thực chiến (các lệnh đã đóng)", f"{so_lenh_thang}/{tong_lenh_xac_dinh} ({so_lenh_thang/tong_lenh_xac_dinh100:.0f}%)")
____________c_btn1, c_btn2 = st.columns(2)
____________csv_data = df_nk.to_csv(index=False).encode('utf-8-sig')
____________c_btn1.download_button("📥 Tải Nhật Ký về máy (CSV)", data=csv_data, file_name="nhat_ky_thuc_chien.csv", mime="text/csv")
____________if c_btn2.button("🗑️ Xóa toàn bộ nhật ký làm lại"):
________________st.session_state.nhat_ky_lenh = []
________________st.rerun()
"""
exec('\n'.join((' ' * (len(l) - len(l.lstrip(chr(95))))) + l.lstrip(chr(95)) for l in raw_code.strip().splitlines()))
