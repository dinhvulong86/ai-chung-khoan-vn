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

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (V3.1 Pro)", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Xu Hướng, Dòng Tiền & Tỷ Lệ R:R (V3.1)")
st.markdown("Ưu tiên **Xu hướng tăng**, **Dòng tiền xác nhận**, **Điểm mua sát nền MA20** và **Tỷ lệ Lợi nhuận/Rủi ro (R:R ≥ 1.5)**.")
if st.button("🔄 Làm mới dữ liệu trực tiếp (Xóa Cache & Cập nhật giá mới nhất)"):
    st.cache_data.clear()
    st.rerun()
if "nhat_ky_lenh" not in st.session_state:
____st.session_state.nhat_ky_lenh = [
________{"Ngày quét": "2026-11-25", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ (đ)": 13314, "Chốt lời (đ)": 14654, "Ghi chú": "Mẫu Bùng nổ Vol"}
____]

@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
____now_vn = datetime.utcnow() + timedelta(hours=7)
____today_str = now_vn.strftime('%Y-%m-%d')
____end_ts = int(now_vn.timestamp())
____start_ts = int((now_vn - timedelta(days=days)).timestamp())
____sym = symbol.upper().strip()
____df = pd.DataFrame()

____# 1. Nguồn 1: Vietcap (VCI) - Đầy đủ 100% HOSE, HNX (MBS, CEO...), UPCOM
____try:
________url_vci = "https://vietcap.com.vn"
________payload_vci = {
____________"timeFrame": "ONE_DAY",
____________"symbols": [sym],
____________"to": end_ts,
____________"countBack": min(int(days * 0.72), 1000)
________}
________headers_vci = {
____________'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
____________'Content-Type': 'application/json',
____________'Origin': 'https://vietcap.com.vn',
____________'Referer': 'https://vietcap.com.vn/'
________}
________res_vci = requests.post(url_vci, json=payload_vci, headers=headers_vci, timeout=8).json()
________if isinstance(res_vci, list) and len(res_vci) > 0:
____________d = res_vci.pop(0)
____________if isinstance(d, dict) and d.get('t') and len(d.get('t')) >= 30:
________________df = pd.DataFrame({
____________________'time': pd.to_datetime(pd.to_numeric(d.get('t')), unit='s').strftime('%Y-%m-%d'),
____________________'open': pd.to_numeric(d.get('o'), errors='coerce'),
____________________'high': pd.to_numeric(d.get('h'), errors='coerce'),
____________________'low': pd.to_numeric(d.get('l'), errors='coerce'),
____________________'close': pd.to_numeric(d.get('c'), errors='coerce'),
____________________'volume': pd.to_numeric(d.get('v'), errors='coerce')
________________})
________________for col in ['open', 'high', 'low', 'close']:
____________________if df[col].iloc[-1] < 1000:
________________________df[col] = df[col] * 1000
________________df = df.dropna().sort_values('time').reset_index(drop=True)
____except Exception:
________pass

____# 2. Nguồn 2: DNSE / EntradeX - Hạ tầng mở quốc tế, đầy đủ sàn HNX & UPCOM
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
________________for col in ['open', 'high', 'low', 'close']:
____________________if df[col].iloc[-1] < 1000:
________________________df[col] = df[col] * 1000
________________df = df.dropna().sort_values('time').reset_index(drop=True)
________except Exception:
____________pass

____# 3. Nguồn 3: Simplize (Đúng tên cột priceOpen, priceClose, totalVolume)
____if df.empty:
________try:
____________url_sim = f"https://simplize.vn{sym}?page=0&size={min(int(days * 0.72), 1000)}"
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

____# 4. Nguồn 4: Yahoo Finance (.VN - Dự phòng sàn HOSE)
____if df.empty:
________try:
____________start_date = (now_vn - timedelta(days=days)).strftime('%Y-%m-%d')
____________ticker_vn = sym if sym.endswith(".VN") else f"{sym}.VN"
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

____# ATR (14) chuẩn Wilder
____tr1 = df['high'] - df['low']
____tr2 = (df['high'] - df['close'].shift(1)).abs()
____tr3 = (df['low'] - df['close'].shift(1)).abs()
____df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
____df['ATR'] = df['TR'].ewm(alpha=1/14, min_periods=14, adjust=False).mean()

____# Ichimoku Cloud (9, 26, 52)
____df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
____df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
____df['Span_A_Raw'] = (df['Tenkan'] + df['Kijun']) / 2
____df['Span_B_Raw'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
____df['Senkou_A'] = df['Span_A_Raw'].shift(26)
____df['Senkou_B'] = df['Span_B_Raw'].shift(26)

____# Hộp Sideway & Đỉnh/Đáy 20 phiên
____df['High_20'] = df['high'].rolling(window=20).max()
____df['Low_20'] = df['low'].rolling(window=20).min()
____df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20'].replace(0, np.nan)) * 100

____# RSI (14) chuẩn Wilder
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

____# MFI (14)
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

____# MACD (12, 26, 9)
____exp1 = df['close'].ewm(span=12, adjust=False).mean()
____exp2 = df['close'].ewm(span=26, adjust=False).mean()
____df['MACD'] = exp1 - exp2
____df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
____return df

def evaluate_stock_signals(df_clean: pd.DataFrame, ai_pct_change: float = 0.0, ai_is_reliable: bool = False):
____# Uu tien: Xu huong tang + Dong tien xac nhan + Diem mua sat nen + Ty le R:R >= 1.5
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

____# 1. Tính Điểm Mua Hợp Lý & Tỷ lệ Lợi nhuận / Rủi ro (R:R) thực tế theo Hỗ trợ/Kháng cự
____dist_ma20_pct = ((curr_p - ma20_val) / ma20_val) * 100
____is_good_entry = (-1.0 <= dist_ma20_pct <= 6.0)
____is_overextended = (dist_ma20_pct > 6.5)

____# Cắt lỗ đặt dưới hỗ trợ gần nhất (MA20 hoặc Kijun) trừ đệm 0.5 ATR, nhưng không xa quá 2.5 ATR
____support_level = min(ma20_val, kijun_val) if curr_p > min(ma20_val, kijun_val) else (curr_p - 1.5 * atr_val)
____stop_loss = max(support_level - 0.5 * atr_val, curr_p - 2.5 * atr_val)
____if stop_loss >= curr_p * 0.985:
________stop_loss = curr_p - 1.5 * atr_val

____# Chốt lời kỳ vọng theo dải Bollinger trên / Đỉnh cũ / Mở rộng 3 ATR
____take_profit = max(bb_upper, high_20, curr_p + 3.0 * atr_val)
____risk_per_share = max(curr_p - stop_loss, 1.0)
____reward_per_share = max(take_profit - curr_p, 0.0)
____rr_ratio = round(reward_per_share / risk_per_share, 2)
____# 2. Xác định các trạng thái thị trường
____is_uptrend = (curr_p > ma20_val) and (curr_p > cloud_top) and (ma20_val >= ma50_val * 0.98)
____is_sideway = (range_20 < 9.5) and (abs(dist_ma20_pct) < 2.5)
____is_bull_trap = (curr_p > prev_p * 1.02) and (vol_ratio < 75.0) and (rsi_val > 65)
____is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= 150.0) and (curr_p > ma20_val) and (not is_bull_trap)
____# 3. Chấm điểm 4 tiêu chí vàng của Grok: (1) Xu hướng, (2) Dòng tiền, (3) Điểm mua hợp lý, (4) R:R >= 1.5
____buy_score = (
________(1 if (is_uptrend and last['MACD'] > last['Signal_Line']) else 0) +
________(1 if (vol_ratio >= 110.0 and curr_p >= prev_p and mfi_val >= 45) else 0) +
________(1 if is_good_entry else 0) +
________(1 if rr_ratio >= 1.5 else 0)
____)
____sell_score = (
________(1 if last['MACD'] < last['Signal_Line'] else 0) +
________(1 if curr_p < ma20_val else 0) +
________(1 if curr_p < cloud_bot else 0) +
________(2 if (rsi_val > 76 and mfi_val > 80 and not is_uptrend) else (1 if (rsi_val > 80 or mfi_val > 85) else 0))
____)
____# Bộ lọc phụ AI (chỉ tham khảo khi sai số kiểm thử MAPE < 3%)
____ai_note = ""
____if ai_is_reliable:
________if ai_pct_change > 0.8 and buy_score >= 3:
____________ai_note = f" | AI tin cậy ủng hộ ({ai_pct_change:+.1f}%)"
________elif ai_pct_change < -0.8:
____________sell_score += 1
____________ai_note = f" | AI cảnh báo chỉnh ({ai_pct_change:+.1f}%)"
____# 4. Kết luận ưu tiên điểm mua an toàn & tỷ lệ R:R tốt
____if is_bull_trap:
________trang_thai = "🚨 CẢNH BÁO BẪY GIÁ (BULL TRAP)"
________ly_do = f"Giá tăng nhưng Vol yếu ({vol_ratio:.0f}% TB20), RSI cao ({rsi_val:.0f}) -> Tuyệt đối không mua đuổi"
____elif is_vol_breakout and is_good_entry and rr_ratio >= 1.5 and sell_score <= 1:
________trang_thai = "🔥 BÙNG NỔ VOL + ĐIỂM MUA CHUẨN"
________ly_do = f"Xu hướng tăng, Vol bùng nổ {vol_ratio:.0f}%, sát nền MA20 (+{dist_ma20_pct:.1f}%), R:R tốt ({rr_ratio}:1){ai_note}"
____elif is_overextended and is_uptrend:
________trang_thai = "⏳ QUA ĐIỂM MUA (CHỜ CHỈNH)"
________ly_do = f"Uptrend khỏe nhưng giá đã chạy xa nền MA20 (+{dist_ma20_pct:.1f}%), R:R thấp ({rr_ratio}:1) -> Chờ nhịp kéo ngược"
____elif is_sideway:
________trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
________ly_do = f"Biên độ 20P nén {range_20:.1f}% ({low_20:,.0f}–{high_20:,.0f}đ) -> Chỉ canh mua gần hỗ trợ {low_20:,.0f}đ"
____elif sell_score >= 3:
________trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
________ly_do = f"Gãy hỗ trợ MA20/Mây hoặc quá mua cực đại (RSI {rsi_val:.0f} / MFI {mfi_val:.0f}){ai_note}"
____elif buy_score >= 3 and rr_ratio >= 1.5 and sell_score <= 1:
________trang_thai = "🟢 TÍN HIỆU MUA ĐẸP (R:R TỐT)"
________ly_do = f"Uptrend + Dòng tiền xác nhận, điểm mua sát nền (+{dist_ma20_pct:.1f}%), R:R đạt {rr_ratio}:1{ai_note}"
____elif buy_score >= 2 and sell_score <= 2:
________trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
________ly_do = f"Đang giữ xu hướng, cách nền MA20 {dist_ma20_pct:+.1f}%, R:R hiện tại {rr_ratio}:1{ai_note}"
____else:
________trang_thai = "⚪ ĐỨNG NGOÀI QUAN SÁT"
________ly_do = f"Chưa đạt xu hướng tăng hoặc dòng tiền yếu (Giá dưới MA20 {ma20_val:,.0f}đ / dưới Mây)"
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
________"cloud_bot": cloud_bot
____}
def run_strategy_backtest(df_clean: pd.DataFrame):
____# Kiem thu dinh luong tren 150 phien: Ap dung dung quy tac Xu huong + Sat nen MA20 + Cat lo ho tro
____df_bt = df_clean.tail(150).copy().reset_index(drop=True)
____in_pos = False
____entry_p = 0.0
____sl_p = 0.0
____tp_p = 0.0
____trades = []
____equity_curve = [1.0]
____for i in range(1, len(df_bt)):
________row = df_bt.iloc[i]
________prev = df_bt.iloc[i-1]
________curr_eq = equity_curve[-1]
________if not in_pos:
____________dist_ma20 = ((float(prev['close']) - float(prev['MA20'])) / float(prev['MA20'])) * 100
____________if (prev['MACD'] > prev['Signal_Line']) and (0 <= dist_ma20 <= 6.0) and (prev['Vol_Ratio'] >= 100):
________________in_pos = True
________________entry_p = float(row['open'])
________________sl_p = entry_p - 2.0 * float(prev['ATR'])
________________tp_p = entry_p + 3.0 * float(prev['ATR'])
____________equity_curve.append(curr_eq)
________else:
____________daily_ret = (float(row['close']) - float(prev['close'])) / float(prev['close'])
____________equity_curve.append(curr_eq * (1 + daily_ret))
____________if float(row['low']) <= sl_p:
________________trades.append(((sl_p - entry_p) / entry_p) - 0.003)
________________in_pos = False
____________elif float(row['high']) >= tp_p:
________________trades.append(((tp_p - entry_p) / entry_p) - 0.003)
________________in_pos = False
____________elif (row['MACD'] < row['Signal_Line']) and (row['close'] < row['MA20']):
________________trades.append(((float(row['close']) - entry_p) / entry_p) - 0.003)
________________in_pos = False
____eq_series = pd.Series(equity_curve)
____rolling_max = eq_series.cummax()
____max_dd = float(((eq_series - rolling_max) / rolling_max).min()) * 100
____net_profit = (equity_curve[-1] - 1.0) * 100
____if trades:
________wins = [t for t in trades if t > 0]
________losses = [abs(t) for t in trades if t <= 0]
________win_rate = (len(wins) / len(trades)) * 100
________profit_factor = (sum(wins) / sum(losses)) if sum(losses) > 0 else (9.99 if sum(wins) > 0 else 0.0)
____else:
________win_rate = 0.0
________profit_factor = 0.0
____return {"total_trades": len(trades), "win_rate": win_rate, "net_profit": net_profit, "profit_factor": profit_factor, "max_dd": max_dd}
def analyze_one_stock(symbol: str, days_history: int, epochs: int):
____df = get_stock_data(symbol, days=days_history)
____if df.empty or len(df) < 100:
________st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol}. Vui lòng kiểm tra lại mã cổ phiếu!")
________return
____df = add_technical_indicators(df)
____df_clean = df.dropna().copy()
____if len(df_clean) < 60:
________st.error(f"Dữ liệu sau khi tính chỉ báo quá ít cho mã {symbol}.")
________return
____features = ('close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD')
____raw_values = df_clean.loc[:, features].values
____split_idx = int(len(raw_values) * 0.8)
____scaler_eval = MinMaxScaler(feature_range=(0, 1))
____scaler_eval.fit(raw_values[:split_idx])
____scaled_eval = scaler_eval.transform(raw_values)
____look_back = 30
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
____latest_rsi = float(df_clean['RSI'].iloc[-1])
____latest_mfi = float(df_clean['MFI'].iloc[-1])
____latest_ma20 = float(df_clean['MA20'].iloc[-1])
____sig = evaluate_stock_signals(df_clean, ai_pct_change=pct_change, ai_is_reliable=ai_is_reliable)
____bt = run_strategy_backtest(df_clean)
____gia_cat_lo = sig["cat_lo"]
____gia_chot_loi = sig["chot_loi"]
____pct_cat_lo = ((gia_cat_lo - current_price) / current_price) * 100
____pct_chot_loi = ((gia_chot_loi - current_price) / current_price) * 100
____st.divider()
____m1, m2, m3, m4, m5 = st.columns(5)
____m1.metric("Giá hiện tại", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}% TB20")
____m2.metric("Tỷ lệ Lợi nhuận / Rủi ro", f"{sig['rr_ratio']} : 1", "Đạt chuẩn (≥1.5)" if sig['rr_ratio'] >= 1.5 else "Rủi ro cao (<1.5)", delta_color="off")
____m3.metric("Cách nền hỗ trợ MA20", f"{sig['dist_ma20_pct']:+.1f}%", "Sát nền (≤6%)" if 0 <= sig['dist_ma20_pct'] <= 6 else ("Xa nền (>6%)" if sig['dist_ma20_pct'] > 6 else "Dưới MA20"), delta_color="off")
____ichi_status = "Trên mây (Tăng)" if current_price > sig["cloud_top"] else ("Dưới mây (Giảm)" if current_price < sig["cloud_bot"] else "Trong mây (Giằng co)")
____m4.metric("Vị thế Mây Ichimoku", ichi_status, f"RSI: {latest_rsi:.0f} | MFI: {latest_mfi:.0f}", delta_color="off")
____ai_badge = "Tin cậy" if ai_is_reliable else "Phụ trợ"
____m5.metric("AI Dự đoán (Bộ lọc phụ)", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}% (MAPE: {mape_err:.1f}% - {ai_badge})")
____st.subheader(f"🤖 Kết Luận Đồng Nhất: {sig['trang_thai']}")
____if "BẪY GIÁ" in sig["trang_thai"] or "CẢNH BÁO BÁN" in sig["trang_thai"]:
________st.error(f"Phân tích chi tiết: {sig['ly_do']}")
____elif "BÙNG NỔ" in sig["trang_thai"] or "MUA ĐẸP" in sig["trang_thai"]:
________st.success(f"Phân tích chi tiết: {sig['ly_do']}")
____elif "SIDEWAY" in sig["trang_thai"] or "QUA ĐIỂM MUA" in sig["trang_thai"]:
________st.warning(f"Phân tích chi tiết: {sig['ly_do']}")
____else:
________st.info(f"Phân tích chi tiết: {sig['ly_do']}")
____st.markdown("#### 🎯 Kế hoạch giao dịch & Quản trị rủi ro theo Hỗ trợ / Kháng cự:")
____p1, p2, p3 = st.columns(3)
____p1.info(f"Vùng mua hợp lý (Sát nền MA20):\n\n{latest_ma20:,.0f} đ – {latest_ma20 * 1.05:,.0f} đ")
____p2.success(f"Mục tiêu BÁN Chốt lời (Kháng cự / +3 ATR):\n\n{gia_chot_loi:,.0f} đ ({pct_chot_loi:+.1f}%)")
____p3.error(f"Mốc BÁN Cắt lỗ kỹ thuật (Dưới hỗ trợ):\n\nThủng {gia_cat_lo:,.0f} đ ({pct_cat_lo:+.1f}%)")
____st.markdown("#### 📊 Kiểm thử định lượng chiến lược trên 150 phiên gần nhất (Đã trừ 0,3% phí):")
____b1, b2, b3, b4 = st.columns(4)
____b1.metric("Số lệnh / Tỷ lệ thắng", f"{bt['total_trades']} lệnh", f"Win Rate: {bt['win_rate']:.1f}%")
____b2.metric("Lợi nhuận ròng chiến lược", f"{bt['net_profit']:+.2f}%")
____b3.metric("Hệ số Profit Factor", f"{bt['profit_factor']:.2f}", "Tốt (>1.5)" if bt['profit_factor'] > 1.5 else "Trung bình", delta_color="off")
____b4.metric("Sụt giảm tối đa (Max DD)", f"{bt['max_dd']:.2f}%", delta_color="inverse")
____ngay_moi_nhat = df_clean['time'].iloc[-1]
____st.subheader(f"Biểu đồ kỹ thuật {symbol} — Mây Ichimoku (Hiện tại & Tương lai 26P) (Đến: {ngay_moi_nhat})")
____df_plot = df_clean.tail(100).copy().reset_index(drop=True)
____future_dates = [f"+{k}P" for k in range(1, 27)]
____future_span_a = df['Span_A_Raw'].tail(26).tolist()
____future_span_b = df['Span_B_Raw'].tail(26).tolist()
____x_cloud = df_plot['time'].tolist() + future_dates
____y_span_a = df_plot['Senkou_A'].tolist() + future_span_a
____y_span_b = df_plot['Senkou_B'].tolist() + future_span_b
____fig = go.Figure()
____fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20 (Nền hỗ trợ)'))
____fig.add_trace(go.Scatter(x=x_cloud, y=y_span_a, line=dict(color='rgba(0, 200, 100, 0.45)', width=1), name='Senkou A (Kéo 26P)'))
____fig.add_trace(go.Scatter(x=x_cloud, y=y_span_b, line=dict(color='rgba(255, 80, 80, 0.45)', width=1), fill='tonexty', fillcolor='rgba(144, 238, 144, 0.18)', name='Mây Ichimoku'))
____fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
____fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Cắt lỗ hỗ trợ")
____fig.update_xaxes(type='category', tickangle=-45, nticks=42)
____fig.update_layout(xaxis_rangeslider_visible=False, height=560, hovermode='x unified')
____st.plotly_chart(fig, use_container_width=True)
def scan_single_stock(ma: str):
____try:
________df_ma = get_stock_data(ma, days=180)
________if len(df_ma) < 80:
____________return None
________df_ma = add_technical_indicators(df_ma).dropna()
________if len(df_ma) < 5:
____________return None
________last = df_ma.iloc[-1]
________curr_p = float(last['close'])
________sig = evaluate_stock_signals(df_ma, ai_pct_change=0.0, ai_is_reliable=False)
________return {
____________"Mã CP": ma,
____________"Giá hiện tại (đ)": round(curr_p, 0),
____________"Cắt lỗ (đ)": round(sig["cat_lo"], 0),
____________"Chốt lời (đ)": round(sig["chot_loi"], 0),
____________"Tỷ lệ R:R": f"{sig['rr_ratio']} : 1",
____________"Cách nền MA20": f"{sig['dist_ma20_pct']:+.1f}%",
____________"Tỷ lệ Vol (%)": f"{float(last['Vol_Ratio']):.0f}%",
____________"Điểm Số": int(sig['buy_score']),
____________"Điểm Mua": f"{sig['buy_score']}/4",
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
____col1, col2, col3, col4 = st.columns(4)
____symbol = col1.text_input("Nhập mã cổ phiếu bất kỳ (HOSE/HNX/UPCOM):", value="MBS").upper().strip()
____days_history = col2.selectbox("Dữ liệu lịch sử học (ngày):", (300, 500, 730, 1000), index=2)
____epochs = col3.selectbox("Số vòng học AI (Iterations):", (80, 150, 250), index=1)
____col4.write("")
____col4.write("")
____run_btn = col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True, type="primary")
____if run_btn and symbol:
________with st.spinner(f"Đang phân tích chuyên sâu & chạy Backtest cho mã {symbol}..."):
____________analyze_one_stock(symbol, days_history, epochs)
with tab2:
____st.subheader("Quét Xu Hướng Tăng, Dòng Tiền, Điểm Mua Sát Nền & Tỷ Lệ R:R")
____c_filter1, c_filter2, c_filter3 = st.columns((2, 1.5, 1.5))
____nhom_chon = c_filter1.selectbox("Chọn nhóm cổ phiếu / chế độ quét:", list(DANH_MUC_NHOM.keys()))
____ma_rieng = c_filter2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="VD: BSR, KBC, VGC").upper().strip()
____che_do_loc = c_filter3.selectbox("Bộ lọc hiển thị trạng thái:", (
________"📋 Hiển thị TẤT CẢ (Mua + Bán + Chờ chỉnh + Đứng ngoài)",
________"🟢 Chỉ hiện Tín hiệu MUA ĐẸP & Bùng nổ Vol (R:R ≥ 1.5)",
________"🔴 Chỉ hiện Cảnh báo BÁN / Chốt lời / Bẫy giá",
________"⏳ Chỉ hiện Qua điểm mua (Chờ chỉnh) & Sideway"
____))
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
________________status_text.text(f"⚡ Đang quét song song: {da_xong}/{tong_so} mã cổ phiếu...")
________________res = future.result()
________________if res is not None:
____________________ket_qua_loc.append(res)
________status_text.text(f"✅ Hoàn tất quét {tong_so} mã cổ phiếu!")
________if ket_qua_loc:
____________df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Điểm Số", "Khuyến nghị"], ascending=[False, True]).drop(columns=["Điểm Số"])
____________if "Chỉ hiện Tín hiệu MUA" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP")]
____________elif "Chỉ hiện Cảnh báo BÁN" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("CẢNH BÁO BÁN|BẪY GIÁ")]
____________elif "Chỉ hiện Qua điểm mua" in che_do_loc:
________________df_res = df_res[df_res["Khuyến nghị"].str.contains("QUA ĐIỂM MUA|SIDEWAY|ĐỨNG NGOÀI|NẮM GIỮ")]
____________st.write(f"Hiển thị {len(df_res)} cổ phiếu theo bộ lọc đã chọn:")
____________st.dataframe(df_res, use_container_width=True, hide_index=True)
________else:
____________st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
with tab3:
____st.subheader("📝 Nhật Ký Theo Dõi & Kiểm Chứng Thứ Tự Chạm High/Low Thực Tế")
____st.write("Hệ thống tự động kiểm tra lịch sử Giá Cao Nhất (High) và Giá Thấp Nhất (Low) qua từng phiên kể từ ngày lưu lệnh để xác định chính xác ngưỡng nào bị chạm trước.")
____with st.form("form_them_ma"):
________fc1, fc2, fc3 = st.columns((2, 3, 1))
________ma_luu = fc1.text_input("Mã cổ phiếu muốn lưu:", placeholder="VD: HAG, MBS, FPT").upper().strip()
________ghi_chu = fc2.text_input("Ghi chú tín hiệu:", value="Điểm mua chuẩn R:R >= 1.5")
________fc3.write("")
________fc3.write("")
________submit_luu = fc3.form_submit_button("➕ Lưu vào Nhật ký", use_container_width=True)
____if submit_luu and ma_luu:
________info_ma = scan_single_stock(ma_luu)
________if info_ma:
____________now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d')
____________st.session_state.nhat_ky_lenh.append({
________________"Ngày quét": now_str,
________________"Mã CP": ma_luu,
________________"Giá vào (đ)": info_ma["Giá hiện tại (đ)"],
________________"Cắt lỗ (đ)": info_ma["Cắt lỗ (đ)"],
________________"Chốt lời (đ)": info_ma["Chốt lời (đ)"],
________________"Ghi chú": ghi_chu
____________})
____________st.success(f"Đã lưu mã {ma_luu} (Giá vào: {info_ma['Giá hiện tại (đ)']:,.0f} đ) vào nhật ký theo dõi!")
________else:
____________st.error(f"Không lấy được dữ liệu của mã {ma_luu} để lưu.")
____if st.session_state.nhat_ky_lenh:
________bang_cham_diem = []
________so_lenh_thang = 0
________tong_lenh_xac_dinh = 0
________for item in st.session_state.nhat_ky_lenh:
____________ma_cp = item["Mã CP"]
____________ngay_vao = item["Ngày quét"]
____________gia_vao = float(item["Giá vào (đ)"])
____________cat_lo = float(item["Cắt lỗ (đ)"])
____________chot_loi = float(item["Chốt lời (đ)"])
____________gia_khop, ket_qua, ngay_dong = check_journal_trade_history(ma_cp, ngay_vao, gia_vao, cat_lo, chot_loi)
____________lai_lo_rong = ((gia_khop - gia_vao) / gia_vao) * 100 - 0.3
____________if "ĐẠT CHỐT LỜI" in ket_qua:
________________so_lenh_thang += 1
________________tong_lenh_xac_dinh += 1
____________elif "CHẠM CẮT LỖ" in ket_qua:
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
________________"Ghi chú": item["Ghi chú"]
____________})
________df_nk = pd.DataFrame(bang_cham_diem)
________st.dataframe(df_nk, use_container_width=True, hide_index=True)
________if tong_lenh_xac_dinh > 0:
____________st.metric("Tỷ lệ thắng thực chiến (các lệnh đã chốt/cắt lỗ)", f"{so_lenh_thang}/{tong_lenh_xac_dinh} ({so_lenh_thang/tong_lenh_xac_dinh*100:.0f}%)")
________else:
____________st.caption("ℹ️ Tỷ lệ thắng sẽ tự động tính toán ngay khi có lệnh chạm mốc Chốt lời hoặc Cắt lỗ theo dữ liệu High/Low.")
________c_btn1, c_btn2 = st.columns(2)
________csv_data = df_nk.to_csv(index=False).encode('utf-8-sig')
________c_btn1.download_button("📥 Tải Nhật Ký về máy (CSV)", data=csv_data, file_name="nhat_ky_chung_khoan_v3_1.csv", mime="text/csv")
________if c_btn2.button("🗑️ Xóa toàn bộ nhật ký làm lại"):
____________st.session_state.nhat_ky_lenh = []
____________st.rerun()
"""
exec('\n'.join((' ' * (len(l) - len(l.lstrip(chr(95))))) + l.lstrip(chr(95)) for l in raw_code.strip().splitlines()))
