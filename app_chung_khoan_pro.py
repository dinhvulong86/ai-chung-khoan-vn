raw_code = r"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (Ultimate)", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích Kỹ Thuật & Quản Trị Danh Mục VN")
st.markdown("Tích hợp **Mây Ichimoku**, **Cắt lỗ động ATR**, **Đột biến Volume**, **Quét đa luồng** & **Nhật ký chấm điểm thực tế**.")

if "nhat_ky_lenh" not in st.session_state:
____st.session_state.nhat_ky_lenh = [
________{"Ngày quét": "25/11/2026", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ ATR (đ)": 13314, "Chốt lời ATR (đ)": 14654, "Ghi chú": "Mẫu theo dõi Bùng nổ Vol"}
____]

@st.cache_data(ttl=60)
def get_stock_data(symbol, days=500):
____now_vn = datetime.utcnow() + timedelta(hours=7)
____today_str = now_vn.strftime('%Y-%m-%d')
____end_ts = int(now_vn.timestamp())
____headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept': 'application/json'}
____df = pd.DataFrame()
____try:
________count_back = min(int(days * 0.72), 1000)
________url_tcbs = "https://tcbs.com.vn"
________params_tcbs = {"ticker": symbol, "type": "stock", "resolution": "D", "to": end_ts, "countBack": count_back}
________res = requests.get(url_tcbs, params=params_tcbs, headers=headers, timeout=8)
________data = res.json().get('data', [])
________if data and len(data) >= 30:
____________df = pd.DataFrame(data)
____________df['time'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
____________df = df.loc[:, ('time', 'open', 'high', 'low', 'close', 'volume')].copy()
____________for col in ['open', 'high', 'low', 'close']:
________________if df[col].iloc[-1] < 1000:
____________________df[col] = df[col] * 1000
____________df = df.dropna().sort_values('time').reset_index(drop=True)
____except Exception:
________pass

____if df.empty:
________try:
____________start_date = (now_vn - timedelta(days=days)).strftime('%Y-%m-%d')
____________ticker_vn = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
____________df_yf = yf.download(ticker_vn, start=start_date, end=today_str, progress=False)
____________if not df_yf.empty:
________________if isinstance(df_yf.columns, pd.MultiIndex):
____________________df_yf.columns = df_yf.columns.get_level_values(0)
________________df_yf = df_yf.reset_index()
________________df_yf = df_yf.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
________________df_yf.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
________________df_yf['time'] = pd.to_datetime(df_yf['time']).dt.strftime('%Y-%m-%d')
________________df = df_yf.dropna().sort_values('time').reset_index(drop=True)
________except Exception:
____________pass

____if not df.empty and df['time'].iloc[-1] != today_str:
________try:
____________url_intra = f"https://tcbs.com.vn{symbol}/investor/his/paging"
____________res_intra = requests.get(url_intra, params={"page": 0, "size": 500}, headers=headers, timeout=6)
____________intra_data = res_intra.json().get('data', [])
____________if intra_data:
________________prices = [float(x['p']) * (1000 if float(x['p']) < 1000 else 1) for x in intra_data if x.get('p')]
________________vols = sum(float(x['v']) for x in intra_data if x.get('v'))
________________if prices:
____________________latest_p = prices.pop(0)
____________________open_p = prices[-1] if prices else latest_p
____________________high_p = max(prices + [latest_p])
____________________low_p = min(prices + [latest_p])
____________________new_row = pd.DataFrame([{'time': today_str, 'open': open_p, 'high': high_p, 'low': low_p, 'close': latest_p, 'volume': vols}])
____________________df = pd.concat([df, new_row], ignore_index=True)
________except Exception:
____________pass

____return df

@st.cache_data(ttl=600)
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

def add_technical_indicators(df):
____df['MA20'] = df['close'].rolling(window=20).mean()
____df['MA50'] = df['close'].rolling(window=50).mean()
____df['STD20'] = df['close'].rolling(window=20).std()
____df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
____df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
____df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
____df['Vol_Ratio'] = (df['volume'] / df['Vol_MA20'].replace(0, np.nan)) * 100
____tr1 = df['high'] - df['low']
____tr2 = (df['high'] - df['close'].shift(1)).abs()
____tr3 = (df['low'] - df['close'].shift(1)).abs()
____df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
____df['ATR'] = df['TR'].rolling(window=14).mean()
____high_9 = df['high'].rolling(window=9).max()
____low_9 = df['low'].rolling(window=9).min()
____df['Tenkan'] = (high_9 + low_9) / 2
____high_26 = df['high'].rolling(window=26).max()
____low_26 = df['low'].rolling(window=26).min()
____df['Kijun'] = (high_26 + low_26) / 2
____df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
____high_52 = df['high'].rolling(window=52).max()
____low_52 = df['low'].rolling(window=52).min()
____df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
____df['High_20'] = df['high'].rolling(window=20).max()
____df['Low_20'] = df['low'].rolling(window=20).min()
____df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20']) * 100
____delta = df['close'].diff()
____gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
____loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
____rs = gain / loss
____df['RSI'] = 100 - (100 / (1 + rs))
____tp = (df['high'] + df['low'] + df['close']) / 3
____raw_mf = tp * df['volume']
____pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
____neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
____mfr = pd.Series(pos_mf).rolling(window=14).sum() / pd.Series(neg_mf).rolling(window=14).sum().replace(0, np.nan)
____df['MFI'] = 100 - (100 / (1 + mfr))
____exp1 = df['close'].ewm(span=12, adjust=False).mean()
____exp2 = df['close'].ewm(span=26, adjust=False).mean()
____df['MACD'] = exp1 - exp2
____df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
____return df

def analyze_one_stock(symbol, days_history, epochs):
____df = get_stock_data(symbol, days=days_history)
____if df.empty or len(df) < 100:
________st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol}. Vui lòng kiểm tra lại mã cổ phiếu!")
________return
____df = add_technical_indicators(df)
____df_clean = df.dropna().copy()
____features = ('close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD')
____data = df_clean.loc[:, features].values
____scaler = MinMaxScaler(feature_range=(0, 1))
____scaled_data = scaler.fit_transform(data)
____look_back = 30
____X = np.array([scaled_data[i-look_back:i].flatten() for i in range(look_back, len(scaled_data))])
____y = np.array([scaled_data[i, 0] for i in range(look_back, len(scaled_data))])
____model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
____model.fit(X, y)
____last_30_days = scaled_data[-look_back:].flatten().reshape(1, -1)
____pred_scaled = model.predict(last_30_days).item()
____dummy_array = np.zeros((1, len(features)))
____dummy_array.put(0, pred_scaled)
____predicted_price = scaler.inverse_transform(dummy_array).item(0)

____current_price = float(df_clean['close'].iloc[-1])
____prev_close = float(df_clean['close'].iloc[-2])
____vol_ratio = float(df_clean['Vol_Ratio'].iloc[-1])
____latest_atr = float(df_clean['ATR'].iloc[-1])
____pct_change = ((predicted_price - current_price) / current_price) * 100
____latest_rsi = float(df_clean['RSI'].iloc[-1])
____latest_mfi = float(df_clean['MFI'].iloc[-1])
____latest_macd = float(df_clean['MACD'].iloc[-1])
____latest_sig = float(df_clean['Signal_Line'].iloc[-1])
____latest_ma20 = float(df_clean['MA20'].iloc[-1])
____bb_upper = float(df_clean['BB_Upper'].iloc[-1])
____bb_lower = float(df_clean['BB_Lower'].iloc[-1])
____senkou_a = float(df_clean['Senkou_A'].iloc[-1])
____senkou_b = float(df_clean['Senkou_B'].iloc[-1])
____cloud_top = max(senkou_a, senkou_b)
____cloud_bot = min(senkou_a, senkou_b)
____range_20 = float(df_clean['Range_20_Pct'].iloc[-1])
____high_20 = float(df_clean['High_20'].iloc[-1])
____low_20 = float(df_clean['Low_20'].iloc[-1])

____is_sideway = (range_20 < 9.5) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
____is_bull_trap = (current_price > prev_close * 1.02) and (vol_ratio < 75.0) and (latest_rsi > 65)
____is_vol_breakout = (current_price > prev_close * 1.015) and (vol_ratio >= 150.0) and (current_price > latest_ma20)

____gia_cat_lo_atr = current_price - (2.0 * latest_atr)
____gia_chot_loi_atr = current_price + (3.0 * latest_atr)
____pct_cat_lo = ((gia_cat_lo_atr - current_price) / current_price) * 100
____pct_chot_loi = ((gia_chot_loi_atr - current_price) / current_price) * 100

____st.divider()
____m1, m2, m3, m4, m5 = st.columns(5)
____m1.metric("Giá hiện tại", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}% TB20")
____m2.metric("AI Dự đoán tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
____m3.metric("Chỉ số RSI / MFI", f"{latest_rsi:.0f} / {latest_mfi:.0f}", "Quá mua" if latest_rsi > 70 else ("Quá bán" if latest_rsi < 30 else "Vùng an toàn"), delta_color="off")
____ichi_status = "Trên mây (Tăng)" if current_price > cloud_top else ("Dưới mây (Giảm)" if current_price < cloud_bot else "Trong mây (Giằng co)")
____m4.metric("Vị thế Mây Ichimoku", ichi_status, f"Mây: {cloud_bot:,.0f} - {cloud_top:,.0f}", delta_color="off")
____m5.metric("Biên độ ATR (14)", f"{latest_atr:,.0f} đ", f"±{(latest_atr/current_price)*100:.1f}% / phiên", delta_color="off")
____st.subheader("🤖 Phân Tích Chuyên Sâu & Khuyến Nghị Hành Động")
____if is_vol_breakout:
________st.success(f"🔥 BÙNG NỔ KHỐI LƯỢNG (VOLUME BREAKOUT): Khối lượng giao dịch đạt {vol_ratio:.0f}% mức trung bình 20 phiên kèm giá bứt phá mạnh! Dòng tiền lớn đang nhập cuộc.")
____if is_bull_trap:
________st.error(f"🚨 CẢNH BÁO BẪY TĂNG GIÁ (BULL TRAP): Giá tăng nhưng khối lượng chỉ đạt {vol_ratio:.0f}% trung bình kèm RSI cao. Tuyệt đối không mua đuổi!")
____diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + (1 if current_price > cloud_top else 0) + (1 if (current_price > latest_ma20 and vol_ratio >= 100) else 0)
____diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + (2 if (latest_rsi > 70 or latest_mfi > 80) else 0) + (1 if current_price < cloud_bot else 0)
____if is_sideway:
________st.warning(f"⏸️ THỊ TRƯỜNG ĐI NGANG (SIDEWAY): Biên độ 20 phiên nén chặt {range_20:.1f}% (trong hộp {low_20:,.0f} đ – {high_20:,.0f} đ). Ưu tiên mua gần biên dưới Bollinger ({bb_lower:,.0f} đ) và chốt lời gần biên trên ({bb_upper:,.0f} đ).")
____elif diem_ban >= 3:
________st.error(f"🔴 KHUYẾN NGHỊ: BÁN HẠ TỶ TRỌNG / CHỐT LỜI NGAY (Điểm rủi ro BÁN: {diem_ban}/5)")
____elif diem_mua >= 3 and diem_ban <= 1:
________st.success(f"🟢 KHUYẾN NGHỊ: CÂN NHẮC MUA MỚI (Điểm ủng hộ MUA: {diem_mua}/4 - Xu hướng & Mây Ichimoku ủng hộ)")
____elif diem_ban == 2:
________st.warning(f"🟠 KHUYẾN NGHỊ: CANH BÁN CHỐT LỜI TỪNG PHẦN / KHÔNG MUA ĐUỔI (Có {diem_ban} cảnh báo bán)")
____else:
________st.info("🟡 KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ & QUAN SÁT (Cung - Cầu đang cân bằng)")
____st.markdown("#### 🎯 Kế hoạch giao dịch & Quản trị rủi ro động theo ATR(14):")
____p1, p2, p3 = st.columns(3)
____p1.info(f"Vùng giá Mua an toàn (MA20 / BB Dưới):\n\n{bb_lower:,.0f} đ – {latest_ma20:,.0f} đ")
____p2.success(f"Mục tiêu BÁN Chốt lời động (+3 ATR):\n\n{gia_chot_loi_atr:,.0f} đ ({pct_chot_loi:+.1f}%)")
____p3.error(f"Mốc BÁN Cắt lỗ động (-2 ATR):\n\nThủng {gia_cat_lo_atr:,.0f} đ ({pct_cat_lo:+.1f}%)")
____ngay_moi_nhat = df_clean['time'].iloc[-1]
____st.subheader(f"Biểu đồ kỹ thuật {symbol} - Mây Ichimoku & Bollinger Bands (Đến ngày: {ngay_moi_nhat})")
____df_plot = df_clean.tail(100)
____fig = go.Figure()
____fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['Senkou_A'], line=dict(color='rgba(0, 200, 100, 0.4)', width=1), name='Mây Senkou A'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['Senkou_B'], line=dict(color='rgba(255, 80, 80, 0.4)', width=1), fill='tonexty', fillcolor='rgba(144, 238, 144, 0.2)', name='Mây Ichimoku'))
____fig.add_hline(y=gia_chot_loi_atr, line_dash="dot", line_color="green", annotation_text="Chốt lời (+3 ATR)")
____fig.add_hline(y=gia_cat_lo_atr, line_dash="dot", line_color="red", annotation_text="Cắt lỗ (-2 ATR)")
____fig.update_xaxes(type='category', tickangle=-45, nticks=45)
____fig.update_layout(xaxis_rangeslider_visible=False, height=540, hovermode='x unified')
____st.plotly_chart(fig, use_container_width=True)
def scan_single_stock(ma):
____try:
________df_ma = get_stock_data(ma, days=160)
________if len(df_ma) < 80:
____________return None
________df_ma = add_technical_indicators(df_ma).dropna()
________last = df_ma.iloc[-1]
________prev = df_ma.iloc[-2]
________curr_p = float(last['close'])
________prev_p = float(prev['close'])
________vol_ratio = float(last['Vol_Ratio'])
________atr_val = float(last['ATR'])
________cloud_top = max(float(last['Senkou_A']), float(last['Senkou_B']))
________cloud_bot = min(float(last['Senkou_A']), float(last['Senkou_B']))
________is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= 150.0) and (curr_p > float(last['MA20']))
________buy_score = (1 if last['MACD'] > last['Signal_Line'] else 0) + (1 if curr_p > float(last['MA20']) else 0) + (1 if curr_p > cloud_top else 0) + (1 if vol_ratio >= 110.0 else 0)
________sell_score = (1 if last['MACD'] < last['Signal_Line'] else 0) + (1 if curr_p < cloud_bot else 0) + (1 if (last['RSI'] > 70 or last['MFI'] > 80) else 0)
________range_20 = float(last['Range_20_Pct'])
________is_sideway = (range_20 < 9.5) and (abs(curr_p - float(last['MA20'])) / float(last['MA20']) < 0.025)
________if is_vol_breakout:
____________trang_thai = "🔥 BÙNG NỔ VOL + MUA"
________elif is_sideway:
____________trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
________elif buy_score >= 3 and sell_score == 0:
____________trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
________elif sell_score >= 2 or last['RSI'] > 72 or last['MFI'] > 80:
____________trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
________elif buy_score == 2:
____________trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
________else:
____________trang_thai = "⚪ ĐỨNG NGOÀI"
________return {
____________"Mã CP": ma,
____________"Giá hiện tại (đ)": round(curr_p, 0),
____________"Cắt lỗ ATR (đ)": round(curr_p - 2 * atr_val, 0),
____________"Chốt lời ATR (đ)": round(curr_p + 3 * atr_val, 0),
____________"Tỷ lệ Vol (%)": f"{vol_ratio:.0f}%",
____________"RSI / MFI": f"{float(last['RSI']):.0f} / {float(last['MFI']):.0f}",
____________"Điểm Mua": f"{buy_score}/4",
____________"Khuyến nghị": trang_thai
________}
____except Exception:
________return None
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
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Chuyên Sâu", "🔍 Quét Đột Biến Dòng Tiền", "📝 Nhật Ký & Chấm Điểm Tín Hiệu"])
with tab1:
____col1, col2, col3, col4 = st.columns(4)
____symbol = col1.text_input("Nhập mã cổ phiếu bất kỳ (HOSE/HNX/UPCOM):", value="MBS").upper().strip()
____days_history = col2.selectbox("Dữ liệu lịch sử học (ngày):", (365, 500, 730, 1000), index=2)
____epochs = col3.selectbox("Số vòng học AI (Iterations):", (100, 200, 300), index=1)
____col4.write("")
____col4.write("")
____run_btn = col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True)
if run_btn:
____analyze_one_stock(symbol, days_history, epochs)
with tab2:
____st.subheader("Quét Đột Biến Khối Lượng (Vol Breakout), Mây Ichimoku & Cắt Lỗ Động ATR")
____c_filter1, c_filter2 = st.columns(2)
____nhom_chon = c_filter1.selectbox("Chọn nhóm cổ phiếu / chế độ quét:", list(DANH_MUC_NHOM.keys()))
____ma_rieng = c_filter2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="VD: BSR, KBC, VGC").upper().strip()
____chi_hien_mua = st.checkbox("Chỉ lọc hiển thị các mã đang có 🔥 BÙNG NỔ VOL, 🟢 MUA ĐẸP hoặc ⏸️ SIDEWAY")
____scan_btn = st.button("⚡ Bắt Đầu Quét Siêu Tốc Ngay")
if scan_btn:
____nhom_val = DANH_MUC_NHOM[nhom_chon]
____if nhom_val == "DYNAMIC":
________ds_quet = get_dynamic_top_tickers(DEFAULT_TOP_125)
____else:
________ds_quet = list(nhom_val)
____if ma_rieng:
________for m in ma_rieng.split(','):
____________m_clean = m.strip()
____________if m_clean and m_clean not in ds_quet:
________________ds_quet.insert(0, m_clean)
____progress_bar = st.progress(0)
____status_text = st.empty()
____ket_qua_loc = []
____tong_so = len(ds_quet)
____da_xong = 0
____with ThreadPoolExecutor(max_workers=10) as executor:
________future_to_ma = {executor.submit(scan_single_stock, ma): ma for ma in ds_quet}
________for future in as_completed(future_to_ma):
____________da_xong += 1
____________progress_bar.progress(da_xong / tong_so)
____________status_text.text(f"⚡ Đang quét song song: {da_xong}/{tong_so} mã cổ phiếu...")
____________res = future.result()
____________if res is not None:
________________ket_qua_loc.append(res)
____status_text.text(f"✅ Hoàn tất quét {tong_so} mã cổ phiếu!")
____if ket_qua_loc:
________df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
________if chi_hien_mua:
____________df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP|SIDEWAY")]
________st.write(f"Tìm thấy {len(df_res)} cổ phiếu phù hợp điều kiện:")
________st.dataframe(df_res, use_container_width=True, hide_index=True)
____else:
________st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
with tab3:
____st.subheader("📝 Nhật Ký Theo Dõi & Tự Động Chấm Điểm Hiệu Quả Phần Mềm")
____st.write("Lưu các mã phần mềm báo Mua đẹp / Bùng nổ Vol vào đây để hệ thống tự động đối chiếu giá thị trường hàng ngày và tính lãi/lỗ thực tế (đã trừ sẵn 0,3% thuế phí MBS).")
____with st.form("form_them_ma"):
________fc1, fc2, fc3 = st.columns()
________ma_luu = fc1.text_input("Mã cổ phiếu muốn lưu:", placeholder="VD: HAG, MBS, FPT").upper().strip()
________ghi_chu = fc2.text_input("Ghi chú tín hiệu:", value="Bùng nổ Vol + Mua")
________fc3.write("")
________fc3.write("")
________submit_luu = fc3.form_submit_button("➕ Lưu vào Nhật ký", use_container_width=True)
____if submit_luu and ma_luu:
________info_ma = scan_single_stock(ma_luu)
________if info_ma:
____________now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%d/%m/%Y')
____________st.session_state.nhat_ky_lenh.append({
________________"Ngày quét": now_str,
________________"Mã CP": ma_luu,
________________"Giá vào (đ)": info_ma["Giá hiện tại (đ)"],
________________"Cắt lỗ ATR (đ)": info_ma["Cắt lỗ ATR (đ)"],
________________"Chốt lời ATR (đ)": info_ma["Chốt lời ATR (đ)"],
________________"Ghi chú": ghi_chu
____________})
____________st.success(f"Đã lưu mã {ma_luu} (Giá vào: {info_ma['Giá hiện tại (đ)']:,.0f} đ) vào nhật ký theo dõi!")
________else:
____________st.error(f"Không lấy được dữ liệu của mã {ma_luu} để lưu.")
____if st.session_state.nhat_ky_lenh:
________bang_cham_diem = []
________so_lenh_thang = 0
________tong_lenh_da_dong = 0
________for item in st.session_state.nhat_ky_lenh:
____________ma_cp = item["Mã CP"]
____________gia_vao = float(item["Giá vào (đ)"])
____________cat_lo = float(item["Cắt lỗ ATR (đ)"])
____________chot_loi = float(item["Chốt lời ATR (đ)"])
____________df_now = get_stock_data(ma_cp, days=30)
____________gia_tt = float(df_now['close'].iloc[-1]) if not df_now.empty else gia_vao
____________# Tính lãi lỗ ròng sau khi trừ 0.3% thuế và phí giao dịch MBS
____________lai_lo_rong = ((gia_tt - gia_vao) / gia_vao) * 100 - 0.3
____________if gia_tt >= chot_loi:
________________ket_qua = "🎉 ĐẠT CHỐT LỜI ATR"
________________so_lenh_thang += 1
________________tong_lenh_da_dong += 1
____________elif gia_tt <= cat_lo:
________________ket_qua = "🛑 CHẠM CẮT LỖ ATR"
________________tong_lenh_da_dong += 1
____________else:
________________ket_qua = "⏳ ĐANG NẮM GIỮ"
____________bang_cham_diem.append({
________________"Ngày quét": item["Ngày quét"],
________________"Mã CP": ma_cp,
________________"Giá vào (đ)": round(gia_vao, 0),
________________"Giá thị trường (đ)": round(gia_tt, 0),
________________"Cắt lỗ ATR (đ)": round(cat_lo, 0),
________________"Chốt lời ATR (đ)": round(chot_loi, 0),
________________"Lãi/Lỗ ròng (%)": f"{lai_lo_rong:+.2f}%",
________________"Trạng thái": ket_qua,
________________"Ghi chú": item["Ghi chú"]
____________})
________df_nk = pd.DataFrame(bang_cham_diem)
________st.dataframe(df_nk, use_container_width=True, hide_index=True)
________c_btn1, c_btn2 = st.columns(2)
________csv_data = df_nk.to_csv(index=False).encode('utf-8-sig')
________c_btn1.download_button("📥 Tải Nhật Ký về máy (File Excel/CSV)", data=csv_data, file_name="nhat_ky_chung_khoan.csv", mime="text/csv")
________if c_btn2.button("🗑️ Xóa toàn bộ nhật ký làm lại"):
____________st.session_state.nhat_ky_lenh = []
____________st.rerun()
"""
exec('\n'.join((' ' * (len(l) - len(l.lstrip(chr(95))))) + l.lstrip(chr(95)) for l in raw_code.strip().splitlines()))
