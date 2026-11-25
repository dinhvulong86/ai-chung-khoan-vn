raw_code = """
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Hệ Thống AI Phân Tích Chứng Khoán VN (Pro)", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích Kỹ Thuật & Quản Trị Danh Mục (Pro)")
st.markdown("Tích hợp **AI Neural Net**, **Bollinger Bands**, **Dòng tiền MFI**, **Cảnh báo Bull/Bear Trap** & **Tỷ số Sharpe**.")

@st.cache_data(ttl=300)
def get_stock_data(symbol, days=500):
____end_ts = int(datetime.now().timestamp())
____headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept': 'application/json'}
____try:
________count_back = min(int(days * 0.72), 1000)
________url_tcbs = "https://tcbs.com.vn"
________params_tcbs = {"ticker": symbol, "type": "stock", "resolution": "D", "to": end_ts, "countBack": count_back}
________res = requests.get(url_tcbs, params=params_tcbs, headers=headers, timeout=10)
________data = res.json().get('data', [])
________if data and len(data) >= 30:
____________df = pd.DataFrame(data)
____________df['time'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
____________df = df.loc[:, ('time', 'open', 'high', 'low', 'close', 'volume')].copy()
____________for col in ['open', 'high', 'low', 'close']:
________________if df[col].iloc[-1] < 1000:
____________________df[col] = df[col] * 1000
____________return df.dropna().sort_values('time').reset_index(drop=True)
____except Exception:
________pass
____try:
________start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
________end_date = datetime.now().strftime('%Y-%m-%d')
________ticker_vn = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
________df = yf.download(ticker_vn, start=start_date, end=end_date, progress=False)
________if not df.empty:
____________if isinstance(df.columns, pd.MultiIndex):
________________df.columns = df.columns.get_level_values(0)
____________df = df.reset_index()
____________df = df.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
____________df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
____________df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
____________return df.dropna().sort_values('time').reset_index(drop=True)
____except Exception:
________pass
____return pd.DataFrame()

def add_technical_indicators(df):
____# 1. Đường trung bình động MA20, MA50 & Dải Bollinger Bands (20, 2)
____df['MA20'] = df['close'].rolling(window=20).mean()
____df['MA50'] = df['close'].rolling(window=50).mean()
____df['STD20'] = df['close'].rolling(window=20).std()
____df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
____df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
____df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['MA20']) * 100
____
____# 2. Trung bình khối lượng 20 phiên (Xác nhận dòng tiền)
____df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
____
____# 3. Biên độ hộp 20 phiên nhận diện Sideway
____df['High_20'] = df['high'].rolling(window=20).max()
____df['Low_20'] = df['low'].rolling(window=20).min()
____df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20']) * 100
____
____# 4. Chỉ báo RSI (14)
____delta = df['close'].diff()
____gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
____loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
____rs = gain / loss
____df['RSI'] = 100 - (100 / (1 + rs))
____
____# 5. Chỉ báo Dòng tiền MFI (Money Flow Index 14)
____tp = (df['high'] + df['low'] + df['close']) / 3
____raw_mf = tp * df['volume']
____pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
____neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
____pos_mf_14 = pd.Series(pos_mf).rolling(window=14).sum()
____neg_mf_14 = pd.Series(neg_mf).rolling(window=14).sum()
____mfr = pos_mf_14 / neg_mf_14.replace(0, np.nan)
____df['MFI'] = 100 - (100 / (1 + mfr))
____
____# 6. Chỉ báo MACD (12, 26, 9)
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
____
____# Đưa cả MFI và Bollinger Width vào cho AI học
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

____# Lấy các thông số phiên mới nhất
____current_price = float(df_clean['close'].iloc[-1])
____prev_close = float(df_clean['close'].iloc[-2])
____current_vol = float(df_clean['volume'].iloc[-1])
____vol_ma20 = float(df_clean['Vol_MA20'].iloc[-1])
____pct_change = ((predicted_price - current_price) / current_price) * 100
____latest_rsi = float(df_clean['RSI'].iloc[-1])
____latest_mfi = float(df_clean['MFI'].iloc[-1])
____latest_macd = float(df_clean['MACD'].iloc[-1])
____latest_sig = float(df_clean['Signal_Line'].iloc[-1])
____latest_ma20 = float(df_clean['MA20'].iloc[-1])
____bb_upper = float(df_clean['BB_Upper'].iloc[-1])
____bb_lower = float(df_clean['BB_Lower'].iloc[-1])
____range_20 = float(df_clean['Range_20_Pct'].iloc[-1])
____high_20 = float(df_clean['High_20'].iloc[-1])
____low_20 = float(df_clean['Low_20'].iloc[-1])

____# Nhận diện Sideway & Bẫy tăng giá (Bull Trap)
____is_sideway = (range_20 < 9.5) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
____is_bull_trap = (current_price > prev_close * 1.02) and (current_vol < vol_ma20 * 0.75) and (latest_rsi > 65)
____
____# Tính toán Tỷ số Sharpe & Độ biến động (Theo lý thuyết Quản lý danh mục)
____daily_rets = df_clean['close'].pct_change().dropna().tail(250)
____ann_return = daily_rets.mean() * 250
____ann_vol = daily_rets.std() * np.sqrt(250)
____sharpe_ratio = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0

____gia_chot_loi = min(current_price * 1.10, bb_upper * 1.02)
____gia_cat_lo = min(current_price * 0.93, latest_ma20 * 0.98)

____st.divider()
____m1, m2, m3, m4, m5 = st.columns(5)
____m1.metric("Giá đóng cửa", f"{current_price:,.0f} đ")
____m2.metric("AI Dự đoán tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
____m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", "Quá mua" if latest_rsi > 70 else ("Quá bán" if latest_rsi < 30 else "Trung tính"), delta_color="off")
____m4.metric("Dòng tiền MFI (14)", f"{latest_mfi:.1f}", "Tiền vào mạnh" if latest_mfi > 80 else ("Rút tiền" if latest_mfi < 20 else "Ổn định"), delta_color="off")
____m5.metric("Tỷ số Sharpe (1Y)", f"{sharpe_ratio:.2f}", "Hấp dẫn (>1)" if sharpe_ratio > 1 else "Rủi ro cao", delta_color="off")

____st.subheader("🤖 Phân Tích Chuyên Sâu & Khuyến Nghị Hành Động")
____if is_bull_trap:
________st.error("🚨 CẢNH BÁO BẪY TĂNG GIÁ (BULL TRAP): Giá tăng nhưng khối lượng giao dịch sụt giảm mạnh dưới trung bình 20 phiên kèm RSI cao. Tuyệt đối không mua đuổi!")
____
____diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + (1 if (latest_rsi < 40 or latest_mfi < 30) else 0) + (1 if (current_price > latest_ma20 and current_vol >= vol_ma20 * 0.9) else 0)
____diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + (2 if (latest_rsi > 70 or latest_mfi > 80) else 0) + (1 if current_price <= latest_ma20 else 0)

____if is_sideway:
________st.warning(f"⏸️ THỊ TRƯỜNG ĐI NGANG (SIDEWAY): Biên độ 20 phiên nén chặt {range_20:.1f}% (trong hộp {low_20:,.0f} đ – {high_20:,.0f} đ). Theo nguyên tắc PTKT, hạn chế dùng MACD; ưu tiên mua gần biên dưới Bollinger ({bb_lower:,.0f} đ) và chốt lời gần biên trên ({bb_upper:,.0f} đ).")
____elif diem_ban >= 3:
________st.error(f"🔴 KHUYẾN NGHỊ: BÁN HẠ TỶ TRỌNG / CHỐT LỜI NGAY (Điểm rủi ro BÁN: {diem_ban}/5)")
____elif diem_mua >= 3 and diem_ban <= 1:
________st.success(f"🟢 KHUYẾN NGHỊ: CÂN NHẮC MUA MỚI (Điểm ủng hộ MUA: {diem_mua}/4 - Có xác nhận dòng tiền)")
____elif diem_ban == 2:
________st.warning(f"🟠 KHUYẾN NGHỊ: CANH BÁN CHỐT LỜI TỪNG PHẦN / KHÔNG MUA ĐUỔI (Có {diem_ban} cảnh báo bán)")
____else:
________st.info("🟡 KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ & QUAN SÁT (Cung - Cầu đang cân bằng)")

____st.markdown("#### 🎯 Kế hoạch giao dịch & Quản trị rủi ro đề xuất:")
____p1, p2, p3 = st.columns(3)
____p1.info(f"**Vùng giá Mua an toàn (Hỗ trợ / BB Dưới):**\\n\\n`{bb_lower:,.0f} đ` – `{latest_ma20:,.0f} đ`")
____p2.success(f"**Mục tiêu BÁN Chốt lời (Kháng cự / BB Trên):**\\n\\n`{gia_chot_loi:,.0f} đ` *(hoặc khi MFI > 80)*")
____p3.error(f"**Mốc BÁN Cắt lỗ bắt buộc (BreakDown):**\\n\\nThủng `{gia_cat_lo:,.0f} đ` *(~ -7% hoặc gãy MA20)*")

____st.subheader("📈 Kiểm thử hiệu quả chiến lược kết hợp RSI + MFI + MACD (100 phiên)")
____df_bt = df_clean.tail(100).copy()
____df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70) & (df_bt['MFI'] < 80), 1, 0)
____df_bt['Daily_Return'] = df_bt['close'].pct_change()
____df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
____total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
____first_close = float(df_bt['close'].head(1).item())
____last_close = float(df_bt['close'].tail(1).item())
____total_hold_return = (last_close / first_close) - 1
____bt1, bt2 = st.columns(2)
____bt1.info(f"Lợi nhuận theo hệ thống (Đã lọc quá mua MFI/RSI): {total_strategy_return*100:+.2f}%")
____bt2.info(f"Lợi nhuận nếu chỉ mua và nắm giữ (Buy & Hold): {total_hold_return*100:+.2f}%")
____st.subheader(f"Biểu đồ kỹ thuật {symbol} & Dải Bollinger Bands (120 phiên)")
____df_plot = df_clean.tail(120)
____fig = go.Figure()
____fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20 (Dải giữa)'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.8)', width=1, dash='dash'), name='Bollinger Trên'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.8)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.15)', name='Bollinger Dưới'))
____fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
____fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Mốc Cắt lỗ")
____fig.update_xaxes(type='category', tickangle=-45, nticks=30)
____fig.update_layout(xaxis_rangeslider_visible=False, height=520, hovermode='x unified')
____st.plotly_chart(fig, use_container_width=True)
def scan_single_stock(ma):
____try:
________df_ma = get_stock_data(ma, days=150)
________if len(df_ma) < 60:
____________return None
________df_ma = add_technical_indicators(df_ma).dropna()
________last = df_ma.iloc[-1]
________prev = df_ma.iloc[-2]
________vol_ok = float(last['volume']) >= float(last['Vol_MA20']) * 0.9
________buy_score = (1 if last['MACD'] > last['Signal_Line'] else 0) + (1 if (last['close'] > last['MA20'] and vol_ok) else 0) + (1 if (30 <= last['RSI'] <= 68 and last['MFI'] < 75) else 0) + (1 if (last['MACD'] > last['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']) else 0)
________sell_score = (1 if last['MACD'] < last['Signal_Line'] else 0) + (1 if last['close'] < last['MA20'] else 0) + (1 if (last['RSI'] > 70 or last['MFI'] > 80) else 0)
________range_20 = float(last['Range_20_Pct'])
________is_sideway = (range_20 < 9.5) and (abs(float(last['close']) - float(last['MA20'])) / float(last['MA20']) < 0.025)
________if is_sideway:
____________trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
________elif buy_score >= 3 and sell_score == 0:
____________trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
________elif sell_score >= 2 or last['RSI'] > 72 or last['MFI'] > 80:
____________trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
________elif buy_score == 2:
____________trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
________else:
____________trang_thai = "⚪ ĐỨNG NGOÀI"
________return {"Mã CP": ma, "Giá hiện tại (đ)": round(float(last['close']), 0), "Hỗ trợ BB Dưới": round(float(last['BB_Lower']), 0), "Kháng cự BB Trên": round(float(last['BB_Upper']), 0), "RSI (14)": round(float(last['RSI']), 1), "MFI (14)": round(float(last['MFI']), 1), "Điểm Mua": f"{buy_score}/4", "Khuyến nghị": trang_thai}
____except Exception:
________return None
tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Điểm Mua/Bán", "🔍 Bộ Lọc Quét Dòng Tiền HOSE & HNX"])
with tab1:
____col1, col2, col3, col4 = st.columns(4)
____symbol = col1.text_input("Nhập mã cổ phiếu HOSE/HNX/UPCOM (VD: CEO, MBS, SHS, FPT):", value="MBS").upper().strip()
____days_history = col2.selectbox("Dữ liệu lịch sử học (ngày):", (365, 500, 730, 1000), index=2)
____epochs = col3.selectbox("Số vòng học AI (Iterations):", (100, 200, 300), index=1)
____col4.write("")
____col4.write("")
____run_btn = col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True)
if run_btn:
____analyze_one_stock(symbol, days_history, epochs)
with tab2:
____st.subheader("Quét tín hiệu Dòng tiền (MFI), Bollinger Bands & Sideway trên rổ HNX & VN30")
____st.write("Tích hợp bộ lọc xác nhận Khối lượng giao dịch (Volume) và Dòng tiền MFI giúp loại bỏ các tín hiệu phá vỡ giả (Bull Trap).")
____scan_btn = st.button("🔍 Bắt đầu quét thị trường ngay")
if scan_btn:
____danh_sach_ma = ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE')
____progress_bar = st.progress(0)
____status_text = st.empty()
____ket_qua_loc = []
____for idx, ma in enumerate(danh_sach_ma):
________status_text.text(f"Đang phân tích dòng tiền & kỹ thuật mã {ma} ({idx+1}/{len(danh_sach_ma)})...")
________progress_bar.progress((idx + 1) / len(danh_sach_ma))
________res = scan_single_stock(ma)
________if res is not None:
____________ket_qua_loc.append(res)
________time.sleep(0.1)
____status_text.text("✅ Hoàn tất quét tín hiệu toàn thị trường!")
____if ket_qua_loc:
________df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
________st.dataframe(df_res, use_container_width=True, hide_index=True)
____else:
________st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
"""
exec('\n'.join((' ' * (len(l) - len(l.lstrip('')))) + l.lstrip('') for l in raw_code.strip().splitlines()))
