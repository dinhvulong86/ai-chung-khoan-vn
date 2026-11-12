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

st.set_page_config(page_title="Hệ Thống AI Phân Tích Chứng Khoán VN", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích, Khuyến Nghị Mua/Bán & Lọc Cổ Phiếu (Pro)")
st.markdown("Hỗ trợ đầy đủ **3 sàn HOSE, HNX, UPCOM** | Tích hợp **AI Neural Network**, **Nhận diện Sideway**, **RSI/MACD** và **Điểm Mua/Bán**.")

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
____df['MA20'] = df['close'].rolling(window=20).mean()
____df['MA50'] = df['close'].rolling(window=50).mean()
____df['High_20'] = df['high'].rolling(window=20).max()
____df['Low_20'] = df['low'].rolling(window=20).min()
____df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20']) * 100
____delta = df['close'].diff()
____gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
____loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
____rs = gain / loss
____df['RSI'] = 100 - (100 / (1 + rs))
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
____features = ('close', 'volume', 'MA20', 'RSI', 'MACD')
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
____pct_change = ((predicted_price - current_price) / current_price) * 100
____latest_rsi = float(df_clean['RSI'].iloc[-1])
____latest_macd = float(df_clean['MACD'].iloc[-1])
____latest_sig = float(df_clean['Signal_Line'].iloc[-1])
____latest_ma20 = float(df_clean['MA20'].iloc[-1])
____range_20 = float(df_clean['Range_20_Pct'].iloc[-1])
____high_20 = float(df_clean['High_20'].iloc[-1])
____low_20 = float(df_clean['Low_20'].iloc[-1])
____is_sideway = (range_20 < 9.0) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
____gia_chot_loi = current_price * 1.10
____gia_cat_lo = min(current_price * 0.93, latest_ma20 * 0.98)
____st.divider()
____m1, m2, m3, m4 = st.columns(4)
____m1.metric("Giá đóng cửa hiện tại", f"{current_price:,.0f} đ")
____m2.metric("AI Dự đoán phiên tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
____rsi_text = "Quá mua (>70) - Rủi ro bán" if latest_rsi > 70 else ("Quá bán (<30) - Cơ hội hồi" if latest_rsi < 30 else "Trung tính")
____m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", rsi_text, delta_color="off")
____macd_text = "Cắt lên (Tích cực)" if latest_macd > latest_sig else "Nằm dưới (Cảnh báo bán)"
____m4.metric("Chỉ báo MACD", f"{latest_macd:.2f}", macd_text)
____st.subheader("🤖 Khuyến nghị Hành động Mua / Bán từ hệ thống")
____diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + (1 if latest_rsi < 35 else 0) + (1 if current_price > latest_ma20 else 0)
____diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + (2 if latest_rsi > 70 else 0) + (1 if current_price <= latest_ma20 else 0)
____if is_sideway:
________st.warning(f"⏸️ TRẠNG THÁI SIDEWAY (ĐI NGANG TÍCH LŨY): Biên độ 20 phiên chỉ dao động {range_20:.1f}% (trong hộp {low_20:,.0f} đ – {high_20:,.0f} đ). Hạn chế mua đuổi, chỉ canh mua gom sát hỗ trợ {low_20:,.0f} đ và bán lướt sóng gần kháng cự {high_20:,.0f} đ.")
____elif diem_ban >= 3:
________st.error(f"🔴 KHUYẾN NGHỊ: BÁN HẠ TỶ TRỌNG / CHỐT LỜI NGAY (Điểm rủi ro BÁN: {diem_ban}/5)")
____elif diem_mua >= 3 and diem_ban <= 1:
________st.success(f"🟢 KHUYẾN NGHỊ: CÂN NHẮC MUA MỚI (Điểm ủng hộ MUA: {diem_mua}/4)")
____elif diem_ban == 2:
________st.warning(f"🟠 KHUYẾN NGHỊ: CANH BÁN CHỐT LỜI TỪNG PHẦN / KHÔNG MUA ĐUỔI (Có {diem_ban} cảnh báo bán)")
____else:
________st.info("🟡 KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ & QUAN SÁT (Tín hiệu Mua/Bán đang cân bằng)")
____st.markdown("#### 🎯 Kế hoạch giao dịch & Quản trị rủi ro đề xuất:")
____p1, p2, p3 = st.columns(3)
____p1.info(f"**Vùng giá Mua an toàn:**\\n\\n`{latest_ma20:,.0f} đ` – `{current_price:,.0f} đ`")
____p2.success(f"**Mục tiêu BÁN Chốt lời (+10%):**\\n\\n`{gia_chot_loi:,.0f} đ` *(hoặc khi RSI > 75)*")
____p3.error(f"**Mốc BÁN Cắt lỗ bắt buộc:**\\n\\nThủng `{gia_cat_lo:,.0f} đ` *(~ -7% hoặc gãy MA20)*")
____st.subheader("📈 Kiểm thử hiệu quả chiến lược Mua/Bán (100 phiên gần nhất)")
____df_bt = df_clean.tail(100).copy()
____df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70), 1, 0)
____df_bt['Daily_Return'] = df_bt['close'].pct_change()
____df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
____total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
____first_close = float(df_bt['close'].head(1).item())
____last_close = float(df_bt['close'].tail(1).item())
____total_hold_return = (last_close / first_close) - 1
____bt1, bt2 = st.columns(2)
____bt1.info(f"**Lợi nhuận khi tuân thủ điểm Mua & Bán của hệ thống:** `{total_strategy_return*100:+.2f}%`")
____bt2.info(f"**Lợi nhuận nếu chỉ mua và nắm giữ (Buy & Hold):** `{total_hold_return*100:+.2f}%`")
____st.subheader(f"Biểu đồ kỹ thuật {symbol} (120 phiên gần nhất)")
____df_plot = df_clean.tail(120)
____fig = go.Figure()
____fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
____fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
____fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
____fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Mốc Cắt lỗ")
____fig.update_layout(xaxis_rangeslider_visible=False, height=480)
____st.plotly_chart(fig, use_container_width=True)

def scan_single_stock(ma):
____try:
________df_ma = get_stock_data(ma, days=150)
________if len(df_ma) < 60:
____________return None
________df_ma = add_technical_indicators(df_ma).dropna()
________last = df_ma.iloc[-1]
________prev = df_ma.iloc[-2]
________buy_score = (1 if last['MACD'] > last['Signal_Line'] else 0) + (1 if last['close'] > last['MA20'] else 0) + (1 if 30 <= last['RSI'] <= 68 else 0) + (1 if (last['MACD'] > last['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']) else 0)
________sell_score = (1 if last['MACD'] < last['Signal_Line'] else 0) + (1 if last['close'] < last['MA20'] else 0) + (1 if last['RSI'] > 70 else 0)
________range_20 = float(last['Range_20_Pct'])
________is_sideway = (range_20 < 9.0) and (abs(float(last['close']) - float(last['MA20'])) / float(last['MA20']) < 0.025)
________if is_sideway:
____________trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
________elif buy_score >= 3 and sell_score == 0:
____________trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
________elif sell_score >= 2 or last['RSI'] > 72:
____________trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
________elif buy_score == 2:
____________trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
________else:
____________trang_thai = "⚪ ĐỨNG NGOÀI"
________return {"Mã CP": ma, "Giá hiện tại (đ)": round(float(last['close']), 0), "Mốc Cắt lỗ (đ)": round(float(last['close']) * 0.93, 0), "Biên độ 20P (%)": round(range_20, 1), "RSI (14)": round(float(last['RSI']), 1), "Điểm Mua": f"{buy_score}/4", "Cảnh báo Bán": f"{sell_score}/3", "Khuyến nghị": trang_thai}
____except Exception:
________return None
tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Điểm Mua/Bán", "🔍 Bộ Lọc Quét Tín Hiệu HOSE & HNX"])
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
____st.subheader("Quét tín hiệu Mua, Bán & Sideway tự động trên rổ HNX & VN30")
____st.write("Danh sách quét gồm các cổ phiếu thanh khoản cao sàn HNX (MBS, SHS, CEO, PVS, IDC, HUT, TNG, VCS) cùng rổ VN30.")
____scan_btn = st.button("🔍 Bắt đầu quét thị trường ngay")
if scan_btn:
____danh_sach_ma = ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE')
____progress_bar = st.progress(0)
____status_text = st.empty()
____ket_qua_loc = []
____for idx, ma in enumerate(danh_sach_ma):
________status_text.text(f"Đang phân tích tín hiệu mã {ma} ({idx+1}/{len(danh_sach_ma)})...")
________progress_bar.progress((idx + 1) / len(danh_sach_ma))
________res = scan_single_stock(ma)
________if res is not None:
____________ket_qua_loc.append(res)
________time.sleep(0.1)
____status_text.text("✅ Hoàn tất quét tín hiệu Mua, Bán & Sideway trên cả HOSE và HNX!")
____if ket_qua_loc:
________df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
________st.dataframe(df_res, use_container_width=True, hide_index=True)
____else:
________st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
"""
exec('\n'.join((' ' * (len(l) - len(l.lstrip('_')))) + l.lstrip('_') for l in raw_code.strip().splitlines()))

