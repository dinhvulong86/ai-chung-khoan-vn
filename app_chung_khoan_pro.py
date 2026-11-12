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
    end_ts = int(datetime.now().timestamp())
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    try:
        count_back = min(int(days * 0.72), 1000)
        url_tcbs = "https://tcbs.com.vn"
        params_tcbs = {
            "ticker": symbol,
            "type": "stock",
            "resolution": "D",
            "to": end_ts,
            "countBack": count_back
        }
        res = requests.get(url_tcbs, params=params_tcbs, headers=headers, timeout=10)
        data = res.json().get('data', [])
        if data and len(data) >= 30:
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
            df = df.loc[:, ('time', 'open', 'high', 'low', 'close', 'volume')].copy()
            for col in ['open', 'high', 'low', 'close']:
                if df[col].iloc[-1] < 1000:
                    df[col] = df[col] * 1000
            return df.dropna().sort_values('time').reset_index(drop=True)
    except Exception:
        pass

    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        ticker_vn = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        df = yf.download(ticker_vn, start=start_date, end=end_date, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df = df.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            return df.dropna().sort_values('time').reset_index(drop=True)
    except Exception:
        pass

    return pd.DataFrame()

def add_technical_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['High_20'] = df['high'].rolling(window=20).max()
    df['Low_20'] = df['low'].rolling(window=20).min()
    df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20']) * 100
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def analyze_one_stock(symbol, days_history, epochs):
    df = get_stock_data(symbol, days=days_history)
    if df.empty or len(df) < 100:
        st.error(f"Không đủ dữ liệu lịch sử cho mã {symbol}. Vui lòng kiểm tra lại mã cổ phiếu!")
        return

    df = add_technical_indicators(df)
    df_clean = df.dropna().copy()
    features = ('close', 'volume', 'MA20', 'RSI', 'MACD')
    data = df_clean.loc[:, features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    look_back = 30
    X = np.array([scaled_data[i-look_back:i].flatten() for i in range(look_back, len(scaled_data))])
    y = np.array([scaled_data[i, 0] for i in range(look_back, len(scaled_data))])

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
    model.fit(X, y)

    last_30_days = scaled_data[-look_back:].flatten().reshape(1, -1)
    pred_scaled = model.predict(last_30_days).item()
    dummy_array = np.zeros((1, len(features)))
    dummy_array.put(0, pred_scaled)
    predicted_price = scaler.inverse_transform(dummy_array).item(0)

    current_price = float(df_clean['close'].iloc[-1])
    pct_change = ((predicted_price - current_price) / current_price) * 100
    latest_rsi = float(df_clean['RSI'].iloc[-1])
    latest_macd = float(df_clean['MACD'].iloc[-1])
    latest_sig = float(df_clean['Signal_Line'].iloc[-1])
    latest_ma20 = float(df_clean['MA20'].iloc[-1])
    range_20 = float(df_clean['Range_20_Pct'].iloc[-1])
    high_20 = float(df_clean['High_20'].iloc[-1])
    low_20 = float(df_clean['Low_20'].iloc[-1])

    is_sideway = (range_20 < 9.0) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
    gia_chot_loi = current_price * 1.10
    gia_cat_lo = min(current_price * 0.93, latest_ma20 * 0.98)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Giá đóng cửa hiện tại", f"{current_price:,.0f} đ")
    m2.metric("AI Dự đoán phiên tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
    rsi_text = "Quá mua (>70) - Rủi ro bán" if latest_rsi > 70 else ("Quá bán (<30) - Cơ hội hồi" if latest_rsi < 30 else "Trung tính")
    m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", rsi_text, delta_color="off")
    macd_text = "Cắt lên (Tích cực)" if latest_macd > latest_sig else "Nằm dưới (Cảnh báo bán)"
    m4.metric("Chỉ báo MACD", f"{latest_macd:.2f}", macd_text)

    st.subheader("🤖 Khuyến nghị Hành động Mua / Bán từ hệ thống")
    diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + (1 if latest_rsi < 35 else 0) + (1 if current_price > latest_ma20 else 0)
    diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + (2 if latest_rsi > 70 else 0) + (1 if current_price <= latest_ma20 else 0)

    if is_sideway:
        st.warning(f"⏸️ TRẠNG THÁI SIDEWAY (ĐI NGANG TÍCH LŨY): Biên độ 20 phiên chỉ dao động {range_20:.1f}% (trong hộp {low_20:,.0f} đ – {high_20:,.0f} đ). Hạn chế mua đuổi, chỉ canh mua gom sát hỗ trợ {low_20:,.0f} đ và bán lướt sóng gần kháng cự {high_20:,.0f} đ.")
    elif diem_ban >= 3:
        st.error(f"🔴 KHUYẾN NGHỊ: BÁN HẠ TỶ TRỌNG / CHỐT LỜI NGAY (Điểm rủi ro BÁN: {diem_ban}/5)")
    elif diem_mua >= 3 and diem_ban <= 1:
        st.success(f"🟢 KHUYẾN NGHỊ: CÂN NHẮC MUA MỚI (Điểm ủng hộ MUA: {diem_mua}/4)")
    elif diem_ban == 2:
        st.warning(f"🟠 KHUYẾN NGHỊ: CANH BÁN CHỐT LỜI TỪNG PHẦN / KHÔNG MUA ĐUỔI (Có {diem_ban} cảnh báo bán)")
    else:
        st.info("🟡 KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ & QUAN SÁT (Tín hiệu Mua/Bán đang cân bằng)")

    st.markdown("#### 🎯 Kế hoạch giao dịch & Quản trị rủi ro đề xuất:")
    p1, p2, p3 = st.columns(3)
    p1.info(f"**Vùng giá Mua an toàn:**\n\n`{latest_ma20:,.0f} đ` – `{current_price:,.0f} đ`")
    p2.success(f"**Mục tiêu BÁN Chốt lời (+10%):**\n\n`{gia_chot_loi:,.0f} đ` *(hoặc khi RSI > 75)*")
    p3.error(f"**Mốc BÁN Cắt lỗ bắt buộc:**\n\nThủng `{gia_cat_lo:,.0f} đ` *(~ -7% hoặc gãy MA20)*")

    st.subheader("📈 Kiểm thử hiệu quả chiến lược Mua/Bán (100 phiên gần nhất)")
    df_bt = df_clean.tail(100).copy()
    df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70), 1, 0)
    df_bt['Daily_Return'] = df_bt['close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
    total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
    first_close = float(df_bt['close'].head(1).item())
    last_close = float(df_bt['close'].tail(1).item())
    total_hold_return = (last_close / first_close) - 1

    bt1, bt2 = st.columns(2)
    bt1.info(f"**Lợi nhuận khi tuân thủ điểm Mua & Bán của hệ thống:** `{total_strategy_return*100:+.2f}%`")
    bt2.info(f"**Lợi nhuận nếu chỉ mua và nắm giữ (Buy & Hold):** `{total_hold_return*100:+.2f}%`")

    st.subheader(f"Biểu đồ kỹ thuật {symbol} (120 phiên gần nhất)")
    df_plot = df_clean.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
    fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Mục tiêu Chốt lời")
    fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Mốc Cắt lỗ")
    fig.update_layout(xaxis_rangeslider_visible=False, height=480)
    st.plotly_chart(fig, use_container_width=True)

def scan_single_stock(ma):
    try:
        df_ma = get_stock_data(ma, days=150)
        if len(df_ma) < 60:
            return None
        df_ma = add_technical_indicators(df_ma).dropna()
        last = df_ma.iloc[-1]
        prev = df_ma.iloc[-2]
        buy_score = (1 if last['MACD'] > last['Signal_Line'] else 0) + (1 if last['close'] > last['MA20'] else 0) + (1 if 30 <= last['RSI'] <= 68 else 0) + (1 if (last['MACD'] > last['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']) else 0)
        sell_score = (1 if last['MACD'] < last['Signal_Line'] else 0) + (1 if last['close'] < last['MA20'] else 0) + (1 if last['RSI'] > 70 else 0)
        range_20 = float(last['Range_20_Pct'])
        is_sideway = (range_20 < 9.0) and (abs(float(last['close']) - float(last['MA20'])) / float(last['MA20']) < 0.025)

        if is_sideway:
            trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
elif buy_score >= 3 and sell_score == 0:
trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
elif sell_score >= 2 or last['RSI'] > 72:
trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
elif buy_score == 2:
trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
else:
trang_thai = "⚪ ĐỨNG NGOÀI"
return {
"Mã CP": ma,
"Giá hiện tại (đ)": round(float(last['close']), 0),
"Mốc Cắt lỗ (đ)": round(float(last['close']) * 0.93, 0),
"Biên độ 20P (%)": round(range_20, 1),
"RSI (14)": round(float(last['RSI']), 1),
"Điểm Mua": f"{buy_score}/4",
"Cảnh báo Bán": f"{sell_score}/3",
"Khuyến nghị": trang_thai
}
except Exception:
return None
tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Điểm Mua/Bán", "🔍 Bộ Lọc Quét Tín Hiệu HOSE & HNX"])
with tab1:
col1, col2, col3, col4 = st.columns(4)
symbol = col1.text_input("Nhập mã cổ phiếu HOSE/HNX/UPCOM (VD: CEO, MBS, SHS, FPT):", value="MBS").upper().strip()
days_history = col2.selectbox("Dữ liệu lịch sử học (ngày):", (365, 500, 730, 1000), index=2)
epochs = col3.selectbox("Số vòng học AI (Iterations):", (100, 200, 300), index=1)
col4.write("")
col4.write("")
run_btn = col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True)
if run_btn:
analyze_one_stock(symbol, days_history, epochs)
with tab2:
st.subheader("Quét tín hiệu Mua, Bán & Sideway tự động trên rổ HNX & VN30")
st.write("Danh sách quét gồm các cổ phiếu thanh khoản cao sàn HNX (MBS, SHS, CEO, PVS, IDC, HUT, TNG, VCS) cùng rổ VN30.")
scan_btn = st.button("🔍 Bắt đầu quét thị trường ngay")
if scan_btn:
danh_sach_ma = ('MBS', 'SHS', 'CEO', 'PVS', 'IDC', 'HUT', 'TNG', 'VCS', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE')
progress_bar = st.progress(0)
status_text = st.empty()
ket_qua_loc = []
for idx, ma in enumerate(danh_sach_ma):
status_text.text(f"Đang phân tích tín hiệu mã {ma} ({idx+1}/{len(danh_sach_ma)})...")
progress_bar.progress((idx + 1) / len(danh_sach_ma))
res = scan_single_stock(ma)
if res is not None:
ket_qua_loc.append(res)
time.sleep(0.1)
status_text.text("✅ Hoàn tất quét tín hiệu Mua, Bán & Sideway trên cả HOSE và HNX!")
if ket_qua_loc:
df_res = pd.DataFrame(ket_qua_loc).sort_values(by=["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
st.dataframe(df_res, use_container_width=True, hide_index=True)
else:
st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")
