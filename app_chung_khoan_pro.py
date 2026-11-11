import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Hệ Thống AI Phân Tích Chứng Khoán VN", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích, Khuyến Nghị & Lọc Cổ Phiếu (Pro)")
st.markdown("Tích hợp **Mạng Nơ-ron AI (Neural Network)**, chỉ báo **RSI/MACD**, **Backtest lợi nhuận** và **Bộ lọc quét thị trường**.")

@st.cache_data(ttl=300)
def get_stock_data(symbol, days=500):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    ticker_vn = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
    
    try:
        df = yf.download(ticker_vn, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df = df.loc[:, ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')].copy()
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
        df = df.dropna().sort_values('time').reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def add_technical_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
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
        st.error("Không đủ dữ liệu lịch sử cho mã này. Vui lòng kiểm tra lại mã cổ phiếu (chỉ áp dụng cho cổ phiếu niêm yết trên HOSE/HNX)!")
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

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Giá đóng cửa hiện tại", f"{current_price:,.0f} đ")
    m2.metric("AI Dự đoán phiên tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
    rsi_text = "Quá mua (>70)" if latest_rsi > 70 else ("Quá bán (<30)" if latest_rsi < 30 else "Trung tính")
    m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", rsi_text, delta_color="off")
    macd_text = "Cắt lên (Tích cực)" if latest_macd > latest_sig else "Nằm dưới (Tiêu cực)"
    m4.metric("Chỉ báo MACD", f"{latest_macd:.2f}", macd_text)

    st.subheader("🤖 Kết luận & Khuyến nghị từ hệ thống")
    diem = (1 if pct_change > 0.5 else (-1 if pct_change < -0.5 else 0)) + (1 if latest_macd > latest_sig else -1) + (1 if latest_rsi < 30 else (-1 if latest_rsi > 70 else 0))
    if diem >= 2:
        st.success(f"🟢 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC MUA (Điểm tín hiệu: {diem}/3)")
    elif diem <= -2:
        st.error(f"🔴 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC BÁN / ĐỨNG NGOÀI (Điểm tín hiệu: {diem}/3)")
    else:
        st.warning(f"🟡 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: QUAN SÁT / NẮM GIỮ (Điểm tín hiệu: {diem}/3)")

    st.write(f"• **AI Neural Net:** Dự báo biến động phiên tới `{pct_change:+.2f}%`.")
    st.write(f"• **MACD:** Đường MACD hiện tại `{latest_macd:.2f}` so với đường tín hiệu `{latest_sig:.2f}`.")
    st.write(f"• **RSI (14):** Đang ở mức `{latest_rsi:.1f}` ({rsi_text}).")

    st.subheader("📈 Kiểm thử hiệu quả chiến lược kỹ thuật (100 phiên gần nhất)")
    df_bt = df_clean.tail(100).copy()
    df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70), 1, 0)
    df_bt['Daily_Return'] = df_bt['close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
    total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
    total_hold_return = (float(df_bt['close'].iloc[-1]) / float(df_bt['close'].iloc[0])) - 1

    bt1, bt2 = st.columns(2)
    bt1.info(f"**Lợi nhuận giao dịch theo tín hiệu hệ thống:** `{total_strategy_return*100:+.2f}%`")
    bt2.info(f"**Lợi nhuận nếu mua và nắm giữ (Buy & Hold):** `{total_hold_return*100:+.2f}%`")

    st.subheader(f"Biểu đồ kỹ thuật {symbol} (120 phiên gần nhất)")
    df_plot = df_clean.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Giá nến'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
    fig.update_layout(xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

def scan_market():
    danh_sach_ma = ('MBS', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE')
    ket_qua_loc = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ma in enumerate(danh_sach_ma):
        status_text.text(f"Đang phân tích mã {ma} ({idx+1}/{len(danh_sach_ma)})...")
        progress_bar.progress((idx + 1) / len(danh_sach_ma))
        try:
            df_ma = get_stock_data(ma, days=150)
            if len(df_ma) < 60:
                continue
            df_ma = add_technical_indicators(df_ma).dropna()
            last = df_ma.iloc[-1]
            prev = df_ma.iloc[-2]
            score = (1 if last['MACD'] > last['Signal_Line'] else 0) + (1 if last['close'] > last['MA20'] else 0) + (1 if 30 <= last['RSI'] <= 68 else 0) + (1 if (last['MACD'] > last['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']) else 0)
            trang_thai = "🟢 TÍN HIỆU MUA ĐẸP" if score >= 3 else ("🟡 THEO DÕI" if score == 2 else "🔴 YẾU / ĐỨNG NGOÀI")
            ket_qua_loc.append({"Mã CP": ma, "Giá hiện tại (đ)": round(float(last['close']), 0), "RSI (14)": round(float(last['RSI']), 1), "MACD": round(float(last['MACD']), 2), "Điểm kỹ thuật": f"{score}/4", "Đánh giá": trang_thai})
            time.sleep(0.05)
        except Exception:
            continue

    status_text.text("✅ Hoàn tất quét dữ liệu thị trường!")
    if ket_qua_loc:
        df_res = pd.DataFrame(ket_qua_loc).sort_values(by="Điểm kỹ thuật", ascending=False)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")

tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Dự Đoán AI", "🔍 Bộ Lọc Quét Rổ VN30 & MBS"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.text_input("Nhập mã cổ phiếu (VD: MBS, FPT, HPG):", value="MBS").upper().strip()
    days_history = col2.selectbox("Dữ liệu lịch sử học (ngày):", (365, 500, 730, 1000), index=2)
    epochs = col3.selectbox("Số vòng học AI (Iterations):", (100, 200, 300), index=1)
    col4.write("")
    col4.write("")
    if col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True):
        with st.spinner(f"Đang tải dữ liệu và huấn luyện AI cho mã {symbol}..."):
            try:
                analyze_one_stock(symbol, days_history, epochs)
            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")

with tab2:
    st.subheader("Quét tín hiệu kỹ thuật tự động trên rổ VN30 & MBS")
    st.write("Hệ thống sẽ quét dữ liệu mới nhất của các cổ phiếu đầu ngành để tìm ra những mã đang thỏa mãn điều kiện **MUA**.")
    if st.button("🔍 Bắt đầu quét thị trường ngay"):
        scan_market()
