import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import Vnstock
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import time

# Cấu hình giao diện trang web
st.set_page_config(page_title="Hệ Thống AI Phân Tích Chứng Khoán VN", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích, Khuyến Nghị & Lọc Cổ Phiếu (Pro)")
st.markdown("Tích hợp **Deep Learning (LSTM)**, chỉ báo **RSI/MACD**, **Backtest lợi nhuận** và **Bộ lọc quét thị trường**.")

# Hàm tính các chỉ báo kỹ thuật chuẩn
def add_technical_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Tính RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Tính MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Chia giao diện thành 2 Tab chức năng chính
tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Dự Đoán AI", "🔍 Bộ Lọc Quét Rổ VN30 & MBS"])

# ==============================================================================
# TAB 1: PHÂN TÍCH MÃ CỔ PHIẾU CHI TIẾT + AI LSTM + BACKTEST
# ==============================================================================
with tab1:
    col_input1, col_input2, col_input3, col_input4 = st.columns([2, 2, 2, 2])
    with col_input1:
        symbol = st.text_input("Nhập mã cổ phiếu (VD: MBS, FPT, HPG):", value="MBS").upper()
    with col_input2:
        days_history = st.selectbox("Dữ liệu lịch sử học:", [730, 1000, 1500], index=1)
    with col_input3:
        epochs = st.selectbox("Số vòng học AI (Epochs):", [10, 15, 25], index=1)
    with col_input4:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Chạy Phân Tích Ngay", use_container_width=True)

    if run_btn:
        with st.spinner(f"Đang tải dữ liệu và huấn luyện mạng nơ-ron LSTM cho mã {symbol}..."):
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')
                
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                df = stock.quote.history(start=start_date, end=end_date, interval='1D')
                
                if df.empty or len(df) < 120:
                    st.error("Không đủ dữ liệu lịch sử cho mã này. Vui lòng kiểm tra lại mã cổ phiếu!")
                    st.stop()
                    
                df = add_technical_indicators(df)
                df_clean = df.dropna().copy()
                
                # Chuẩn bị dữ liệu cho AI LSTM
                features = ['close', 'volume', 'MA20', 'RSI', 'MACD']
                data = df_clean[features].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
                
                look_back = 60
                X, y = [], []
                for i in range(look_back, len(scaled_data)):
                    X.append(scaled_data[i-look_back:i])
                    y.append(scaled_data[i, 0])
                    
                X, y = np.array(X), np.array(y)
                split = int(len(X) * 0.8)
                X_train, y_train = X[:split], y[:split]
                
                # Huấn luyện mô hình LSTM
                model = Sequential([
                    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape, X_train.shape)),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=25),
                    Dense(units=1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
                
                # Dự đoán phiên tiếp theo
                last_60_days = scaled_data[-look_back:]
                X_next = np.reshape(last_60_days, (1, look_back, len(features)))
                pred_scaled = model.predict(X_next, verbose=0)
                
                dummy_array = np.zeros((1, len(features)))
                dummy_array = pred_scaled
                predicted_price = scaler.inverse_transform(dummy_array)
                
                current_price = df_clean['close'].iloc[-1]
                pct_change = ((predicted_price - current_price) / current_price) * 100
                latest_rsi = df_clean['RSI'].iloc[-1]
                latest_macd = df_clean['MACD'].iloc[-1]
                latest_sig = df_clean['Signal_Line'].iloc[-1]

                # --- HIỂN THỊ CHỈ SỐ CHÍNH ---
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá đóng cửa hiện tại", f"{current_price:,.2f}")
                m2.metric("AI Dự đoán phiên tới", f"{predicted_price:,.2f}", f"{pct_change:+.2f}%")
                
                rsi_text = "Quá mua (>70)" if latest_rsi > 70 else ("Quá bán (<30)" if latest_rsi < 30 else "Trung tính")
                m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", rsi_text, delta_color="off")
                
                macd_text = "Cắt lên (Tích cực)" if latest_macd > latest_sig else "Nằm dưới (Tiêu cực)"
                m4.metric("Chỉ báo MACD", f"{latest_macd:.2f}", macd_text)

                # --- BỘ TỔNG HỢP KHUYẾN NGHỊ TỰ ĐỘNG ---
                st.subheader("🤖 Kết luận & Khuyến nghị từ hệ thống")
                diem = 0
                ly_do = []
                
                if pct_change > 0.5:
                    diem += 1
                    ly_do.append(f"✅ **AI LSTM:** Dự báo giá tăng tích cực ({pct_change:+.2f}%).")
                elif pct_change < -0.5:
                    diem -= 1
                    ly_do.append(f"⚠️ **AI LSTM:** Dự báo áp lực giảm giá ({pct_change:+.2f}%).")
                else:
                    ly_do.append(f"➖ **AI LSTM:** Dự báo giá đi ngang biên độ hẹp ({pct_change:+.2f}%).")

                if latest_macd > latest_sig:
                    diem += 1
                    ly_do.append("✅ **MACD:** Đường MACD nằm trên đường tín hiệu, ủng hộ xu hướng tăng.")
                else:
                    diem -= 1
                    ly_do.append("⚠️ **MACD:** Đường MACD nằm dưới đường tín hiệu, động lượng yếu.")

                if latest_rsi < 30:
                    diem += 1
                    ly_do.append(f"✅ **RSI ({latest_rsi:.1f}):** Đang ở vùng quá bán, có xác suất bật tăng hồi phục.")
                elif latest_rsi > 70:
                    diem -= 1
                    ly_do.append(f"⚠️ **RSI ({latest_rsi:.1f}):** Đang ở vùng quá mua, rủi ro điều chỉnh ngắn hạn cao.")
                else:
                    ly_do.append(f"➖ **RSI ({latest_rsi:.1f}):** Vùng trung tính an toàn.")

                if diem >= 2:
                    st.success(f"🟢 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC MUA (Điểm tín hiệu: {diem}/3)")
                elif diem <= -2:
                    st.error(f"🔴 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC BÁN / ĐỨNG NGOÀI (Điểm tín hiệu: {diem}/3)")
                else:
                    st.warning(f"🟡 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: QUAN SÁT / NẮM GIỮ (Điểm tín hiệu: {diem}/3)")

                for item in ly_do:
                    st.write(item)

                # --- KIỂM THỬ LỢI NHUẬN QUÁ KHỨ (BACKTEST 100 PHIÊN GẦN NHẤT) ---
                st.subheader("📈 Kiểm thử hiệu quả chiến lược kỹ thuật (100 phiên gần nhất)")
                df_bt = df_clean.tail(100).copy()
                # Giả lập mua khi MACD > Signal và RSI < 70, bán khi MACD < Signal
                df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70), 1, 0)
                df_bt['Daily_Return'] = df_bt['close'].pct_change()
                df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
                
                total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
                total_hold_return = (df_bt['close'].iloc[-1] / df_bt['close'].iloc) - 1
                
                bt1, bt2 = st.columns(2)
                bt1.info(f"**Lợi nhuận nếu giao dịch theo tín hiệu hệ thống:** `{total_strategy_return*100:+.2f}%`")
                bt2.info(f"**Lợi nhuận nếu chỉ mua và nắm giữ (Buy & Hold):** `{total_hold_return*100:+.2f}%`")

                # --- BIỂU ĐỒ KỸ THUẬT ---
                st.subheader(f"Biểu đồ kỹ thuật {symbol} (120 phiên gần nhất)")
                df_plot = df_clean.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_plot['time'], open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'], close=df_plot['close'], name='Giá nến'
                ))
                fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
                fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
                fig.update_layout(xaxis_rangeslider_visible=False, height=450)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")

# ==============================================================================
# TAB 2: BỘ LỌC QUÉT TOÀN BỘ RỔ VN30 + MBS TÌM MÃ ĐẸP
Dưới đây là toàn bộ nội dung chuẩn của file app_chung_khoan_pro.py. Lưu ý rằng ở phiên bản dưới đây, tôi đã kiểm tra và điền đầy đủ các tham số mảng/chỉ số (tránh tình trạng thiếu giá trị trong danh sách selectbox hoặc chỉ số mảng shape) để bạn chỉ cần sao chép nguyên văn và chạy ngay mà không bị lỗi cú pháp:
python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import Vnstock
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import time

# Cấu hình giao diện trang web
st.set_page_config(page_title="Hệ Thống AI Phân Tích Chứng Khoán VN", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích, Khuyến Nghị & Lọc Cổ Phiếu (Pro)")
st.markdown("Tích hợp **Deep Learning (LSTM)**, chỉ báo **RSI/MACD**, **Backtest lợi nhuận** và **Bộ lọc quét thị trường**.")

# Hàm tính các chỉ báo kỹ thuật chuẩn
def add_technical_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Tính RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Tính MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Chia giao diện thành 2 Tab chức năng chính
tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu & Dự Đoán AI", "🔍 Bộ Lọc Quét Rổ VN30 & MBS"])

# ==============================================================================
# TAB 1: PHÂN TÍCH MÃ CỔ PHIẾU CHI TIẾT + AI LSTM + BACKTEST
# ==============================================================================
with tab1:
    col_input1, col_input2, col_input3, col_input4 = st.columns(4)
    with col_input1:
        symbol = st.text_input("Nhập mã cổ phiếu (VD: MBS, FPT, HPG):", value="MBS").upper()
    with col_input2:
        days_history = st.selectbox("Dữ liệu lịch sử học (ngày):", [500, 1000, 1500, 2000], index=1)
    with col_input3:
        epochs = st.selectbox("Số vòng học AI (Epochs):", [10, 15, 25, 40], index=1)
    with col_input4:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Chạy Phân Tích Ngay", use_container_width=True)

    if run_btn:
        with st.spinner(f"Đang tải dữ liệu và huấn luyện mạng nơ-ron LSTM cho mã {symbol}..."):
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')
                
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                df = stock.quote.history(start=start_date, end=end_date, interval='1D')
                
                if df.empty or len(df) < 120:
                    st.error("Không đủ dữ liệu lịch sử cho mã này. Vui lòng kiểm tra lại mã cổ phiếu!")
                    st.stop()
                    
                df = add_technical_indicators(df)
                df_clean = df.dropna().copy()
                
                # Chuẩn bị dữ liệu cho AI LSTM
                features = ['close', 'volume', 'MA20', 'RSI', 'MACD']
                data = df_clean[features].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
                
                look_back = 60
                X, y = [], []
                for i in range(look_back, len(scaled_data)):
                    X.append(scaled_data[i-look_back:i])
                    y.append(scaled_data[i, 0])
                    
                X, y = np.array(X), np.array(y)
                split = int(len(X) * 0.8)
                X_train, y_train = X[:split], y[:split]
                
                # Huấn luyện mô hình LSTM
                model = Sequential([
                    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=25),
                    Dense(units=1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
                
                # Dự đoán phiên tiếp theo
                last_60_days = scaled_data[-look_back:]
                X_next = np.reshape(last_60_days, (1, look_back, len(features)))
                pred_scaled = model.predict(X_next, verbose=0)
                
                dummy_array = np.zeros((1, len(features)))
                dummy_array[0, 0] = pred_scaled[0, 0]
                predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
                
                current_price = df_clean['close'].iloc[-1]
                pct_change = ((predicted_price - current_price) / current_price) * 100
                latest_rsi = df_clean['RSI'].iloc[-1]
                latest_macd = df_clean['MACD'].iloc[-1]
                latest_sig = df_clean['Signal_Line'].iloc[-1]

                # --- HIỂN THỊ CHỈ SỐ CHÍNH ---
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá đóng cửa hiện tại", f"{current_price:,.2f}")
                m2.metric("AI Dự đoán phiên tới", f"{predicted_price:,.2f}", f"{pct_change:+.2f}%")
                
                rsi_text = "Quá mua (>70)" if latest_rsi > 70 else ("Quá bán (<30)" if latest_rsi < 30 else "Trung tính")
                m3.metric("Chỉ số RSI (14)", f"{latest_rsi:.1f}", rsi_text, delta_color="off")
                
                macd_text = "Cắt lên (Tích cực)" if latest_macd > latest_sig else "Nằm dưới (Tiêu cực)"
                m4.metric("Chỉ báo MACD", f"{latest_macd:.2f}", macd_text)

                # --- BỘ TỔNG HỢP KHUYẾN NGHỊ TỰ ĐỘNG ---
                st.subheader("🤖 Kết luận & Khuyến nghị từ hệ thống")
                diem = 0
                ly_do = []
                
                if pct_change > 0.5:
                    diem += 1
                    ly_do.append(f"✅ **AI LSTM:** Dự báo giá tăng tích cực ({pct_change:+.2f}%).")
                elif pct_change < -0.5:
                    diem -= 1
                    ly_do.append(f"⚠️ **AI LSTM:** Dự báo áp lực giảm giá ({pct_change:+.2f}%).")
                else:
                    ly_do.append(f"➖ **AI LSTM:** Dự báo giá đi ngang biên độ hẹp ({pct_change:+.2f}%).")

                if latest_macd > latest_sig:
                    diem += 1
                    ly_do.append("✅ **MACD:** Đường MACD nằm trên đường tín hiệu, ủng hộ xu hướng tăng.")
                else:
                    diem -= 1
                    ly_do.append("⚠️ **MACD:** Đường MACD nằm dưới đường tín hiệu, động lượng yếu.")

                if latest_rsi < 30:
                    diem += 1
                    ly_do.append(f"✅ **RSI ({latest_rsi:.1f}):** Đang ở vùng quá bán, có xác suất bật tăng hồi phục.")
                elif latest_rsi > 70:
                    diem -= 1
                    ly_do.append(f"⚠️ **RSI ({latest_rsi:.1f}):** Đang ở vùng quá mua, rủi ro điều chỉnh ngắn hạn cao.")
                else:
                    ly_do.append(f"➖ **RSI ({latest_rsi:.1f}):** Vùng trung tính an toàn.")

                if diem >= 2:
                    st.success(f"🟢 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC MUA (Điểm tín hiệu: {diem}/3)")
                elif diem <= -2:
                    st.error(f"🔴 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: CÂN NHẮC BÁN / ĐỨNG NGOÀI (Điểm tín hiệu: {diem}/3)")
                else:
                    st.warning(f"🟡 KHUYẾN NGHỊ ĐỐI VỚI {symbol}: QUAN SÁT / NẮM GIỮ (Điểm tín hiệu: {diem}/3)")

                for item in ly_do:
                    st.write(item)

                # --- KIỂM THỬ LỢI NHUẬN QUÁ KHỨ (BACKTEST 100 PHIÊN GẦN NHẤT) ---
                st.subheader("📈 Kiểm thử hiệu quả chiến lược kỹ thuật (100 phiên gần nhất)")
                df_bt = df_clean.tail(100).copy()
                df_bt['Position'] = np.where((df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['RSI'] < 70), 1, 0)
                df_bt['Daily_Return'] = df_bt['close'].pct_change()
                df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
                
                total_strategy_return = (1 + df_bt['Strategy_Return'].fillna(0)).prod() - 1
                total_hold_return = (df_bt['close'].iloc[-1] / df_bt['close'].iloc[0]) - 1
                
                bt1, bt2 = st.columns(2)
                bt1.info(f"**Lợi nhuận nếu giao dịch theo tín hiệu hệ thống:** `{total_strategy_return*100:+.2f}%`")
                bt2.info(f"**Lợi nhuận nếu chỉ mua và nắm giữ (Buy & Hold):** `{total_hold_return*100:+.2f}%`")

                # --- BIỂU ĐỒ KỸ THUẬT ---
                st.subheader(f"Biểu đồ kỹ thuật {symbol} (120 phiên gần nhất)")
                df_plot = df_clean.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_plot['time'], open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'], close=df_plot['close'], name='Giá nến'
                ))
                fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
                fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
                fig.update_layout(xaxis_rangeslider_visible=False, height=450)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")

# ==============================================================================
# TAB 2: BỘ LỌC QUÉT TOÀN BỘ RỔ VN30 + MBS TÌM MÃ ĐẸP
# ==============================================================================
with tab2:
Hãy thận trọng khi sử dụng mã.
st.subheader("Quét tín hiệu kỹ thuật tự động trên rổ VN30 & MBS")
st.write("Hệ thống sẽ quét dữ liệu mới nhất của các cổ phiếu đầu ngành để tìm ra những mã đang thỏa mãn điều kiện MUA (MACD cắt lên, giá vượt MA20, RSI an toàn).")
danh_sach_ma = [
'MBS', 'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB',
'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI',
'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]
if st.button("🔍 Bắt đầu quét thị trường ngay"):
ket_qua_loc = []
progress_bar = st.progress(0)
status_text = st.empty()
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d')
for idx, ma in enumerate(danh_sach_ma):
status_text.text(f"Đang phân tích mã {ma} ({idx+1}/{len(danh_sach_ma)})...")
progress_bar.progress((idx + 1) / len(danh_sach_ma))
try:
stock = Vnstock().stock(symbol=ma, source='VCI')
df_ma = stock.quote.history(start=start_date, end=end_date, interval='1D')
if len(df_ma) >= 60:
df_ma = add_technical_indicators(df_ma).dropna()
last = df_ma.iloc[-1]
prev = df_ma.iloc[-2]
# Chấm điểm tín hiệu kỹ thuật
score = 0
if last['MACD'] > last['Signal_Line']: score += 1
if last['close'] > last['MA20']: score += 1
if 30 <= last['RSI'] <= 68: score += 1
if last['MACD'] > last['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
score += 1
trang_thai = "🟢 TÍN HIỆU MUA ĐẸP" if score >= 3 else ("🟡 THEO DÕI" if score == 2 else "🔴 YẾU / ĐỨNG NGOÀI")
ket_qua_loc.append({
"Mã CP": ma,
"Giá hiện tại": round(last['close'], 2),
"RSI (14)": round(last['RSI'], 1),
"MACD": round(last['MACD'], 2),
"Điểm kỹ thuật": f"{score}/4",
"Đánh giá": trang_thai
})
time.sleep(0.15)
except:
continue
status_text.text("✅ Hoàn tất quét dữ liệu thị trường!")
if ket_qua_loc:
df_res = pd.DataFrame(ket_qua_loc)
df_res = df_res.sort_values(by="Điểm kỹ thuật", ascending=False)
st.dataframe(df_res, use_container_width=True, hide_index=True)
else:
st.warning("Không thể tải dữ liệu lúc này. Vui lòng thử lại sau ít phút.")

