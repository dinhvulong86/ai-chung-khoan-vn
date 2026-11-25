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
import hashlib
import joblib
import os

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (Optimized)", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích Kỹ Thuật & Quản Trị Danh Mục VN")
st.markdown("**Phiên bản tối ưu**: Cache model AI • Dữ liệu ổn định hơn • Ngưỡng chỉnh được • Quét nhanh hơn")

# ====================== SIDEBAR CẤU HÌNH ======================
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    VOL_BREAKOUT = st.slider("Ngưỡng Vol Breakout (%)", 120, 250, 150, 10)
    RSI_OVERBOUGHT = st.slider("RSI quá mua", 65, 85, 70)
    MFI_OVERBOUGHT = st.slider("MFI quá mua", 70, 90, 80)
    ATR_STOP = st.slider("Cắt lỗ (x ATR)", 1.0, 3.0, 2.0, 0.1)
    ATR_TAKE = st.slider("Chốt lời (x ATR)", 2.0, 5.0, 3.0, 0.1)
    MAX_WORKERS = st.slider("Số luồng quét", 4, 16, 8)

# ====================== SESSION STATE ======================
if "nhat_ky_lenh" not in st.session_state:
    st.session_state.nhat_ky_lenh = [
        {"Ngày quét": "25/11/2026", "Mã CP": "HAG", "Giá vào (đ)": 13850,
         "Cắt lỗ ATR (đ)": 13314, "Chốt lời ATR (đ)": 14654, "Ghi chú": "Mẫu theo dõi Bùng nổ Vol"}
    ]

# ====================== DATA FETCHING (TỐI ƯU) ======================
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Lấy dữ liệu ổn định: ưu tiên yfinance, fallback TCBS."""
    now_vn = datetime.utcnow() + timedelta(hours=7)
    today_str = now_vn.strftime('%Y-%m-%d')
    start_date = (now_vn - timedelta(days=days + 30)).strftime('%Y-%m-%d')
    
    df = pd.DataFrame()
    
    # 1. Ưu tiên yfinance (ổn định nhất hiện nay)
    try:
        ticker = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        df_yf = yf.download(ticker, start=start_date, end=today_str, progress=False, auto_adjust=True)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            df_yf = df_yf.reset_index()
            df_yf = df_yf.rename(columns={
                'Date': 'time', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            df_yf['time'] = pd.to_datetime(df_yf['time']).dt.strftime('%Y-%m-%d')
            df = df_yf[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
            df = df.sort_values('time').reset_index(drop=True)
    except Exception:
        pass
    
    # 2. Fallback TCBS (public endpoint cũ vẫn đôi khi hoạt động)
    if df.empty or len(df) < 80:
        try:
            end_ts = int(now_vn.timestamp())
            count_back = min(int(days * 0.8), 1000)
            url = "https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"
            params = {
                "ticker": symbol, "type": "stock", "resolution": "D",
                "to": end_ts, "countBack": count_back
            }
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            data = res.json().get('data', [])
            if data and len(data) >= 50:
                df_tcbs = pd.DataFrame(data)
                df_tcbs['time'] = pd.to_datetime(df_tcbs['tradingDate']).dt.strftime('%Y-%m-%d')
                df_tcbs = df_tcbs[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
                # Chuẩn hóa giá (một số mã trả về đơn vị nghìn)
                for col in ['open', 'high', 'low', 'close']:
                    if df_tcbs[col].iloc[-1] < 1000:
                        df_tcbs[col] = df_tcbs[col] * 1000
                df = df_tcbs.dropna().sort_values('time').reset_index(drop=True)
        except Exception:
            pass
    
    return df

# ====================== TECHNICAL INDICATORS ======================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['STD20'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']
    
    df['Vol_MA20'] = df['volume'].rolling(20).mean()
    df['Vol_Ratio'] = (df['volume'] / df['Vol_MA20'].replace(0, np.nan)) * 100
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Ichimoku
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['Tenkan'] = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(26).max()
    low_26 = df['low'].rolling(26).min()
    df['Kijun'] = (high_26 + low_26) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    high_52 = df['high'].rolling(52).max()
    low_52 = df['low'].rolling(52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    
    df['High_20'] = df['high'].rolling(20).max()
    df['Low_20'] = df['low'].rolling(20).min()
    df['Range_20_Pct'] = ((df['High_20'] - df['Low_20']) / df['Low_20']) * 100
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MFI
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
    neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
    mfr = pd.Series(pos_mf).rolling(14).sum() / pd.Series(neg_mf).rolling(14).sum().replace(0, np.nan)
    df['MFI'] = 100 - (100 / (1 + mfr))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# ====================== AI MODEL (CACHE) ======================
@st.cache_resource(show_spinner="Đang train / load model AI...")
def get_or_train_model(symbol: str, days: int, epochs: int, data_hash: str):
    """Cache model theo symbol + days + epochs + hash dữ liệu."""
    df = get_stock_data(symbol, days)
    if df.empty or len(df) < 120:
        return None, None, None
    
    df = add_technical_indicators(df).dropna()
    features = ['close', 'volume', 'MA20', 'RSI', 'MFI', 'MACD']
    data = df[features].values
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    look_back = 30
    X = np.array([scaled[i-look_back:i].flatten() for i in range(look_back, len(scaled))])
    y = scaled[look_back:, 0]
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=epochs,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    )
    model.fit(X, y)
    return model, scaler, features

def predict_price(model, scaler, features, df_clean):
    look_back = 30
    data = df_clean[features].values
    scaled = scaler.transform(data)
    last = scaled[-look_back:].flatten().reshape(1, -1)
    pred_scaled = model.predict(last)[0]
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = pred_scaled
    return scaler.inverse_transform(dummy)[0, 0]

# ====================== PHÂN TÍCH 1 MÃ ======================
def analyze_one_stock(symbol, days_history, epochs):
    df = get_stock_data(symbol, days=days_history)
    if df.empty or len(df) < 100:
        st.error(f"❌ Không đủ dữ liệu cho mã **{symbol}**. Thử mã khác hoặc đợi vài phút.")
        return
    
    df = add_technical_indicators(df)
    df_clean = df.dropna().copy()
    
    # Hash dữ liệu để cache model
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df_clean[['close', 'volume']].tail(50)).values).hexdigest()
    
    model, scaler, features = get_or_train_model(symbol, days_history, epochs, data_hash)
    if model is None:
        st.error("Không train được model.")
        return
    
    predicted_price = predict_price(model, scaler, features, df_clean)
    
    current_price = float(df_clean['close'].iloc[-1])
    prev_close = float(df_clean['close'].iloc[-2])
    vol_ratio = float(df_clean['Vol_Ratio'].iloc[-1])
    latest_atr = float(df_clean['ATR'].iloc[-1])
    pct_change = ((predicted_price - current_price) / current_price) * 100
    latest_rsi = float(df_clean['RSI'].iloc[-1])
    latest_mfi = float(df_clean['MFI'].iloc[-1])
    latest_macd = float(df_clean['MACD'].iloc[-1])
    latest_sig = float(df_clean['Signal_Line'].iloc[-1])
    latest_ma20 = float(df_clean['MA20'].iloc[-1])
    bb_upper = float(df_clean['BB_Upper'].iloc[-1])
    bb_lower = float(df_clean['BB_Lower'].iloc[-1])
    senkou_a = float(df_clean['Senkou_A'].iloc[-1])
    senkou_b = float(df_clean['Senkou_B'].iloc[-1])
    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    range_20 = float(df_clean['Range_20_Pct'].iloc[-1])
    high_20 = float(df_clean['High_20'].iloc[-1])
    low_20 = float(df_clean['Low_20'].iloc[-1])
    
    is_sideway = (range_20 < 9.5) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
    is_bull_trap = (current_price > prev_close * 1.02) and (vol_ratio < 75) and (latest_rsi > 65)
    is_vol_breakout = (current_price > prev_close * 1.015) and (vol_ratio >= VOL_BREAKOUT) and (current_price > latest_ma20)
    
    gia_cat_lo = current_price - (ATR_STOP * latest_atr)
    gia_chot_loi = current_price + (ATR_TAKE * latest_atr)
    
    # ===== UI =====
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Giá hiện tại", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}%")
    m2.metric("AI Dự đoán", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
    m3.metric("RSI / MFI", f"{latest_rsi:.0f} / {latest_mfi:.0f}",
              "Quá mua" if latest_rsi > RSI_OVERBOUGHT else ("Quá bán" if latest_rsi < 30 else "An toàn"))
    ichi = "Trên mây (Tăng)" if current_price > cloud_top else ("Dưới mây (Giảm)" if current_price < cloud_bot else "Trong mây")
    m4.metric("Mây Ichimoku", ichi, f"{cloud_bot:,.0f} – {cloud_top:,.0f}")
    m5.metric("ATR(14)", f"{latest_atr:,.0f} đ", f"±{(latest_atr/current_price)*100:.1f}%")
    
    st.subheader("🤖 Phân tích & Khuyến nghị")
    if is_vol_breakout:
        st.success(f"🔥 **BÙNG NỔ VOLUME**: Vol đạt {vol_ratio:.0f}% trung bình 20 phiên + giá bứt phá!")
    if is_bull_trap:
        st.error(f"🚨 **BẪY TĂNG GIÁ**: Giá tăng nhưng Vol yếu ({vol_ratio:.0f}%) + RSI cao.")
    
    diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + \
               (1 if current_price > cloud_top else 0) + (1 if (current_price > latest_ma20 and vol_ratio >= 100) else 0)
    diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + \
               (2 if (latest_rsi > RSI_OVERBOUGHT or latest_mfi > MFI_OVERBOUGHT) else 0) + \
               (1 if current_price < cloud_bot else 0)
    
    if is_sideway:
        st.warning(f"⏸️ **SIDEWAY**: Biên độ 20 phiên {range_20:.1f}%. Ưu tiên mua gần BB dưới ({bb_lower:,.0f}).")
    elif diem_ban >= 3:
        st.error(f"🔴 **BÁN / CHỐT LỜI** (Điểm rủi ro: {diem_ban}/5)")
    elif diem_mua >= 3 and diem_ban <= 1:
        st.success(f"🟢 **CÂN NHẮC MUA** (Điểm mua: {diem_mua}/4)")
    elif diem_ban == 2:
        st.warning(f"🟠 **CANH BÁN TỪNG PHẦN** (Có {diem_ban} cảnh báo)")
    else:
        st.info("🟡 **NẮM GIỮ & QUAN SÁT**")
    
    st.markdown("#### 🎯 Kế hoạch ATR động")
    p1, p2, p3 = st.columns(3)
    p1.info(f"**Vùng mua an toàn**\n\n{bb_lower:,.0f} – {latest_ma20:,.0f} đ")
    p2.success(f"**Chốt lời (+{ATR_TAKE} ATR)**\n\n{gia_chot_loi:,.0f} đ")
    p3.error(f"**Cắt lỗ (-{ATR_STOP} ATR)**\n\n{gia_cat_lo:,.0f} đ")
    
    # Chart
    st.subheader(f"Biểu đồ {symbol} – Ichimoku & Bollinger")
    df_plot = df_clean.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot['time'], open=df_plot['open'], high=df_plot['high'],
                                 low=df_plot['low'], close=df_plot['close'], name='Nến'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['Senkou_A'], line=dict(color='rgba(0,200,100,0.4)', width=1), name='Senkou A'))
    fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['Senkou_B'], line=dict(color='rgba(255,80,80,0.4)', width=1),
                             fill='tonexty', fillcolor='rgba(144,238,144,0.15)', name='Mây Ichimoku'))
    fig.add_hline(y=gia_chot_loi, line_dash="dot", line_color="green", annotation_text="Chốt lời")
    fig.add_hline(y=gia_cat_lo, line_dash="dot", line_color="red", annotation_text="Cắt lỗ")
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, hovermode='x unified',
                      xaxis_type='category', xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ====================== QUÉT NHANH ======================
def scan_single_stock(ma: str):
    try:
        df = get_stock_data(ma, days=160)
        if len(df) < 80:
            return None
        df = add_technical_indicators(df).dropna()
        last = df.iloc[-1]
        prev = df.iloc[-2]
        curr_p = float(last['close'])
        prev_p = float(prev['close'])
        vol_ratio = float(last['Vol_Ratio'])
        atr_val = float(last['ATR'])
        cloud_top = max(float(last['Senkou_A']), float(last['Senkou_B']))
        cloud_bot = min(float(last['Senkou_A']), float(last['Senkou_B']))
        
        is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= VOL_BREAKOUT) and (curr_p > float(last['MA20']))
        buy_score = (1 if last['MACD'] > last['Signal_Line'] else 0) + \
                    (1 if curr_p > float(last['MA20']) else 0) + \
                    (1 if curr_p > cloud_top else 0) + \
                    (1 if vol_ratio >= 110 else 0)
        sell_score = (1 if last['MACD'] < last['Signal_Line'] else 0) + \
                     (1 if curr_p < cloud_bot else 0) + \
                     (1 if (last['RSI'] > RSI_OVERBOUGHT or last['MFI'] > MFI_OVERBOUGHT) else 0)
        range_20 = float(last['Range_20_Pct'])
        is_sideway = (range_20 < 9.5) and (abs(curr_p - float(last['MA20'])) / float(last['MA20']) < 0.025)
        
        if is_vol_breakout:
            trang_thai = "🔥 BÙNG NỔ VOL + MUA"
        elif is_sideway:
            trang_thai = "⏸️ SIDEWAY"
        elif buy_score >= 3 and sell_score == 0:
            trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
        elif sell_score >= 2 or last['RSI'] > 72 or last['MFI'] > 80:
            trang_thai = "🔴 CẢNH BÁO BÁN"
        elif buy_score == 2:
            trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
        else:
            trang_thai = "⚪ ĐỨNG NGOÀI"
        
        return {
            "Mã CP": ma,
            "Giá hiện tại (đ)": round(curr_p, 0),
            "Cắt lỗ ATR (đ)": round(curr_p - ATR_STOP * atr_val, 0),
            "Chốt lời ATR (đ)": round(curr_p + ATR_TAKE * atr_val, 0),
            "Tỷ lệ Vol (%)": f"{vol_ratio:.0f}%",
            "RSI / MFI": f"{float(last['RSI']):.0f} / {float(last['MFI']):.0f}",
            "Điểm Mua": f"{buy_score}/4",
            "Khuyến nghị": trang_thai
        }
    except Exception:
        return None

# ====================== DANH MỤC ======================
DEFAULT_TOP = (
    'MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS', 'EVS', 'TVS',
    'ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB',
    'CEO', 'DIG', 'DXG', 'HDC', 'HDG', 'HUT', 'IDC', 'KBC', 'KDH', 'NLG', 'NVL', 'PDR', 'SCR', 'VHM', 'VIC', 'VRE',
    'HPG', 'HSG', 'NKG', 'BSR', 'GAS', 'PLX', 'POW', 'PVD', 'PVS', 'FPT', 'CMG', 'CTR',
    'DGC', 'DCM', 'DPM', 'GVR', 'MWG', 'MSN', 'PNJ', 'FRT', 'VNM', 'SAB', 'HAG', 'ANV', 'VHC',
    'GMD', 'HAH', 'VJC', 'CTD', 'HHV', 'LCG'
)

DANH_MUC_NHOM = {
    "🌐 Top cổ phiếu sôi động (mặc định)": "DYNAMIC",
    "🏛️ Nhóm Chứng khoán": ('MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS'),
    "🏦 Nhóm Ngân hàng": ('ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB'),
    "🏗️ Bất động sản": ('CEO', 'DIG', 'DXG', 'HDC', 'HDG', 'HUT', 'IDC', 'KBC', 'KDH', 'NLG', 'NVL', 'PDR', 'VHM', 'VIC', 'VRE'),
    "⚙️ Thép - Dầu khí - Công nghệ": ('HPG', 'HSG', 'NKG', 'BSR', 'GAS', 'PLX', 'POW', 'PVD', 'FPT', 'CMG', 'CTR', 'DGC', 'MWG', 'PNJ')
}

# ====================== UI TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Chuyên Sâu", "🔍 Quét Đột Biến Dòng Tiền", "📝 Nhật Ký & Chấm Điểm"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("Mã cổ phiếu:", value="MBS").upper().strip()
    days_history = c2.selectbox("Số ngày lịch sử:", (365, 500, 730, 1000), index=1)
    epochs = c3.selectbox("Số vòng học AI:", (80, 150, 200, 300), index=1)
    run_btn = c4.button("🚀 Chạy Phân Tích", use_container_width=True, type="primary")
    
    if run_btn and symbol:
        with st.spinner(f"Đang phân tích {symbol}..."):
            analyze_one_stock(symbol, days_history, epochs)

with tab2:
    st.subheader("Quét Volume Breakout • Ichimoku • ATR")
    f1, f2 = st.columns(2)
    nhom = f1.selectbox("Chọn nhóm:", list(DANH_MUC_NHOM.keys()))
    ma_rieng = f2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="BSR, KBC, FPT").upper().strip()
    chi_hien_mua = st.checkbox("Chỉ hiện 🔥 Bùng nổ / 🟢 Mua đẹp / ⏸️ Sideway", value=True)
    
    if st.button("⚡ Bắt đầu quét", type="primary"):
        ds = list(DEFAULT_TOP) if DANH_MUC_NHOM[nhom] == "DYNAMIC" else list(DANH_MUC_NHOM[nhom])
        if ma_rieng:
            for m in ma_rieng.split(","):
                m = m.strip()
                if m and m not in ds:
                    ds.insert(0, m)
        
        progress = st.progress(0)
        status = st.empty()
        results = []
        total = len(ds)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(scan_single_stock, ma): ma for ma in ds}
            done = 0
            for future in as_completed(futures):
                done += 1
                progress.progress(done / total)
                status.text(f"Đang quét: {done}/{total}")
                res = future.result()
                if res:
                    results.append(res)
        
        status.text(f"✅ Hoàn tất {total} mã")
        if results:
            df_res = pd.DataFrame(results).sort_values(["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
            if chi_hien_mua:
                df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP|SIDEWAY")]
            st.write(f"Tìm thấy **{len(df_res)}** mã phù hợp:")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning("Không lấy được dữ liệu. Thử lại sau.")

with tab3:
    st.subheader("📝 Nhật ký theo dõi & chấm điểm")
    with st.form("form_luu"):
        fc1, fc2, fc3 = st.columns(3)
        ma_luu = fc1.text_input("Mã muốn lưu:").upper().strip()
        ghi_chu = fc2.text_input("Ghi chú:", value="Bùng nổ Vol + Mua")
        submit = fc3.form_submit_button("➕ Lưu vào nhật ký", use_container_width=True)
    
    if submit and ma_luu:
        info = scan_single_stock(ma_luu)
        if info:
            now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%d/%m/%Y')
            st.session_state.nhat_ky_lenh.append({
                "Ngày quét": now_str,
                "Mã CP": ma_luu,
                "Giá vào (đ)": info["Giá hiện tại (đ)"],
                "Cắt lỗ ATR (đ)": info["Cắt lỗ ATR (đ)"],
                "Chốt lời ATR (đ)": info["Chốt lời ATR (đ)"],
                "Ghi chú": ghi_chu
            })
            st.success(f"Đã lưu {ma_luu} @ {info['Giá hiện tại (đ)']:,.0f} đ")
        else:
            st.error(f"Không lấy được dữ liệu {ma_luu}")
    
    if st.session_state.nhat_ky_lenh:
        bang = []
        for item in st.session_state.nhat_ky_lenh:
            ma = item["Mã CP"]
            gia_vao = float(item["Giá vào (đ)"])
            cat_lo = float(item["Cắt lỗ ATR (đ)"])
            chot_loi = float(item["Chốt lời ATR (đ)"])
            df_now = get_stock_data(ma, days=30)
            gia_tt = float(df_now['close'].iloc[-1]) if not df_now.empty else gia_vao
            lai_lo = ((gia_tt - gia_vao) / gia_vao) * 100 - 0.3
            
            if gia_tt >= chot_loi:
                ket_qua = "🎉 ĐẠT CHỐT LỜI"
            elif gia_tt <= cat_lo:
                ket_qua = "🛑 CHẠM CẮT LỖ"
            else:
                ket_qua = "⏳ ĐANG NẮM GIỮ"
            
            bang.append({
                "Ngày quét": item["Ngày quét"], "Mã CP": ma,
                "Giá vào (đ)": round(gia_vao), "Giá TT (đ)": round(gia_tt),
                "Cắt lỗ": round(cat_lo), "Chốt lời": round(chot_loi),
                "Lãi/Lỗ ròng (%)": f"{lai_lo:+.2f}%", "Trạng thái": ket_qua,
                "Ghi chú": item["Ghi chú"]
            })
        
        df_nk = pd.DataFrame(bang)
        st.dataframe(df_nk, use_container_width=True, hide_index=True)
        
        c1, c2 = st.columns(2)
        csv = df_nk.to_csv(index=False).encode('utf-8-sig')
        c1.download_button("📥 Tải CSV", data=csv, file_name="nhat_ky.csv", mime="text/csv")
        if c2.button("🗑️ Xóa toàn bộ nhật ký"):
            st.session_state.nhat_ky_lenh = []
            st.rerun()
