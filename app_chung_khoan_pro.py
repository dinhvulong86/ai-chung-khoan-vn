import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from vnstock import Quote

st.set_page_config(page_title="Hệ Thống AI Chứng Khoán VN (VnStock)", layout="wide")
st.title("🚀 Hệ Thống AI Phân Tích Kỹ Thuật & Quản Trị Danh Mục VN")
st.markdown("Tích hợp **Mây Ichimoku**, **Cắt lỗ động ATR**, **Đột biến Volume**, **Quét đa luồng** & **Nhật ký chấm điểm thực tế** — Nguồn dữ liệu: **VnStock (VCI)**")

if "nhat_ky_lenh" not in st.session_state:
    st.session_state.nhat_ky_lenh = [
        {"Ngày quét": "25/11/2026", "Mã CP": "HAG", "Giá vào (đ)": 13850, "Cắt lỗ ATR (đ)": 13314, "Chốt lời ATR (đ)": 14654, "Ghi chú": "Mẫu theo dõi Bùng nổ Vol"}
    ]

@st.cache_data(ttl=120, show_spinner=False)
def get_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Lấy dữ liệu lịch sử từ VnStock (VCI)"""
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
        
        quote = Quote(symbol=symbol.upper(), source="VCI")
        df = quote.history(start=start, end=end, interval="1D")
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Chuẩn hóa cột
        df = df.rename(columns={
            "time": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        
        # Chỉ giữ các cột cần thiết
        cols = ["time", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].copy()
        
        # Đảm bảo kiểu dữ liệu
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna().sort_values("time").reset_index(drop=True)
        
        # Lọc đúng số ngày gần nhất
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.warning(f"Lỗi lấy dữ liệu {symbol}: {e}")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["STD20"] = df["close"].rolling(window=20).std()
    df["BB_Upper"] = df["MA20"] + (2 * df["STD20"])
    df["BB_Lower"] = df["MA20"] - (2 * df["STD20"])
    df["Vol_MA20"] = df["volume"].rolling(window=20).mean()
    df["Vol_Ratio"] = (df["volume"] / df["Vol_MA20"].replace(0, np.nan)) * 100

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()

    # Ichimoku
    high_9 = df["high"].rolling(window=9).max()
    low_9 = df["low"].rolling(window=9).min()
    df["Tenkan"] = (high_9 + low_9) / 2
    high_26 = df["high"].rolling(window=26).max()
    low_26 = df["low"].rolling(window=26).min()
    df["Kijun"] = (high_26 + low_26) / 2
    df["Senkou_A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    high_52 = df["high"].rolling(window=52).max()
    low_52 = df["low"].rolling(window=52).min()
    df["Senkou_B"] = ((high_52 + low_52) / 2).shift(26)

    df["High_20"] = df["high"].rolling(window=20).max()
    df["Low_20"] = df["low"].rolling(window=20).min()
    df["Range_20_Pct"] = ((df["High_20"] - df["Low_20"]) / df["Low_20"]) * 100

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MFI
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = tp * df["volume"]
    pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
    neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
    mfr = pd.Series(pos_mf).rolling(14).sum() / pd.Series(neg_mf).rolling(14).sum().replace(0, np.nan)
    df["MFI"] = 100 - (100 / (1 + mfr))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

def analyze_one_stock(symbol: str, days_history: int, epochs: int):
    df = get_stock_data(symbol, days=days_history)
    if df.empty or len(df) < 100:
        st.error(f"Không đủ dữ liệu lịch sử cho mã **{symbol}**. Vui lòng kiểm tra lại mã cổ phiếu!")
        return

    df = add_technical_indicators(df)
    df_clean = df.dropna().copy()

    if len(df_clean) < 50:
        st.error(f"Dữ liệu sau khi tính chỉ báo quá ít cho mã {symbol}.")
        return

    # --- AI Prediction (minh họa) ---
    features = ["close", "volume", "MA20", "RSI", "MFI", "MACD"]
    data = df_clean[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    look_back = 30
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i].flatten())
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42, early_stopping=True)
    model.fit(X, y)

    last_30 = scaled_data[-look_back:].flatten().reshape(1, -1)
    pred_scaled = model.predict(last_30)[0]
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = pred_scaled
    predicted_price = scaler.inverse_transform(dummy)[0, 0]

    # --- Các chỉ số hiện tại ---
    current_price = float(df_clean["close"].iloc[-1])
    prev_close = float(df_clean["close"].iloc[-2])
    vol_ratio = float(df_clean["Vol_Ratio"].iloc[-1])
    latest_atr = float(df_clean["ATR"].iloc[-1])
    pct_change = ((predicted_price - current_price) / current_price) * 100
    latest_rsi = float(df_clean["RSI"].iloc[-1])
    latest_mfi = float(df_clean["MFI"].iloc[-1])
    latest_macd = float(df_clean["MACD"].iloc[-1])
    latest_sig = float(df_clean["Signal_Line"].iloc[-1])
    latest_ma20 = float(df_clean["MA20"].iloc[-1])
    bb_upper = float(df_clean["BB_Upper"].iloc[-1])
    bb_lower = float(df_clean["BB_Lower"].iloc[-1])
    senkou_a = float(df_clean["Senkou_A"].iloc[-1])
    senkou_b = float(df_clean["Senkou_B"].iloc[-1])
    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    range_20 = float(df_clean["Range_20_Pct"].iloc[-1])
    high_20 = float(df_clean["High_20"].iloc[-1])
    low_20 = float(df_clean["Low_20"].iloc[-1])

    is_sideway = (range_20 < 9.5) and (abs(current_price - latest_ma20) / latest_ma20 < 0.025)
    is_bull_trap = (current_price > prev_close * 1.02) and (vol_ratio < 75.0) and (latest_rsi > 65)
    is_vol_breakout = (current_price > prev_close * 1.015) and (vol_ratio >= 150.0) and (current_price > latest_ma20)

    gia_cat_lo_atr = current_price - (2.0 * latest_atr)
    gia_chot_loi_atr = current_price + (3.0 * latest_atr)
    pct_cat_lo = ((gia_cat_lo_atr - current_price) / current_price) * 100
    pct_chot_loi = ((gia_chot_loi_atr - current_price) / current_price) * 100

    # --- Hiển thị ---
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Giá hiện tại", f"{current_price:,.0f} đ", f"Vol: {vol_ratio:.0f}% TB20")
    m2.metric("AI Dự đoán tới", f"{predicted_price:,.0f} đ", f"{pct_change:+.2f}%")
    m3.metric("RSI / MFI", f"{latest_rsi:.0f} / {latest_mfi:.0f}",
              "Quá mua" if latest_rsi > 70 else ("Quá bán" if latest_rsi < 30 else "Vùng an toàn"), delta_color="off")
    ichi_status = "Trên mây (Tăng)" if current_price > cloud_top else ("Dưới mây (Giảm)" if current_price < cloud_bot else "Trong mây (Giằng co)")
    m4.metric("Vị thế Mây Ichimoku", ichi_status, f"Mây: {cloud_bot:,.0f} - {cloud_top:,.0f}", delta_color="off")
    m5.metric("Biên độ ATR (14)", f"{latest_atr:,.0f} đ", f"±{(latest_atr/current_price)*100:.1f}% / phiên", delta_color="off")

    st.subheader("🤖 Phân Tích Chuyên Sâu & Khuyến Nghị Hành Động")
    if is_vol_breakout:
        st.success(f"🔥 **BÙNG NỔ KHỐI LƯỢNG**: Vol đạt {vol_ratio:.0f}% TB20 kèm giá bứt phá mạnh!")
    if is_bull_trap:
        st.error(f"🚨 **CẢNH BÁO BẪY TĂNG GIÁ**: Giá tăng nhưng Vol chỉ {vol_ratio:.0f}% + RSI cao. Không mua đuổi!")

    diem_mua = (1 if pct_change > 0.5 else 0) + (1 if latest_macd > latest_sig else 0) + \
               (1 if current_price > cloud_top else 0) + (1 if (current_price > latest_ma20 and vol_ratio >= 100) else 0)
    diem_ban = (1 if pct_change < -0.5 else 0) + (1 if latest_macd <= latest_sig else 0) + \
               (2 if (latest_rsi > 70 or latest_mfi > 80) else 0) + (1 if current_price < cloud_bot else 0)

    if is_sideway:
        st.warning(f"⏸️ **THỊ TRƯỜNG ĐI NGANG**: Biên độ 20 phiên {range_20:.1f}% (hộp {low_20:,.0f} – {high_20:,.0f}). "
                   f"Ưu tiên mua gần BB Dưới ({bb_lower:,.0f}) và chốt gần BB Trên ({bb_upper:,.0f}).")
    elif diem_ban >= 3:
        st.error(f"🔴 **KHUYẾN NGHỊ: BÁN / CHỐT LỜI** (Điểm rủi ro BÁN: {diem_ban}/5)")
    elif diem_mua >= 3 and diem_ban <= 1:
        st.success(f"🟢 **KHUYẾN NGHỊ: CÂN NHẮC MUA MỚI** (Điểm MUA: {diem_mua}/4)")
    elif diem_ban == 2:
        st.warning(f"🟠 **KHUYẾN NGHỊ: CANH BÁN TỪNG PHẦN / KHÔNG MUA ĐUỔI**")
    else:
        st.info("🟡 **KHUYẾN NGHỊ: NẮM GIỮ & QUAN SÁT**")

    st.markdown("#### 🎯 Kế hoạch giao dịch theo ATR(14)")
    p1, p2, p3 = st.columns(3)
    p1.info(f"**Vùng mua an toàn**\n\n{bb_lower:,.0f} – {latest_ma20:,.0f} đ")
    p2.success(f"**Chốt lời (+3 ATR)**\n\n{gia_chot_loi_atr:,.0f} đ ({pct_chot_loi:+.1f}%)")
    p3.error(f"**Cắt lỗ (-2 ATR)**\n\nThủng {gia_cat_lo_atr:,.0f} đ ({pct_cat_lo:+.1f}%)")

    # Biểu đồ
    ngay_moi_nhat = df_clean["time"].iloc[-1]
    st.subheader(f"Biểu đồ kỹ thuật {symbol} — Mây Ichimoku & Bollinger (đến {ngay_moi_nhat})")
    df_plot = df_clean.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot["time"], open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"], name="Giá nến"
    ))
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["MA20"], line=dict(color="orange", width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["Senkou_A"], line=dict(color="rgba(0,200,100,0.4)", width=1), name="Senkou A"))
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["Senkou_B"],
        line=dict(color="rgba(255,80,80,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(144,238,144,0.2)", name="Mây Ichimoku"
    ))
    fig.add_hline(y=gia_chot_loi_atr, line_dash="dot", line_color="green", annotation_text="Chốt lời (+3 ATR)")
    fig.add_hline(y=gia_cat_lo_atr, line_dash="dot", line_color="red", annotation_text="Cắt lỗ (-2 ATR)")
    fig.update_xaxes(type="category", tickangle=-45, nticks=40)
    fig.update_layout(xaxis_rangeslider_visible=False, height=560, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def scan_single_stock(ma: str):
    try:
        df_ma = get_stock_data(ma, days=180)
        if len(df_ma) < 80:
            return None
        df_ma = add_technical_indicators(df_ma).dropna()
        if df_ma.empty:
            return None

        last = df_ma.iloc[-1]
        prev = df_ma.iloc[-2]
        curr_p = float(last["close"])
        prev_p = float(prev["close"])
        vol_ratio = float(last["Vol_Ratio"])
        atr_val = float(last["ATR"])
        cloud_top = max(float(last["Senkou_A"]), float(last["Senkou_B"]))
        cloud_bot = min(float(last["Senkou_A"]), float(last["Senkou_B"]))

        is_vol_breakout = (curr_p > prev_p * 1.015) and (vol_ratio >= 150.0) and (curr_p > float(last["MA20"]))
        buy_score = (1 if last["MACD"] > last["Signal_Line"] else 0) + \
                    (1 if curr_p > float(last["MA20"]) else 0) + \
                    (1 if curr_p > cloud_top else 0) + \
                    (1 if vol_ratio >= 110.0 else 0)
        sell_score = (1 if last["MACD"] < last["Signal_Line"] else 0) + \
                     (1 if curr_p < cloud_bot else 0) + \
                     (1 if (last["RSI"] > 70 or last["MFI"] > 80) else 0)
        range_20 = float(last["Range_20_Pct"])
        is_sideway = (range_20 < 9.5) and (abs(curr_p - float(last["MA20"])) / float(last["MA20"]) < 0.025)

        if is_vol_breakout:
            trang_thai = "🔥 BÙNG NỔ VOL + MUA"
        elif is_sideway:
            trang_thai = "⏸️ SIDEWAY (ĐI NGANG)"
        elif buy_score >= 3 and sell_score == 0:
            trang_thai = "🟢 TÍN HIỆU MUA ĐẸP"
        elif sell_score >= 2 or last["RSI"] > 72 or last["MFI"] > 80:
            trang_thai = "🔴 CẢNH BÁO BÁN / CHỐT LỜI"
        elif buy_score == 2:
            trang_thai = "🟡 NẮM GIỮ / THEO DÕI"
        else:
            trang_thai = "⚪ ĐỨNG NGOÀI"

        return {
            "Mã CP": ma,
            "Giá hiện tại (đ)": round(curr_p, 0),
            "Cắt lỗ ATR (đ)": round(curr_p - 2 * atr_val, 0),
            "Chốt lời ATR (đ)": round(curr_p + 3 * atr_val, 0),
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
    'CEO', 'DIG', 'DXG', 'HDC', 'HDG', 'HUT', 'IDC', 'KBC', 'KDH', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VRE',
    'HPG', 'HSG', 'NKG', 'BSR', 'GAS', 'PLX', 'POW', 'PVD', 'PVS', 'FPT', 'CMG', 'CTR', 'DGC', 'DCM', 'DPM',
    'MWG', 'MSN', 'PNJ', 'FRT', 'VNM', 'SAB', 'HAG', 'ANV', 'VHC', 'GMD', 'HAH', 'VJC', 'HVN', 'CTD', 'VCG'
)

DANH_MUC_NHOM = {
    "🌐 Top cổ phiếu thanh khoản cao (mặc định)": DEFAULT_TOP,
    "🏛️ Nhóm Chứng khoán": ('MBS', 'SHS', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'CTS', 'AGR', 'VIX', 'ORS', 'BVS'),
    "🏦 Nhóm Ngân hàng": ('ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'NAB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB'),
    "🏗️ Bất động sản & KCN": ('CEO', 'DIG', 'DXG', 'HDC', 'HDG', 'HUT', 'IDC', 'KBC', 'KDH', 'NLG', 'NVL', 'PDR', 'SCR', 'SZC', 'TCH', 'VHM', 'VIC', 'VRE'),
    "⚙️ Thép - Dầu khí - Công nghệ - Tiêu dùng": ('HPG', 'HSG', 'NKG', 'BSR', 'GAS', 'PLX', 'POW', 'FPT', 'CMG', 'CTR', 'DGC', 'MWG', 'MSN', 'PNJ', 'FRT', 'VNM', 'HAG')
}

# ====================== GIAO DIỆN ======================
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Chuyên Sâu", "🔍 Quét Đột Biến Dòng Tiền", "📝 Nhật Ký & Chấm Điểm"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.text_input("Nhập mã cổ phiếu (HOSE/HNX/UPCOM):", value="MBS").upper().strip()
    days_history = col2.selectbox("Dữ liệu lịch sử (ngày):", (300, 500, 730, 1000), index=1)
    epochs = col3.selectbox("Số vòng học AI:", (80, 150, 250), index=1)
    col4.write("")
    col4.write("")
    run_btn = col4.button("🚀 Chạy Phân Tích Ngay", use_container_width=True, type="primary")

    if run_btn and symbol:
        with st.spinner(f"Đang phân tích {symbol}..."):
            analyze_one_stock(symbol, days_history, epochs)

with tab2:
    st.subheader("Quét Đột Biến Volume + Ichimoku + ATR")
    c1, c2 = st.columns(2)
    nhom_chon = c1.selectbox("Chọn nhóm cổ phiếu:", list(DANH_MUC_NHOM.keys()))
    ma_rieng = c2.text_input("Thêm mã riêng (cách nhau dấu phẩy):", placeholder="VD: BSR, KBC, VGC").upper().strip()
    chi_hien_mua = st.checkbox("Chỉ hiện 🔥 BÙNG NỔ VOL / 🟢 MUA ĐẸP / ⏸️ SIDEWAY", value=True)
    scan_btn = st.button("⚡ Bắt Đầu Quét Siêu Tốc", type="primary")

    if scan_btn:
        ds_quet = list(DANH_MUC_NHOM[nhom_chon])
        if ma_rieng:
            for m in ma_rieng.split(","):
                m = m.strip()
                if m and m not in ds_quet:
                    ds_quet.insert(0, m)

        progress = st.progress(0)
        status = st.empty()
        ket_qua = []
        tong = len(ds_quet)
        da_xong = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(scan_single_stock, ma): ma for ma in ds_quet}
            for future in as_completed(futures):
                da_xong += 1
                progress.progress(da_xong / tong)
                status.text(f"⚡ Đang quét: {da_xong}/{tong} mã...")
                res = future.result()
                if res:
                    ket_qua.append(res)

        status.text(f"✅ Hoàn tất quét {tong} mã!")
        if ket_qua:
            df_res = pd.DataFrame(ket_qua).sort_values(by=["Khuyến nghị", "Điểm Mua"], ascending=[True, False])
            if chi_hien_mua:
                df_res = df_res[df_res["Khuyến nghị"].str.contains("BÙNG NỔ|MUA ĐẸP|SIDEWAY")]
            st.write(f"Tìm thấy **{len(df_res)}** mã phù hợp:")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            st.warning("Không lấy được dữ liệu. Thử lại sau vài phút.")

with tab3:
    st.subheader("📝 Nhật Ký Theo Dõi & Chấm Điểm Thực Tế")
    st.write("Lưu các mã báo **Mua đẹp / Bùng nổ Vol** để hệ thống tự động đối chiếu giá thị trường và tính lãi/lỗ (đã trừ 0,3% phí).")

    with st.form("form_them_ma"):
        fc1, fc2, fc3 = st.columns([2, 3, 1])
        ma_luu = fc1.text_input("Mã muốn lưu:", placeholder="HAG").upper().strip()
        ghi_chu = fc2.text_input("Ghi chú tín hiệu:", value="Bùng nổ Vol + Mua")
        submit = fc3.form_submit_button("➕ Lưu vào Nhật ký", use_container_width=True)

    if submit and ma_luu:
        info = scan_single_stock(ma_luu)
        if info:
            now_str = datetime.now().strftime("%d/%m/%Y")
            st.session_state.nhat_ky_lenh.append({
                "Ngày quét": now_str,
                "Mã CP": ma_luu,
                "Giá vào (đ)": info["Giá hiện tại (đ)"],
                "Cắt lỗ ATR (đ)": info["Cắt lỗ ATR (đ)"],
                "Chốt lời ATR (đ)": info["Chốt lời ATR (đ)"],
                "Ghi chú": ghi_chu
            })
            st.success(f"Đã lưu **{ma_luu}** (Giá vào: {info['Giá hiện tại (đ)']:,.0f} đ)")
        else:
            st.error(f"Không lấy được dữ liệu mã {ma_luu}")

    if st.session_state.nhat_ky_lenh:
        bang = []
        so_thang = 0
        tong_dong = 0
        for item in st.session_state.nhat_ky_lenh:
            ma = item["Mã CP"]
            gia_vao = float(item["Giá vào (đ)"])
            cat_lo = float(item["Cắt lỗ ATR (đ)"])
            chot_loi = float(item["Chốt lời ATR (đ)"])
            df_now = get_stock_data(ma, days=30)
            gia_tt = float(df_now["close"].iloc[-1]) if not df_now.empty else gia_vao
            lai_lo = ((gia_tt - gia_vao) / gia_vao) * 100 - 0.3

            if gia_tt >= chot_loi:
                ket_qua = "🎉 ĐẠT CHỐT LỜI ATR"
                so_thang += 1
                tong_dong += 1
            elif gia_tt <= cat_lo:
                ket_qua = "🛑 CHẠM CẮT LỖ ATR"
                tong_dong += 1
            else:
                ket_qua = "⏳ ĐANG NẮM GIỮ"

            bang.append({
                "Ngày quét": item["Ngày quét"],
                "Mã CP": ma,
                "Giá vào (đ)": round(gia_vao),
                "Giá TT (đ)": round(gia_tt),
                "Cắt lỗ ATR": round(cat_lo),
                "Chốt lời ATR": round(chot_loi),
                "Lãi/Lỗ ròng (%)": f"{lai_lo:+.2f}%",
                "Trạng thái": ket_qua,
                "Ghi chú": item["Ghi chú"]
            })

        df_nk = pd.DataFrame(bang)
        st.dataframe(df_nk, use_container_width=True, hide_index=True)

        if tong_dong > 0:
            st.metric("Tỷ lệ thắng (đã đóng lệnh)", f"{so_thang}/{tong_dong} ({so_thang/tong_dong*100:.0f}%)")

        c1, c2 = st.columns(2)
        csv = df_nk.to_csv(index=False).encode("utf-8-sig")
        c1.download_button("📥 Tải Nhật Ký (CSV)", data=csv, file_name="nhat_ky_chung_khoan.csv", mime="text/csv")
        if c2.button("🗑️ Xóa toàn bộ nhật ký"):
            st.session_state.nhat_ky_lenh = []
            st.rerun()
