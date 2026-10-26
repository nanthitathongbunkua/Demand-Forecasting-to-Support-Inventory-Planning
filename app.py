import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Retail Decision Support System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# 2. Custom CSS (Modern Pastel)
# ---------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #F4F5F7; }
    [data-testid="stSidebar"] { background-color: #121212 !important; padding-top: 1rem; }
    [data-testid="stSidebar"] * { color: #E0E0E0 !important; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; color: #111827; font-size: 2rem !important; }
    .metric-card { padding: 20px; border-radius: 18px; margin-bottom: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.03); }
    .card-blue { background-color: #EBF3FE; color: #1E3A8A; }
    .card-purple { background-color: #F3E8FF; color: #581C87; }
    .card-green { background-color: #DCFCE7; color: #14532D; }
    .card-yellow { background-color: #FEF9C3; color: #713F12; }
    .card-title { font-size: 0.85rem; font-weight: 600; opacity: 0.8; margin-bottom: 4px; }
    .card-value { font-size: 1.5rem; font-weight: 800; }
    .card-badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem;
                  font-weight: 700; margin-top: 8px; background-color: rgba(255,255,255,0.6); }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = ["Date", "Description", "Quantity", "Sales_Value"]

# ---------------------------------------------------------------
# 3. ฟังก์ชันโหลดและประมวลผลข้อมูล
# ---------------------------------------------------------------
@st.cache_data
def load_csv(name):
    return pd.read_csv(f"app_data/{name}")

@st.cache_data
def load_default_daily():
    d = pd.read_csv("app_data/daily_sales.csv.gz")
    d["Date"] = pd.to_datetime(d["Date"])
    return d

def load_uploaded(file):
    try:
        if file.name.endswith(".csv"):
            d = pd.read_csv(file)
        else:
            d = pd.read_excel(file, sheet_name=0)
        miss = [c for c in REQUIRED_COLS if c not in d.columns]
        if miss:
            st.sidebar.error(f"ไฟล์ขาดคอลัมน์: {', '.join(miss)}")
            return None
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Description"] = d["Description"].astype(str).str.strip()
        return d.dropna(subset=["Date"])
    except Exception as e:
        st.sidebar.error(f"อ่านไฟล์ไม่ได้: {e}")
        return None

# ฟังก์ชันประมวลผลข้อมูลที่อัปโหลดแบบ Dynamic 100%
@st.cache_data
def process_dynamic_upload(df):
    df = df.copy()  # ไม่แก้ไขข้อมูลต้นฉบับของผู้เรียก
    rule_cols = ['period', 'antecedents', 'consequents', 'support', 'confidence', 'lift']
    rules_df = pd.DataFrame(columns=rule_cols)

    # 1. Product Bundling (FP-Growth): 1 Invoice = 1 ตะกร้า ตามนิยามในรายงาน
    #    ถ้าไฟล์ไม่มีคอลัมน์ Invoice จะไม่วิเคราะห์ (การใช้วันที่แทนตะกร้าไม่ถูกต้อง)
    if 'Invoice' in df.columns:
        tx = df.groupby('Invoice')['Description'].apply(lambda x: list(set(x))).tolist()
        te = TransactionEncoder()
        arr = te.fit(tx).transform(tx, sparse=True)
        enc = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_)
        its = fpgrowth(enc, min_support=0.01, use_colnames=True, max_len=3)  # จำกัดความยาวชุด ป้องกันหน่วยความจำเต็มเมื่อข้อมูลน้อย
        if not its.empty:
            r = association_rules(its, metric='confidence', min_threshold=0.1)
            r = r[r['lift'] > 1.0].copy()
            r['period'] = 'Uploaded Store'
            r['antecedents'] = r['antecedents'].apply(lambda s: ' + '.join(sorted(s)))
            r['consequents'] = r['consequents'].apply(lambda s: ' + '.join(sorted(s)))
            rules_df = r[rule_cols].sort_values('lift', ascending=False).head(2000)  # แสดงสูงสุด 2,000 กฎที่ Lift สูงสุด
        
    # 2. Demand Forecast (Holt-Winters)
    df['Month_Period'] = df['Date'].dt.to_period('M').astype(str)
    monthly_q = df.groupby('Month_Period')['Quantity'].sum()
    
    forecast_bt = pd.DataFrame(columns=['Month', 'Actual', 'Forecast'])
    forecast_fut = pd.DataFrame(columns=['Month', 'Forecast'])
    meta_dict = {'mae_total': 0, 'rmse_total': 0, 'mape_total': 0, 'wape_total': 0}
    
    if len(monthly_q) >= 4:
        try:
            train_len = max(2, len(monthly_q) - 3)
            train = monthly_q.iloc[:train_len]
            test = monthly_q.iloc[train_len:]
            
            seasonal_per = 12 if len(train) >= 12 else None   # ข้อมูลรายเดือน: ฤดูกาลรายปี = 12 เดือน
            if seasonal_per:
                model = ExponentialSmoothing(train.clip(lower=0.1), seasonal='add', seasonal_periods=seasonal_per, initialization_method='legacy-heuristic').fit()
            else:
                model = ExponentialSmoothing(train.clip(lower=0.1), initialization_method='legacy-heuristic').fit()
                
            preds = model.forecast(len(test)).values
            forecast_bt = pd.DataFrame({'Month': test.index, 'Actual': test.values, 'Forecast': np.clip(preds, 0, None)})
            
            if len(test) > 0:
                act = forecast_bt['Actual']
                fore = forecast_bt['Forecast']
                meta_dict['mae_total'] = float(np.mean(np.abs(act - fore)))
                meta_dict['rmse_total'] = float(np.sqrt(np.mean((act - fore) ** 2)))
                meta_dict['mape_total'] = float(np.mean(np.abs((act - fore) / act.replace(0, 1))) * 100)
                meta_dict['wape_total'] = float(np.abs(act - fore).sum() / act.sum() * 100) if act.sum() > 0 else 0
            
            fut_idx = pd.period_range(pd.Period(monthly_q.index[-1]) + 1, periods=3, freq='M').astype(str)
            if seasonal_per and len(monthly_q) >= seasonal_per:
                fut_model = ExponentialSmoothing(monthly_q.clip(lower=0.1), seasonal='add', seasonal_periods=seasonal_per, initialization_method='legacy-heuristic').fit()
            else:
                fut_model = ExponentialSmoothing(monthly_q.clip(lower=0.1), initialization_method='legacy-heuristic').fit()
            fut_preds = fut_model.forecast(3).values
            forecast_fut = pd.DataFrame({'Month': fut_idx, 'Forecast': np.clip(fut_preds, 0, None)})
        except Exception:
            pass
            
    meta_dict = {k: round(v, 1) for k, v in meta_dict.items()}
    return rules_df, forecast_bt, forecast_fut, meta_dict

def style_ax(ax):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------
# 4. Sidebar
# ---------------------------------------------------------------
st.sidebar.markdown("<h2 style='color:white;font-weight:800;'>🛒 Retail Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = st.sidebar.radio("NAVIGATION", ["Overview", "Top Products", "Product Bundling", "Demand Forecast"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("📁 อัปโหลดข้อมูลร้านอื่น")
uploaded = st.sidebar.file_uploader(
    "อัปโหลดไฟล์เพื่อเปลี่ยนข้อมูลทั้งระบบ\n"
    "ต้องมีคอลัมน์ Date, Description, Quantity, Sales_Value\n"
    "(ถ้ามีคอลัมน์ Invoice จะวิเคราะห์ Product Bundling ได้ด้วย)",
    type=["xlsx", "xls", "csv"],
)

df_data = load_uploaded(uploaded) if uploaded is not None else None
if df_data is not None and df_data.empty:
    st.sidebar.error("ไฟล์ที่อัปโหลดไม่มีข้อมูลที่ใช้ได้ (ตรวจสอบคอลัมน์ Date)")
    df_data = None
using_upload = df_data is not None

if using_upload:
    rules, forecast_bt, forecast_fut, meta = process_dynamic_upload(df_data)
else:
    df_data = load_default_daily()
    rules = load_csv("rules.csv")
    forecast_bt = load_csv("forecast_backtest.csv")
    forecast_fut = load_csv("forecast_future.csv")
    meta = load_csv("meta.csv").iloc[0]
    sku_fc = load_csv("forecast_top10_sku.csv")

# ---------------------------------------------------------------
# 5. หัวข้อ + การ์ดสรุป
# ---------------------------------------------------------------
st.markdown("<h1>📊 Dashboard & Decision Support</h1>", unsafe_allow_html=True)
if using_upload:
    st.caption("🟢 กำลังใช้ข้อมูลจากไฟล์ร้านค้าที่คุณอัปโหลด (ทุกหน้าอัปเดตแบบ Dynamic)")
else:
    st.caption("🔵 ข้อมูลโครงการ: Online Retail II (1 ธ.ค. 2009 – 9 ธ.ค. 2011)")

total_sales = df_data["Sales_Value"].sum()
total_qty = df_data["Quantity"].sum()
n_products = df_data["Description"].nunique()
d_min, d_max = df_data["Date"].min().date(), df_data["Date"].max().date()

cards = [
    ("card-blue", "TOTAL REVENUE", f"£ {total_sales:,.0f}", "↗ ยอดขายรวม"),
    ("card-purple", "TOTAL QUANTITY SOLD", f"{total_qty:,.0f} pcs", "🛒 จำนวนชิ้นที่ขาย"),
    ("card-green", "UNIQUE PRODUCTS", f"{n_products:,}", "📦 จำนวนสินค้า"),
    ("card-yellow", "PERIOD", f"{d_min:%b %Y} – {d_max:%b %Y}", "🗓 ช่วงข้อมูล"),
]
for col, (cls, title, value, badge) in zip(st.columns(4), cards):
    with col:
        st.markdown(
            f"""<div class="metric-card {cls}"><div class="card-title">{title}</div>
            <div class="card-value">{value}</div><div class="card-badge">{badge}</div></div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------
# 6. เนื้อหาตามเมนู
# ---------------------------------------------------------------
if menu == "Overview":
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### 📈 Sales Trend & Seasonality")
        monthly = df_data.groupby(df_data["Date"].dt.to_period("M"))["Sales_Value"].sum().reset_index()
        monthly["Date"] = monthly["Date"].astype(str)
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="none")
        style_ax(ax)
        ax.plot(monthly["Date"], monthly["Sales_Value"], color="#2563EB", marker="o", linewidth=2.5)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        ax.set_ylabel("Sales Value (£)")
        st.pyplot(fig)
        if not using_upload:
            st.caption("หมายเหตุ: เดือนธันวาคมมีข้อมูลไม่ครบเดือน (ถึงวันที่ 23 ในปี 2009–2010 และวันที่ 9 ในปี 2011)")
    with right:
        if not df_data.empty:
            df_temp = df_data.copy()
            df_temp["Month_Name"] = df_temp["Date"].dt.strftime("%B")
            peak_month = df_temp.groupby("Month_Name")["Sales_Value"].sum().idxmax()
            
            insight_text = (
                f"ยอดขายพุ่งสูงเป็นพิเศษในช่วงเดือน <b>{peak_month}</b> "
                f"แนะนำให้ตรวจสอบคู่สินค้าที่ถูกซื้อร่วมกันเพื่อจัดโปรโมชันพ่วง (Bundling)"
            )
        else:
            insight_text = "แนะนำตรวจสอบคู่สินค้าที่มีค่า Lift สูง เพื่อจัดโปรโมชันพ่วง (Bundling)"

        st.markdown(f"""
        <div style="background-color:#111827;color:white;padding:25px;border-radius:20px;">
            <h3 style="color:white;font-size:1.2rem;margin-bottom:10px;">💡 Seasonal Bundle Insight</h3>
            <p style="font-size:0.9rem;color:#9CA3AF;">{insight_text}<br><br>
            ดูรายละเอียดเพิ่มเติมได้ที่เมนู <b>Product Bundling</b></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📅 ยอดขายตามวันในสัปดาห์")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df_data.groupby(df_data["Date"].dt.day_name())["Sales_Value"].sum().reindex(order).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(8, 3), facecolor="none")
    style_ax(ax2)
    ax2.bar(dow.index, dow.values, color="#8B5CF6")
    ax2.set_ylabel("Sales Value (£)")
    plt.xticks(fontsize=8)
    st.pyplot(fig2)

    st.markdown("### 📋 ข้อมูลยอดขายรายวันต่อสินค้า")
    term = st.text_input("🔍 ค้นหา (ชื่อสินค้า รหัส หรือวันที่):", "")
    view = df_data
    if term:
        mask = df_data.astype(str).apply(lambda x: x.str.contains(term, case=False, regex=False)).any(axis=1)
        view = df_data[mask]
    st.caption(f"พบ {len(view):,} รายการ")
    st.dataframe(view, height=400)

elif menu == "Top Products":
    st.markdown("### 🏆 Top Selling Products")
    top_n = st.slider("จำนวนอันดับที่แสดง:", 5, 50, 10)
    top = (df_data.groupby("Description")["Quantity"].sum().reset_index()
           .nlargest(top_n, "Quantity").sort_values("Quantity"))
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.32)), facecolor="none")
    style_ax(ax)
    ax.barh(top["Description"], top["Quantity"], color="#8B5CF6")
    ax.set_xlabel("Quantity (pcs)")
    plt.yticks(fontsize=8)
    st.pyplot(fig)
    st.markdown("### 📋 ตารางสรุป")
    st.dataframe(top.sort_values("Quantity", ascending=False), hide_index=True)

elif menu == "Product Bundling":
    st.markdown("### 🔗 Market Basket Association Rules (FP-Growth)")
    if using_upload:
        st.info("ผลวิเคราะห์จากไฟล์ที่อัปโหลด: Support ≥ 0.01, Confidence ≥ 0.1, Lift > 1 (1 Invoice = 1 ตะกร้า)")
        if 'Invoice' not in df_data.columns:
            st.warning("ไฟล์ที่อัปโหลดไม่มีคอลัมน์ Invoice จึงไม่สามารถวิเคราะห์ Product Bundling ได้")
    else:
        st.info("ผลจาก FP-Growth บนข้อมูลโครงการ: Support ≥ 0.02, Confidence ≥ 0.2, Lift > 1 "
                "แบ่งเป็น Regular (ม.ค.–ต.ค.) และ Festive (พ.ย.–ธ.ค.)")
    
    with st.container(border=True):
        st.markdown("#### ⚙️ ตัวกรองข้อมูล (Filters)")
        c1, c2 = st.columns([1, 2])
        with c1:
            if not using_upload:
                period = st.selectbox("ช่วงเวลา:", ["Festive", "Regular"],
                                      format_func=lambda x: "Festive (พ.ย.–ธ.ค.)" if x == "Festive" else "Regular (ม.ค.–ต.ค.)")
                rules_filtered = rules[rules["period"] == period]
            else:
                rules_filtered = rules
        with c2:
            if not rules_filtered.empty and "antecedents" in rules_filtered.columns:
                items = set()
                for col_name in ["antecedents", "consequents"]:
                    for v in rules_filtered[col_name]:
                        items.update(v.split(" + "))
                all_items = sorted(items)
                selected_product = st.selectbox("🔍 เลือกสินค้า:", ["-- ทั้งหมด (All Products) --"] + all_items)
            else:
                selected_product = "-- ทั้งหมด (All Products) --"
        
    min_lift = 1.0
    view = rules_filtered[rules_filtered["lift"] >= min_lift] if not rules_filtered.empty else rules_filtered
    if selected_product != "-- ทั้งหมด (All Products) --" and not view.empty:
        view = view[
            view["antecedents"].apply(lambda v: selected_product in v.split(" + ")) |
            view["consequents"].apply(lambda v: selected_product in v.split(" + "))
        ]
        
    st.write(f"พบ {len(view)} กฎ (Lift > 1)")
    if not view.empty:
        st.dataframe(view.drop(columns="period", errors="ignore").round(3), hide_index=True, height=450)
    else:
        st.warning("ไม่พบกฎความสัมพันธ์ในเงื่อนไขนี้ (ข้อมูลอาจน้อยเกินไปสำหรับการทำ Market Basket Analysis)")
    if not using_upload:
        st.caption("คู่ที่เด่นส่วนใหญ่เป็นสินค้าคอลเลกชันเดียวกันต่างสี/ลาย จึงเหมาะกับการจัดเป็นชุด (Bundle) "
                   "ช่วง Festive ใช้ข้อมูลธันวาคมเท่าที่มี (ไม่ครบเดือน)")

else:  # Demand Forecast
    st.markdown("### 📈 Demand Forecasting")
    mode = st.radio("มุมมอง:", ["🌐 ยอดรวมรายเดือน (Holt-Winters)", "📦 รายสินค้า"], horizontal=True)
    st.markdown("---")

    if mode.startswith("🌐"):
        if using_upload:
            st.success("Holt-Winters รายเดือน | ทดสอบ 3 เดือนสุดท้ายของไฟล์ | หน่วย: จำนวนชิ้น")
            st.caption("เดือนสุดท้ายของไฟล์อาจไม่ครบเดือน ซึ่งทำให้ค่าความคลาดเคลื่อนสูงกว่าความเป็นจริง")
        else:
            st.success("Holt-Winters รายเดือน (รอบฤดูกาล 12 เดือน) | ฝึก 18 เดือน ทดสอบ 6 เดือน (ก.ค.–พ.ย. 2011) | หน่วย: จำนวนชิ้น")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{meta.get('mae_total', 0):,.0f} pcs")
        k2.metric("RMSE", f"{meta.get('rmse_total', 0):,.0f} pcs")
        k3.metric("MAPE", f"{meta.get('mape_total', 0)}%")
        k4.metric("WAPE", f"{meta.get('wape_total', 0)}%")
        
        if not forecast_bt.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="none")
            style_ax(ax)
            ax.plot(forecast_bt["Month"], forecast_bt["Actual"], marker="o", color="#2563EB", linewidth=2, label="Actual")
            ax.plot(forecast_bt["Month"], forecast_bt["Forecast"], marker="x", linestyle="--", color="#F59E0B", linewidth=2, label="Holt-Winters")
            ax.set_ylabel("Quantity (pcs)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("ข้อมูลรายเดือนไม่เพียงพอสำหรับการทำ Backtest (แนะนำให้มีข้อมูลอย่างน้อย 4 เดือนขึ้นไป)")

        st.markdown("#### พยากรณ์ล่วงหน้า 3 เดือน")
        if not forecast_fut.empty:
            st.dataframe(forecast_fut.rename(columns={"Forecast": "Forecast (pcs)"}).round(0), hide_index=True)
        else:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับพยากรณ์ล่วงหน้า")
        if not using_upload:
            st.caption("ข้อมูลมี 24 เดือนเต็ม จึงเรียนรู้ฤดูกาลได้ไม่ครบ 2 รอบเต็ม ควรใช้เป็นแนวทางภาพรวม "
                       "การพยากรณ์ตัดสินค้า 23843 (ออเดอร์เดียว 80,995 ชิ้น) ออก")
    else:
        if not using_upload:
            st.markdown("#### Top 10 สินค้า: Moving Average 3 เดือน")
            st.dataframe(sku_fc.rename(columns={"M-2": "3 เดือนก่อน", "M-1": "2 เดือนก่อน", "M0": "เดือนล่าสุด",
                                                "Forecast_next_month": "พยากรณ์เดือนถัดไป (pcs)",
                                                "WAPE_backtest_%": "WAPE ทดสอบ (%)"}), hide_index=True)
            st.warning("รายสินค้ามีความคลาดเคลื่อนสูง (ดู WAPE) Moving Average ให้ผลดีกว่า Holt-Winters ในระดับสินค้ารายตัว "
                       "ใช้เป็นแนวทาง ไม่ใช่ตัวเลขสั่งซื้อจริง")
        st.markdown("#### ดูแนวโน้มยอดขายรายสินค้า (ข้อมูลจริง)")
        products = sorted(df_data["Description"].dropna().unique().tolist())
        sel = st.selectbox("🔍 เลือกหรือพิมพ์ชื่อสินค้า:", products)
        s = df_data[df_data["Description"] == sel].groupby("Date")["Quantity"].sum()
        s = s.asfreq("D", fill_value=0)
        roll = s.rolling(7, min_periods=1).mean()
        m1, m2 = st.columns(2)
        m1.metric("📦 ปริมาณขายสะสม", f"{s.sum():,.0f} ชิ้น")
        m2.metric("📊 ยอดขายเฉลี่ยต่อวัน", f"{s.mean():.1f} ชิ้น/วัน")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor="none")
        style_ax(ax2)
        ax2.plot(s.index, s.values, color="#9CA3AF", alpha=0.5, linewidth=1, label="Daily Sales")
        ax2.plot(roll.index, roll.values, color="#DC2626", linewidth=2.2, label="7-Day Moving Average (รวมวันที่ไม่มียอดขาย)")
        ax2.set_ylabel("Quantity (pcs)")
        ax2.legend()
        st.pyplot(fig2)

st.markdown("---")
st.caption("ข้อจำกัด: ชุดข้อมูลไม่มีสต็อกคงเหลือและต้นทุน ระบบนี้เป็นเครื่องมือสนับสนุนการตัดสินใจ "
           "ไม่สามารถตรวจสต็อกสินค้าที่พ่วงหรือคำนวณจุดสั่งซื้อซ้ำได้")