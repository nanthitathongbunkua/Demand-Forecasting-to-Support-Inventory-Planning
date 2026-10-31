import hashlib
import html
import io
import re
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ======================================================================
# 1. ตั้งค่าหน้าเว็บและธีม (Sidebar เมนู / Hero / การ์ดสีพาสเทล / แผงด้านขวา)
# ======================================================================
st.set_page_config(page_title="Demand Forecasting to Support Inventory Planning", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

ACCENT, FESTIVE, GREY = "#1F6FF5", "#F28C28", "#9AA5B5"

st.markdown("""
<style>
  #MainMenu, footer, .stAppDeployButton {display:none !important;}
  header[data-testid="stHeader"] {background:transparent;}
  .stApp {background:linear-gradient(180deg,#E6F1FF 0%,#F3F8FF 45%,#F7FAFF 100%);}
  html, body, [class*="css"] {font-family:'Inter','Noto Sans Thai','Segoe UI',sans-serif;}
  .block-container {max-width:1500px; padding:1.4rem 2rem 2rem; font-size:1.02rem;}
  h1 {font-size:1.9rem !important; font-weight:800 !important;}
  h2, h3 {font-weight:700 !important; color:#0F1B3D;}
  [data-testid="stMetricValue"] {font-size:1.7rem; font-weight:800;}
  [data-testid="stMetricLabel"] p {font-size:.95rem; color:#475467;}

  /* ----- Sidebar ----- */
  section[data-testid="stSidebar"] {background:#F8FBFF; border-right:1px solid #E3EDFA;}
  section[data-testid="stSidebar"] [role="radiogroup"] {gap:.25rem;}
  section[data-testid="stSidebar"] [role="radiogroup"] label {padding:.7rem 1rem; border-radius:14px; width:100%; cursor:pointer;}
  section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
  section[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight:600; color:#475467; font-size:1rem;}
  section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#DCEBFF;}
  section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#1F6FF5;}
  .brand {display:flex; align-items:center; gap:.7rem; margin:.3rem 0 1.2rem;}
  .brand .logo {width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#BFDDFF,#7CB6FF);
                display:flex;align-items:center;justify-content:center;font-size:1.6rem;}
  .brand b {font-size:1.35rem; color:#0F1B3D;} .brand small {display:block; color:#667085; font-size:.72rem;}
  .st-key-upload_card {background:#E4EFFF; border-radius:20px; padding:14px 14px 6px;}
  .st-key-upload_card [data-testid="stFileUploaderDropzone"] {background:#fff; border-radius:14px;}
  .store-chip {display:flex; align-items:center; gap:.7rem; background:#fff; border:1px solid #E3EDFA; border-radius:18px;
               padding:.7rem .9rem; margin-top:.8rem;}
  .store-chip .av {width:40px;height:40px;border-radius:50%;background:#DCEBFF;display:flex;align-items:center;justify-content:center;}
  .store-chip b {font-size:.92rem; color:#0F1B3D;} .store-chip small {display:block; color:#667085; font-size:.74rem;}

  /* ----- Inputs / buttons ----- */
  [data-testid="stTextInput"] input {border-radius:999px; background:#fff; border:1px solid #DCE8F8; padding:.65rem 1.1rem;}
  div[data-baseweb="select"] > div {border-radius:14px !important;}
  .stButton > button, [data-testid="stBaseButton-secondary"] {border-radius:999px; font-weight:700; padding:.55rem 1.4rem;}
  button[kind="primary"], [data-testid="stBaseButton-primary"] {background:#1F6FF5 !important; color:#fff !important; border:none !important;}
  [data-testid="stMain"] [role="radiogroup"] label {background:#fff; padding:.2rem .85rem; border-radius:999px; border:1px solid #DCE8F8;}
  [data-testid="stMain"] [role="radiogroup"] label > div:first-child {display:none;}
  [data-testid="stMain"] [role="radiogroup"] label:has(input:checked) {background:#1F6FF5; border-color:#1F6FF5;}
  [data-testid="stMain"] [role="radiogroup"] label:has(input:checked) p {color:#fff !important;}

  /* ----- Cards ----- */
  [data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:22px; border:1px solid #E3EDFA;
        box-shadow:0 6px 24px rgba(40,90,170,.06);}
  .st-key-hero {background:linear-gradient(135deg,#BFDDFF 0%,#E3F0FF 60%,#F1F8FF 100%); border-radius:26px; padding:30px 34px;
        box-shadow:0 8px 28px rgba(40,90,170,.10);}
  .hero-h {font-size:2.3rem; font-weight:800; line-height:1.2; color:#0B1736; margin:0;}
  .hero-h span {color:#1F6FF5;}
  .hero-p {font-size:1.05rem; color:#1F2A44; margin:.8rem 0 1.1rem;}
  .hero-art {font-size:8rem; text-align:center; line-height:1; filter:drop-shadow(0 12px 18px rgba(30,80,160,.25));}
  .hero-art small {display:block; font-size:2rem; margin-top:.6rem; letter-spacing:.4rem;}
  .pillbadge {display:inline-block; background:#fff; border:1px solid #DCE8F8; border-radius:999px; padding:.45rem 1rem;
              font-size:.92rem; font-weight:700; margin-left:.5rem;}
  .tile {border-radius:20px; padding:16px 16px 14px; min-height:112px;}
  .tile .ic {width:44px;height:44px;border-radius:14px;background:#fff;display:flex;align-items:center;justify-content:center;
             font-size:1.35rem; margin-bottom:.5rem;}
  .tile b {display:block; font-size:1.02rem; color:#0F1B3D;} .tile span {font-size:.88rem; color:#475467;}
  .chips {display:flex; flex-wrap:wrap; gap:.7rem;}
  .chip {display:flex; align-items:center; gap:.6rem; background:#fff; border:1px solid #E3EDFA; border-radius:16px; padding:.55rem .9rem;
         box-shadow:0 3px 12px rgba(40,90,170,.05); max-width:260px;}
  .chip .d {font-size:1.3rem;} .chip b {display:block; font-size:.88rem; color:#0F1B3D; white-space:nowrap; overflow:hidden;
         text-overflow:ellipsis; max-width:190px;} .chip small {color:#667085; font-size:.78rem;}
  .donut {width:118px;height:118px;border-radius:50%;display:flex;align-items:center;justify-content:center;}
  .donut-in {width:88px;height:88px;border-radius:50%;background:#fff;display:flex;flex-direction:column;
             align-items:center;justify-content:center;}
  .donut-in b {font-size:1.45rem; color:#0F1B3D;} .donut-in span {font-size:.72rem; color:#667085;}
  .mini {display:flex; gap:.5rem; align-items:flex-end; height:74px; margin-top:.4rem;}
  .mini div {flex:1; text-align:center; font-size:.72rem; color:#667085;} .mini i {display:block; border-radius:7px 7px 3px 3px; margin-bottom:3px;}
  .lb {display:flex; align-items:center; gap:.6rem; padding:.55rem .6rem; border-radius:14px; margin-bottom:.2rem;}
  .lb.me {background:#E8F0FB;} .lb .nm {flex:1; font-size:.86rem; font-weight:600; color:#0F1B3D; line-height:1.25;}
  .lb .lf {font-size:.8rem; font-weight:800; color:#1F6FF5;}
  .medal {width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;
          font-size:.85rem; color:#fff; flex-shrink:0;}
  .m1 {background:#F5B82E;} .m2 {background:#A9B4C4;} .m3 {background:#F08A4B;} .m4, .m5 {background:#fff; color:#475467; border:1px solid #DCE8F8;}
  .gift {background:linear-gradient(135deg,#DCEBFF,#EEF5FF); border-radius:22px; padding:18px 18px 16px; border:1px solid #D3E4FA;}
  .pbar {height:9px; border-radius:99px; background:#C9DDF8; margin-top:.5rem;} .pbar div {height:9px; border-radius:99px; background:#1F6FF5;}
  .card-t {font-weight:800; font-size:1.05rem; color:#0F1B3D; margin-bottom:.5rem;}
  .bundle-title {font-size:1.12rem; font-weight:700; color:#101828; margin-bottom:.2rem;}
  .badge {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.82rem; font-weight:700;}
  .b-high {background:#DCFCE7; color:#166534;} .b-mid {background:#FEF3C7; color:#92400E;}
  .lead {font-size:1.02rem; color:#475467; margin:.1rem 0 1rem;}
  .note {font-size:.88rem; color:#667085;}
</style>
""", unsafe_allow_html=True)

FESTIVE_MONTHS, REGULAR_MONTHS = (11, 12), tuple(range(1, 11))
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES"}
TH_MONTH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
TH_DAY = {"Monday": "จันทร์", "Tuesday": "อังคาร", "Wednesday": "พุธ", "Thursday": "พฤหัสฯ",
          "Friday": "ศุกร์", "Saturday": "เสาร์", "Sunday": "อาทิตย์"}


def thm(period):  # "2011-11" -> "พ.ย. 2011"
    y, m = str(period).split("-")[:2]
    return f"{TH_MONTH[int(m) - 1]} {y}"


# ======================================================================
# 2. โหลดและทำความสะอาดข้อมูล
# ======================================================================
def _norm(c):
    return re.sub(r"[^a-z0-9ก-๙]", "", str(c).lower())


SYN = {
    "Date": ["invoicedate", "date", "orderdate", "transactiondate", "saledate", "datetime", "วันที่"],
    "Invoice": ["invoice", "invoiceno", "invoicenumber", "orderid", "transactionid", "receipt", "billno", "basket"],
    "StockCode": ["stockcode", "sku", "productid", "itemcode"],
    "Description": ["description", "product", "productname", "item", "itemname", "สินค้า"],
    "Quantity": ["quantity", "qty", "units", "จำนวน"],
    "Price": ["price", "unitprice", "priceperunit"],
    "Sales_Value": ["salesvalue", "revenue", "amount", "totalprice", "sales", "linetotal", "total", "ยอดขาย"],
}


def find_cols(cols):
    normed = {_norm(c): c for c in cols}
    return {k: next((normed[s] for s in v if s in normed), None) for k, v in SYN.items()}


def prepare(raw):
    """แปลงไฟล์ของร้านใดก็ได้ให้เป็นรูปแบบมาตรฐาน Date / Description / Quantity / Sales_Value (+Invoice)"""
    m = find_cols(raw.columns)
    if m["Date"] is None or m["Quantity"] is None or (m["Description"] is None and m["StockCode"] is None):
        return None, None, ("ไม่พบคอลัมน์ที่จำเป็น: ต้องมีวันที่ (Date/InvoiceDate), ชื่อหรือรหัสสินค้า "
                            "(Description/StockCode) และจำนวน (Quantity)")
    if m["Sales_Value"] is None and m["Price"] is None:
        return None, None, "ต้องมีคอลัมน์ยอดขาย (Sales_Value/Revenue) หรือราคาต่อหน่วย (Price/UnitPrice)"

    d = pd.DataFrame({
        "Date": pd.to_datetime(raw[m["Date"]], errors="coerce"),
        "Quantity": pd.to_numeric(raw[m["Quantity"]], errors="coerce"),
    })
    d["Description"] = raw[m["Description"] or m["StockCode"]].astype(str).str.strip()
    d["Sales_Value"] = (pd.to_numeric(raw[m["Sales_Value"]], errors="coerce") if m["Sales_Value"]
                        else d["Quantity"] * pd.to_numeric(raw[m["Price"]], errors="coerce"))
    if m["Invoice"]:
        d["Invoice"] = raw[m["Invoice"]].astype(str).str.strip()
    stock = raw[m["StockCode"]].astype(str).str.strip().str.upper() if m["StockCode"] else None

    n0, removed = len(d), {}
    keep = d["Date"].notna() & d["Quantity"].notna() & d["Sales_Value"].notna() & (d["Description"] != "nan")
    removed["วันที่/จำนวน/ยอดขายว่างหรืออ่านไม่ได้"] = int((~keep).sum())
    if "Invoice" in d:
        canc = d["Invoice"].str.upper().str.startswith("C") & keep
        removed["รายการยกเลิก (Invoice ขึ้นต้นด้วย C)"] = int(canc.sum())
        keep &= ~canc
    neg = (d["Quantity"] <= 0) & keep
    removed["Quantity ≤ 0 (คืนสินค้า/ผิดปกติ)"] = int(neg.sum())
    keep &= ~neg
    zero = (d["Sales_Value"] <= 0) & keep
    removed["ยอดขายหรือราคา ≤ 0"] = int(zero.sum())
    keep &= ~zero
    if stock is not None:
        nonp = stock.isin(NON_PRODUCT) & keep
        removed["รายการที่ไม่ใช่สินค้า (POST, DOT, M, C2, BANK CHARGES)"] = int(nonp.sum())
        keep &= ~nonp
    d = d[keep].sort_values("Date").reset_index(drop=True)
    if d.empty:
        return None, None, "หลังทำความสะอาดแล้วไม่เหลือข้อมูลที่ใช้ได้"
    info = {"rows_raw": n0, "rows_clean": len(d), "removed": removed,
            "has_invoice": "Invoice" in d.columns, "mapped": {k: v for k, v in m.items() if v}}
    return d, info, None


@st.cache_data(show_spinner="กำลังอ่านและทำความสะอาดไฟล์...")
def load_upload(data: bytes, name: str):
    try:
        if name.lower().endswith(".csv"):
            frames = [pd.read_csv(io.BytesIO(data))]
        else:
            xl = pd.ExcelFile(io.BytesIO(data))
            frames = [xl.parse(s) for s in xl.sheet_names]
        frames = [f for f in frames if f.shape[1] >= 3 and len(f) > 0]
        if not frames:
            return None, None, "ไม่พบตารางข้อมูลในไฟล์"
        return prepare(pd.concat(frames, ignore_index=True))
    except Exception as e:
        return None, None, f"อ่านไฟล์ไม่ได้: {e}"


@st.cache_data
def load_project():
    d = pd.read_csv("app_data/daily_sales.csv.gz")
    d["Date"] = pd.to_datetime(d["Date"])
    return d


@st.cache_data
def load_csv(name):
    return pd.read_csv(f"app_data/{name}")


# ======================================================================
# 3. ฟังก์ชันวิเคราะห์ (ใช้กับไฟล์ร้านอื่น)
# ======================================================================
def run_rules(its, n_tx, conf):
    try:
        return association_rules(its, metric="confidence", min_threshold=conf)
    except TypeError:
        return association_rules(its, num_itemsets=n_tx, metric="confidence", min_threshold=conf)


@st.cache_data(show_spinner="กำลังหาคู่สินค้า (FP-Growth)...")
def mine_rules(_df, key, period, min_support, min_conf):
    sub = _df
    if period == "Festive":
        sub = _df[_df["Date"].dt.month.isin(FESTIVE_MONTHS)]
    elif period == "Regular":
        sub = _df[_df["Date"].dt.month.isin(REGULAR_MONTHS)]
    tx = sub.groupby("Invoice")["Description"].apply(lambda x: list(set(x))).tolist()
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    if len(tx) < 50:
        return pd.DataFrame(columns=cols), len(tx)
    te = TransactionEncoder()
    enc = pd.DataFrame.sparse.from_spmatrix(te.fit(tx).transform(tx, sparse=True), columns=te.columns_)
    its = fpgrowth(enc, min_support=min_support, use_colnames=True, max_len=3)
    if its.empty:
        return pd.DataFrame(columns=cols), len(tx)
    r = run_rules(its, len(tx), min_conf)
    r = r[r["lift"] > 1].copy()
    r["antecedents"] = r["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    r["consequents"] = r["consequents"].apply(lambda s: " + ".join(sorted(s)))
    return r[cols].sort_values("lift", ascending=False).head(2000).reset_index(drop=True), len(tx)


def monthly_qty(df):
    m = df.groupby(df["Date"].dt.to_period("M"))["Quantity"].sum().astype(float)
    partial = False
    last = df["Date"].max()
    if len(m) > 1 and last < last.to_period("M").end_time.normalize() - pd.Timedelta(days=3):
        m, partial = m.iloc[:-1], True
    m.index = m.index.astype(str)
    return m, partial


def hw_fit(series, seasonal):
    return ExponentialSmoothing(series.clip(lower=0.1), seasonal="add" if seasonal else None,
                                seasonal_periods=12 if seasonal else None,
                                initialization_method="legacy-heuristic").fit()


@st.cache_data(show_spinner="กำลังพยากรณ์...")
def forecast_upload(_df, key):
    m, partial = monthly_qty(_df)
    if len(m) < 8:
        return None
    test_n = min(6, max(3, len(m) // 4))
    tr = len(m) - test_n
    train, test = m.iloc[:tr], m.iloc[tr:]
    pred = np.clip(hw_fit(train, len(train) >= 12).forecast(test_n).values, 0, None)
    bt = pd.DataFrame({"Month": test.index, "Actual": test.values, "Forecast": pred})
    err = bt.Actual - bt.Forecast
    nz = bt.Actual > 0
    base = np.repeat(train.iloc[-3:].mean(), test_n)
    fm = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
          "mape": float((np.abs(err[nz]) / bt.Actual[nz]).mean() * 100),
          "wape": float(np.abs(err).sum() / bt.Actual.sum() * 100),
          "base_wape": float(np.abs(test.values - base).sum() / test.sum() * 100),
          "train": tr, "test": test_n}
    idx = pd.period_range(pd.Period(m.index[-1]) + 1, periods=3, freq="M").astype(str)
    fut = pd.DataFrame({"Month": idx, "Forecast": np.clip(hw_fit(m, len(m) >= 12).forecast(3).values, 0, None)})

    top = _df.groupby("Description")["Quantity"].sum().nlargest(10).index
    sub = _df[_df["Description"].isin(top)]
    pv = sub.groupby([sub["Date"].dt.to_period("M").astype(str), "Description"])["Quantity"].sum().unstack(fill_value=0)
    pv = pv.reindex(m.index, fill_value=0)
    rows = []
    for c in top:
        s = pv[c].astype(float)
        t, f = s.iloc[tr:].values, s.iloc[tr - 3:tr].mean()
        rows.append([c, s.iloc[-3], s.iloc[-2], s.iloc[-1], round(s.iloc[-3:].mean()),
                     round(np.abs(t - f).sum() / max(t.sum(), 1) * 100, 1)])
    sku = pd.DataFrame(rows, columns=["Description", "M-2", "M-1", "M0", "Forecast_next_month", "WAPE_%"])
    return bt, fut, fm, sku, partial

def money_short(v):
    return f"£{v/1e6:,.2f} ล้าน" if v >= 1e6 else f"£{v:,.0f}"


def area_line(df, x, y, order, y_title, tip, height=300):
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=True)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format="~s")), tooltip=tip)
    peak = df.loc[[df[y].idxmax()]].assign(txt="peak")
    pk = alt.Chart(peak).encode(x=alt.X(f"{x}:N", sort=order), y=f"{y}:Q")
    return (base.mark_area(opacity=.15, color=ACCENT) + base.mark_line(color=ACCENT, strokeWidth=3, point=True)
            + pk.mark_point(color=FESTIVE, filled=True, size=160)
            + pk.mark_text(dy=-16, fontWeight="bold", color=FESTIVE).encode(text="txt:N")
            ).properties(width="container", height=height)


def bars_h(df, cat, val, title, height):
    ch = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(f"{val}:Q", title=title, axis=alt.Axis(format="~s")), tooltip=[cat, alt.Tooltip(f"{val}:Q", format=",.0f")])
    return (ch.mark_bar(color=ACCENT, cornerRadiusEnd=4)
            + ch.mark_text(align="left", dx=4).encode(text=alt.Text(f"{val}:Q", format=",.0f"))
            ).properties(width="container", height=height)


def bars_v(df, cat, val, order, title, color_field=None, height=260):
    enc = dict(x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
               y=alt.Y(f"{val}:Q", title=title, axis=alt.Axis(format="~s")),
               tooltip=[cat, alt.Tooltip(f"{val}:Q", format=",.0f")])
    if color_field:
        enc["color"] = alt.Color(f"{color_field}:N", title=None, legend=alt.Legend(orient="bottom"),
                                 scale=alt.Scale(domain=["Festive Period", "Regular Period"], range=[FESTIVE, ACCENT]))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=ACCENT).encode(**enc).properties(
        width="container", height=height)


def forecast_chart(bt, fut):
    rows = []
    for _, r in bt.iterrows():
        rows += [[thm(r.Month), r.Actual, "Actual"], [thm(r.Month), r.Forecast, "Predicted Value"]]
    rows += [[thm(r.Month), r.Forecast, "Forecast"] for _, r in fut.iterrows()]
    d = pd.DataFrame(rows, columns=["month", "units", "items"])
    order = [thm(m) for m in list(bt["Month"]) + list(fut["Month"])]
    return alt.Chart(d).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("units:Q", title="units", axis=alt.Axis(format="~s")),
        color=alt.Color("items:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Actual", "Predicted Value", "Forecast"],
                                        range=[ACCENT, GREY, FESTIVE])),
        tooltip=["month", alt.Tooltip("units:Q", format=",.0f"), "items"]).properties(width="container", height=320)


def tidy_rules(r):
    """รวมคู่ที่ซ้ำกันคนละทิศทาง (A→B กับ B→A) ให้เหลือแถวเดียว"""
    r = r.copy()
    r["key"] = r.apply(lambda x: frozenset(x["antecedents"].split(" + ")) | frozenset(x["consequents"].split(" + ")), axis=1)
    return r.sort_values(["lift", "confidence"], ascending=False).drop_duplicates("key").drop(columns="key")


# ======================================================================
# 5. Sidebar: เมนู + อัปโหลดข้อมูลร้านอื่น + ข้อมูลที่ใช้อยู่
# ======================================================================
NAV = {"🏠  Home": "home", "📊  Sales": "sales", "🔗  Bundle": "bundle",
       "📈  Forecast": "forecast", "📦  Product": "product", "ℹ️  About": "about"}
st.sidebar.markdown('<div class="brand"><div class="logo">🛍️</div><div><b>Retail Insight</b>'
                    '<small>Demand Forecasting & Bundling</small></div></div>', unsafe_allow_html=True)
nav = st.sidebar.radio("เมนู", list(NAV), key="nav", label_visibility="collapsed")
page = NAV[nav]
st.sidebar.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

store, store_sub, using_upload, info = "Online Retail II", " ", False, None
with st.sidebar.container(key="upload_card"):
    st.markdown("**Upload file**")
    st.caption("(Excel/CSV)")
    uploaded = st.file_uploader("อัปโหลด", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
if uploaded is not None:
    data = uploaded.getvalue()
    df, info, err = load_upload(data, uploaded.name)
    if err:
        st.sidebar.error(err)
    else:
        using_upload, store, store_sub = True, uploaded.name, "ไฟล์ที่อัปโหลด"
        data_key = hashlib.md5(data).hexdigest()
if not using_upload:
    try:
        df = load_project()
    except FileNotFoundError:
        st.error("ไม่พบโฟลเดอร์ app_data/ — รัน `python export_data.py` ก่อน แล้วเปิดแอปใหม่")
        st.stop()
    data_key = "project"
st.sidebar.markdown(f'<div class="store-chip"><div class="av">🏬</div><div><b>{html.escape(store[:26])}</b>'
                    f'<small>{store_sub}</small></div></div>', unsafe_allow_html=True)

# ---------- ตัวเลขหลัก ----------
total_sales, total_qty = df["Sales_Value"].sum(), df["Quantity"].sum()
n_products = df["Description"].nunique()
d_min, d_max = df["Date"].min(), df["Date"].max()
monthly = df.groupby(df["Date"].dt.to_period("M"))["Sales_Value"].sum()
monthly.index = monthly.index.astype(str)
peak = monthly.idxmax()
dow = df.groupby(df["Date"].dt.day_name())["Sales_Value"].sum()
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
festive_share = df.loc[df["Date"].dt.month.isin(FESTIVE_MONTHS), "Sales_Value"].sum() / total_sales * 100


def get_forecast():
    if using_upload:
        res = forecast_upload(df, data_key)
        if res is None:
            return None
        bt, fut, fm, sku, partial = res
        note = f"ทดสอบกับ {fm['test']} เดือนสุดท้ายของไฟล์" + (" (ตัดเดือนสุดท้ายที่ไม่ครบเดือนออก)" if partial else "")
        return dict(bt=bt, fut=fut, fm=fm, sku=sku, note=note)
    meta = load_csv("meta.csv").iloc[0]
    fm = {"mae": meta["mae_total"], "rmse": meta["rmse_total"], "mape": meta["mape_total"],
          "wape": meta["wape_total"], "base_wape": None}
    sku = load_csv("forecast_top10_sku.csv").rename(columns={"WAPE_backtest_%": "WAPE_%"})
    return dict(bt=load_csv("forecast_backtest.csv"), fut=load_csv("forecast_future.csv"), fm=fm, sku=sku,
                note="ทดสอบกับ 6 เดือนสุดท้าย (ก.ค.–พ.ย. 2011)")


fc = get_forecast()

# ---------- แถบบน: ค้นหา + ตัวเลขสรุป ----------
t1, t2 = st.columns([2.2, 1.6])
q = t1.text_input("ค้นหา", key="q", placeholder="🔍 Search", label_visibility="collapsed")
t2.markdown(f'<div style="text-align:right"><span class="pillbadge"> {total_qty:,.0f} ชิ้น</span>'
            f'<span class="pillbadge"> {money_short(total_sales)}</span></div>', unsafe_allow_html=True)
st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

# ======================================================================
# หน้าแรก
# ======================================================================
if page == "home":
    main, side = st.columns([3.1, 1.25], gap="large")
    with main:
        if q:
            hit = df[df["Description"].str.contains(q, case=False, regex=False)]
            with st.container(border=True):
                st.markdown(f'<div class="card-t">ผลการค้นหา “{html.escape(q)}”</div>', unsafe_allow_html=True)
                if hit.empty:
                    st.info("ไม่พบสินค้าที่ตรงกับคำค้นหา")
                else:
                    rs = (hit.groupby("Description").agg(ขายได้=("Quantity", "sum"), ยอดขาย=("Sales_Value", "sum"))
                          .nlargest(8, "ขายได้").reset_index().rename(columns={"Description": "สินค้า"}))
                    rs["ยอดขาย"] = rs["ยอดขาย"].round(0)
                    st.dataframe(rs, hide_index=True)
                    st.caption(f"พบสินค้า {hit['Description'].nunique():,} รายการ · หน้า “สินค้าขายคู่กัน” จะกรองตามคำค้นหานี้ด้วย")

        with st.container(key="hero"):
            h1, h2 = st.columns([3, 2])
            with h1:
                tip = (f"เดือนที่ขายดีที่สุดคือ <b>{thm(peak)}</b> — ช่วงเทศกาล ควรเตรียมสินค้าล่วงหน้า"
                       if festive_share > 22 else f"เดือนที่ขายดีที่สุดคือ <b>{thm(peak)}</b>")
                st.markdown(f'<div class="hero-h">สวัสดี! นี่คือ<br><span>ภาพรวมร้านของคุณ</span> 👋</div>'
                            f'<div class="hero-p">{tip}</div>',
                            unsafe_allow_html=True)
                st.button("ดูสินค้าที่ควรขายคู่กัน  →", type="primary", key="cta_bundle",
                          on_click=lambda: st.session_state.update(nav="🔗  Bundle"))
            h2.markdown('<div class="hero-art">🛍️<small>📦 🏷️ ✨</small></div>', unsafe_allow_html=True)

        st.markdown("### Next Month Forecast")
        with st.container(key="nextcard", border=True):
            if fc is None:
                st.info("ข้อมูลน้อยกว่า 8 เดือน จึงยังคาดการณ์ไม่ได้")
            else:
                mq, _ = monthly_qty(df)
                nxt = fc["fut"].iloc[0]
                prog = float(min(max(nxt["Forecast"] / mq.max(), 0), 1))
                n1, n2, n3 = st.columns([0.7, 3, 1.2])
                n1.markdown('<div class="tile" style="background:#E4EFFF;text-align:center;font-size:2.2rem;'
                            'min-height:96px;padding-top:22px">📈</div>', unsafe_allow_html=True)
                n2.markdown(f"**{thm(nxt['Month'])}** — คาดว่าจะขายได้ประมาณ **{nxt['Forecast']:,.0f} ชิ้น**<br>"
                            f"<span class='note'>เทียบกับเดือนที่ขายสูงสุด ({prog:.0%})</span>"
                            f"<div class='pbar'><div style='width:{prog*100:.0f}%'></div></div>", unsafe_allow_html=True)
                n3.button("ดูรายละเอียด", type="primary", key="go_fc",
                          on_click=lambda: st.session_state.update(nav="📈  Forecast"))

        st.markdown("### Key Metrics")
        k = st.columns(4)
        tiles = [("#EAE5FB", "💷", "Total Revenue", money_short(total_sales)), ("#FDE6E6", "📦", "Total Quantity Sold", f"{total_qty:,.0f} ชิ้น"),
                 ("#DDF3E4", "🏷️", "Total Products", f"{n_products:,} รายการ"), ("#FDEBD3", "🏆", "Peak Month", thm(peak))]
        for col, (bg, ic, t, v) in zip(k, tiles):
            col.markdown(f'<div class="tile" style="background:{bg}"><div class="ic">{ic}</div><b>{t}</b><span>{v}</span></div>',
                         unsafe_allow_html=True)

        st.markdown("### Top Products")
        topc = df.groupby("Description")["Quantity"].sum().nlargest(5)
        chips = "".join(f'<div class="chip"><span class="d">🏷️</span><div><b title="{html.escape(n)}">{html.escape(n)}</b>'
                        f'<small>ขายได้ {v:,.0f} ชิ้น</small></div></div>' for n, v in topc.items())
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    with side:
        with st.container(border=True):
            st.markdown('<div class="card-t">Forecast Accuracy</div>', unsafe_allow_html=True)
            if fc is None:
                st.caption("ยังไม่มีข้อมูลพอสำหรับคาดการณ์")
            else:
                w = fc["fm"]["wape"]
                pct = max(0, min(100, 100 - w))
                q_txt = "High Accuracy" if w < 10 else ("Moderate" if w < 20 else "Low Accuracy")
                d1, d2 = st.columns([1.1, 1])
                d1.markdown(f'<div class="donut" style="background:conic-gradient({ACCENT} {pct}%, #DCE8F8 0)">'
                            f'<div class="donut-in"><b>{pct:.0f}%</b><span>High Accuracy</span></div></div>', unsafe_allow_html=True)
                d2.markdown(f"**{q_txt}**<br><span class='note'>คลาดเคลื่อนจากของจริง {w:.1f}% (ใกล้เคียง = 100 − WAPE)</span>", unsafe_allow_html=True)
            st.markdown('<div class="card-t" style="margin-top:1rem">Sales by Day of Week</div>', unsafe_allow_html=True)
            dv = dow.reindex(DOW_ORDER).fillna(0)
            mx = max(dv.max(), 1)
            bars = "".join(f'<div><i style="height:{8 + 52 * v / mx:.0f}px;background:{ACCENT if v == dv.max() else "#CFE0F7"}"></i>'
                           f'{TH_DAY[d][:2] if d != "Thursday" else "พฤ"}</div>' for d, v in dv.items())
            st.markdown(f'<div class="mini">{bars}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-t">Top Product Pairs</div>', unsafe_allow_html=True)
            if using_upload:
                st.caption("ดูผลของไฟล์นี้ได้ที่เมนู “Bundle”")
            else:
                pr = tidy_rules(load_csv("rules.csv").query("period == 'Festive'")).head(5)
                for i, (_, r) in enumerate(pr.iterrows(), 1):
                    nm = f"{r['antecedents']} + {r['consequents']}"
                    st.markdown(f'<div class="lb {"me" if i == 2 else ""}"><span class="medal m{i}">{i}</span>'
                                f'<div class="nm" title="{html.escape(nm)}">{html.escape(nm[:46])}{"…" if len(nm) > 46 else ""}</div>'
                                f'<span class="lf">×{r["lift"]:.0f}</span></div>', unsafe_allow_html=True)
                st.caption("ช่วงเทศกาล พ.ย.–ธ.ค. · ×N = ซื้อคู่กันบ่อยกว่าปกติกี่เท่า")

        st.markdown(f'<div class="gift"><div style="font-size:1.8rem">🎁</div><div class="card-t" style="margin:.3rem 0 0">Festive Season Impact</div>'
                    f'<div class="note">พ.ย.–ธ.ค. ยอดขายประมาณ <b>{festive_share:.0f}%</b> ของยอดทั้งหมด</div>'
                    f'<div class="pbar"><div style="width:{min(festive_share, 100):.0f}%"></div></div>'
                    f'<div class="note" style="margin-top:.3rem">เทียบสัดส่วนเวลา 2 เดือน เท่ากับ 17%</div></div>', unsafe_allow_html=True)

# ======================================================================
# ยอดขาย
# ======================================================================
elif page == "sales":
    st.markdown("## Total Sales")
    st.info(f"**Summary**\n\n- ขายได้รวม **{money_short(total_sales)}** หรือ **{total_qty:,.0f} ชิ้น**\n"
            f"- เดือนที่ขายดีที่สุดคือ **{thm(peak)}** \n"
            f"- วันที่ขายดีที่สุดของสัปดาห์คือ **วัน{TH_DAY[dow.idxmax()]}**")
    with st.container(border=True):
        st.markdown("### Monthly Sales")
        rng = st.radio("ช่วงเวลา", ["ทั้งหมด", "12 เดือนล่าสุด", "6 เดือนล่าสุด"], horizontal=True, label_visibility="collapsed")
        mm = monthly if rng == "ทั้งหมด" else monthly.iloc[-(12 if rng.startswith("12") else 6):]
        md = pd.DataFrame({"เดือน": [thm(i) for i in mm.index], "ยอดขาย (£)": mm.values})
        st.altair_chart(area_line(md, "เดือน", "ยอดขาย (£)", list(md["เดือน"]), "ยอดขาย (£)",
                                  ["เดือน", alt.Tooltip("ยอดขาย (£):Q", format=",.0f")]))
        st.caption("เดือนสุดท้ายและเดือนธันวาคมอาจมีข้อมูลไม่ครบเดือน จึงดูต่ำกว่าความเป็นจริง")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1, st.container(border=True):
        topn = st.selectbox("Top Selling Products", [5, 10, 15], index=1)
        st.markdown(f"### Top {topn} Best Sellers ")
        top = df.groupby("Description")["Quantity"].sum().nlargest(topn).reset_index()
        top.columns = ["สินค้า", "จำนวนที่ขาย (ชิ้น)"]
        st.altair_chart(bars_h(top, "สินค้า", "จำนวนที่ขาย (ชิ้น)", "จำนวนที่ขาย (ชิ้น)", 34 * topn + 30))
    with c2, st.container(border=True):
        st.markdown("### Peak Day of Week")
        dd = dow.reindex(DOW_ORDER).fillna(0).reset_index()
        dd.columns = ["วัน", "ยอดขาย (£)"]
        dd["วัน"] = dd["วัน"].map(TH_DAY)
        st.altair_chart(bars_v(dd, "วัน", "ยอดขาย (£)", [TH_DAY[o] for o in DOW_ORDER], "ยอดขาย (£)", height=330))

# ======================================================================
# สินค้าขายคู่กัน
# ======================================================================
elif page == "bundle":
    st.markdown("## Market Basket Analysis")
    st.markdown('<div class="lead">ใช้เลือกสินค้าไปจัดโปรโมชั่น </div>', unsafe_allow_html=True)
    if using_upload and not info["has_invoice"]:
        st.warning("ไฟล์ที่อัปโหลดไม่มีเลขที่บิล จึงบอกไม่ได้ว่าสินค้าใดถูกซื้อพร้อมกัน (เมนูอื่นยังใช้งานได้ตามปกติ)")
        st.stop()
    f1, f2 = st.columns([1, 2])
    if using_upload:
        period = f1.selectbox("ช่วงเวลา", ["All", "Festive", "Regular"],
                              format_func=lambda x: {"All": "ทั้งหมด", "Festive": "Festive Period", "Regular": "Regular Period"}[x])
        with st.expander("⚙️ ตั้งค่าขั้นสูง"):
            s1, s2 = st.columns(2)
            min_sup = s1.slider("ต้องพบในบิลอย่างน้อย (สัดส่วน)", 0.005, 0.05, 0.02, 0.005)
            min_conf = s2.slider("ซื้อ A แล้วซื้อ B ด้วยอย่างน้อย", 0.1, 0.8, 0.2, 0.05)
        rules, n_tx = mine_rules(df, data_key, period, min_sup, min_conf)
        how = f"วิเคราะห์จาก {n_tx:,} บิล"
    else:
        period = f1.selectbox("ช่วงเวลา", ["Festive", "Regular"],
                              format_func=lambda x: "Festive Period" if x == "Festive" else "Regular Period")
        rules = load_csv("rules.csv")
        rules = rules[rules["period"] == period].drop(columns="period").reset_index(drop=True)
        how = "วิเคราะห์จากบิลทั้งหมด"
    if q:
        f2.info(f"กำลังกรองด้วยคำค้นหา “{q}” (แก้ที่ช่องค้นหาด้านบน)")
    if rules.empty:
        st.info("ไม่พบสินค้าที่ซื้อคู่กันบ่อยพอในช่วงนี้ ลองเลือกช่วงเวลา 'ทั้งหมด' หรือลดเกณฑ์ในตั้งค่าขั้นสูง")
        st.stop()
    pairs = tidy_rules(rules)
    if q:
        k = q.upper()
        pairs = pairs[pairs["antecedents"].str.upper().str.contains(k, regex=False)
                      | pairs["consequents"].str.upper().str.contains(k, regex=False)]
    st.caption(f"{how} · พบ {len(pairs):,} คู่สินค้า")
    for _, r in pairs.head(5).iterrows():
        with st.container(border=True):
            a, b, c = st.columns([4, 1.3, 1.3])
            level = ("<span class='badge b-high'>Recommended</span>" if r["lift"] >= 15 else "<span class='badge b-mid'>น่าสนใจ</span>")
            a.markdown(f"<div class='bundle-title'>{html.escape(r['antecedents'])}  +  {html.escape(r['consequents'])}</div>{level}",
                       unsafe_allow_html=True)
            b.metric("ซื้อคู่กันบ่อยกว่าปกติ", f"{r['lift']:.0f} เท่า")
            c.metric("ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย", f"{r['confidence']:.0%}")
    if len(pairs) > 5:
        st.markdown("#### รายการทั้งหมด")
        tbl = pairs.assign(**{"พบในกี่ % ของบิลทั้งหมด": (pairs["support"] * 100).round(1)}).rename(columns={
            "antecedents": "สินค้าที่ซื้อ", "consequents": "มักซื้อคู่กับ",
            "confidence": "ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย (%)", "lift": "ซื้อคู่กันบ่อยกว่าปกติ (เท่า)"})
        tbl["ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย (%)"] = (tbl["ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย (%)"] * 100).round(0)
        tbl["ซื้อคู่กันบ่อยกว่าปกติ (เท่า)"] = tbl["ซื้อคู่กันบ่อยกว่าปกติ (เท่า)"].round(1)
        st.dataframe(tbl[["สินค้าที่ซื้อ", "มักซื้อคู่กับ", "พบในกี่ % ของบิลทั้งหมด", "ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย (%)",
                          "ซื้อคู่กันบ่อยกว่าปกติ (เท่า)"]], hide_index=True, height=380)
    st.info("คู่ที่เด่นส่วนใหญ่เป็น **สินค้าชุดเดียวกันต่างสีหรือต่างลาย** จึงเหมาะกับการขายเป็นเซ็ต "
            "(ข้อมูลนี้ไม่มีจำนวนสต็อก ก่อนจัดโปรโมชั่นควรตรวจสต็อกสินค้าที่จะพ่วงด้วยตนเอง)")
    with st.expander("ค่าทางเทคนิค (สำหรับผู้วิเคราะห์)"):
        st.markdown("Support = สัดส่วนบิลที่มีทั้งสองสินค้า · Confidence = ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย · "
                    "Lift = ซื้อคู่กันบ่อยกว่าที่ควรเป็นโดยบังเอิญกี่เท่า")
        st.dataframe(rules.round(3), hide_index=True, height=300)

# ======================================================================
# คาดการณ์ยอดขาย
# ======================================================================
elif page == "forecast":
    st.markdown("## 3-Month Sales Forecast")
    if fc is None:
        st.warning("ข้อมูลรายเดือนน้อยกว่า 8 เดือน จึงคาดการณ์ไม่ได้ (แนะนำข้อมูลอย่างน้อย 12 เดือน)")
        st.stop()
    bt, fut, fm = fc["bt"], fc["fut"], fc["fm"]
    cols = st.columns(3)
    for col, (_, r) in zip(cols, fut.iterrows()):
        with col, st.container(border=True):
            st.metric(thm(r["Month"]), f"{r['Forecast']:,.0f} ชิ้น")
    wape = fm["wape"]
    quality = "แม่นยำดี" if wape < 10 else ("พอใช้" if wape < 20 else "คลาดเคลื่อนค่อนข้างสูง")
    st.success(f"**ความแม่นยำ: {quality}** — "
               f"ค่าที่ทายคลาดจากยอดจริงโดยรวมประมาณ **{wape:.1f}%**")
    with st.container(border=True):
        st.altair_chart(forecast_chart(bt, fut))
    st.caption("เป็นการคาดการณ์ยอดรวมทุกสินค้ารายเดือน ใช้เป็นแนวทางภาพรวมเพื่อวางแผนสั่งซื้อ "
               "ข้อมูลมีไม่ถึง 2 ปีเต็ม ฤดูกาลจึงเรียนรู้ได้ไม่สมบูรณ์")
    with st.container(border=True):
        st.markdown("### นำผลคาดการณ์ไปใช้วางแผนสินค้าอย่างไร")
        st.markdown(
            "1. **เทียบกับสต็อกที่มีจริง** — นำตัวเลขคาดการณ์ 3 เดือนข้างหน้าไปเทียบกับจำนวนสินค้าคงเหลือของร้าน \n"
            f"2. **เตรียมสินค้าก่อนช่วงยอดสูง** — ข้อมูลนี้ยอดขายพุ่งช่วง ก.ย.–พ.ย. (สูงสุดเดือน {thm(peak)}) "
            "ควรเริ่มเตรียมสินค้าก่อนช่วงดังกล่าว\n"
            "3. **ใช้เป็นแนวทาง** — ความคลาดเคลื่อนระดับสินค้ารายตัวสูงกว่ายอดรวมมาก "
            "ควรตรวจยอดขายล่าสุดก่อนสั่งซื้อ\n"
            "4. **จัดโปรโมชั่นคู่สินค้า** — เลือกจากเมนู “Bundle” แล้วตรวจสต็อกสินค้าที่จะพ่วงก่อนเสมอ")
    with st.expander("ค่าทางสถิติ (สำหรับผู้วิเคราะห์)"):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("MAE", f"{fm['mae']:,.0f} ชิ้น")
        s2.metric("RMSE", f"{fm['rmse']:,.0f} ชิ้น")
        s3.metric("MAPE", f"{fm['mape']:.1f}%")
        s4.metric("WAPE", f"{fm['wape']:.1f}%")
        if fm.get("base_wape") is not None:
            st.caption(f"เทียบกับวิธีง่ายๆ (ค่าเฉลี่ย 3 เดือนล่าสุด) คลาดเคลื่อน {fm['base_wape']:.1f}%")

# ======================================================================
# สินค้ารายตัว
# ======================================================================
elif page == "product":
    st.markdown("## Top 10 Best Sellers - Forecasted Demand")
    if fc is None:
        st.warning("ข้อมูลรายเดือนน้อยกว่า 8 เดือน จึงคาดการณ์ไม่ได้")
    else:
        sv = fc["sku"][["Description", "M-2", "M-1", "M0", "Forecast_next_month", "WAPE_%"]].copy()
        for c in ["M-2", "M-1", "M0", "Forecast_next_month"]:
            sv[c] = sv[c].round(0).astype(int)
        sv = sv.rename(columns={"Description": "สินค้า", "M-2": "3 เดือนก่อน (ชิ้น)", "M-1": "2 เดือนก่อน (ชิ้น)",
                                "M0": "เดือนล่าสุด (ชิ้น)", "Forecast_next_month": "คาดว่าเดือนหน้า (ชิ้น)",
                                "WAPE_%": "คลาดเคลื่อนช่วงทดสอบ (%)"})
        st.dataframe(sv, hide_index=True)
        st.warning("ยอดขายของสินค้าแต่ละรายการมีความคลาดเคลื่อน ตัวเลขนี้เป็นเพียงแนวทาง "
                   "ไม่ควรใช้เป็นจำนวนสั่งซื้อจริงโดยไม่ตรวจสต็อกและยอดล่าสุดก่อน")
    st.markdown("### Product Trends")
    opts = sorted(df["Description"].unique().tolist())
    idx = next((i for i, o in enumerate(opts) if q and q.upper() in o.upper()), 0)
    sel = st.selectbox("เลือกสินค้า", opts, index=idx, label_visibility="collapsed")
    wk = df[df["Description"] == sel].groupby("Date")["Quantity"].sum().resample("W").sum()
    wd = pd.DataFrame({"สัปดาห์": wk.index, "ขายรายสัปดาห์": wk.values,
                       "เฉลี่ย 4 สัปดาห์": wk.rolling(4, min_periods=1).mean().values})
    long = wd.melt("สัปดาห์", var_name="ชนิด", value_name="ชิ้น")
    p1, p2, p3 = st.columns(3)
    p1.metric("ขายได้ทั้งหมด", f"{wk.sum():,.0f} ชิ้น")
    p2.metric("เฉลี่ยต่อสัปดาห์", f"{wk.mean():,.1f} ชิ้น")
    p3.metric("สัปดาห์ที่ขายได้", f"{(wk > 0).sum():,} จาก {len(wk):,}")
    with st.container(border=True):
        st.altair_chart(alt.Chart(long).mark_line(strokeWidth=2.5).encode(
            x=alt.X("สัปดาห์:T", title=None, axis=alt.Axis(format="%b %y")), y=alt.Y("ชิ้น:Q", title="จำนวนชิ้น"),
            color=alt.Color("ชนิด:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["ขายรายสัปดาห์", "เฉลี่ย 4 สัปดาห์"], range=[GREY, FESTIVE])),
            tooltip=[alt.Tooltip("สัปดาห์:T", format="%d/%m/%Y"), "ชนิด", alt.Tooltip("ชิ้น:Q", format=",.0f")]
        ).properties(width="container", height=280))
# ======================================================================
# เกี่ยวกับข้อมูล
# ======================================================================
else:
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown("### ข้อมูลที่ใช้")
        if using_upload:
            st.markdown(f"ไฟล์ **{uploaded.name}** · ใช้ {info['rows_clean']:,} จาก {info['rows_raw']:,} แถว")
            st.markdown("**แถวที่ตัดออกและเหตุผล**")
            st.table(pd.DataFrame({"เหตุผล": list(info["removed"].keys()), "จำนวนแถว": list(info["removed"].values())}))
            st.markdown("**ระบบจับคู่คอลัมน์ได้ดังนี้:** " + ", ".join(f"{k} ← `{v}`" for k, v in info["mapped"].items()))
        else:
            st.markdown("**Online Retail II** จาก UCI Machine Learning Repository (Chen, 2019) ข้อมูลการขายของร้านค้าปลีกออนไลน์ "
                        "ในสหราชอาณาจักร ตั้งแต่ 1 ธ.ค. 2009 ถึง 9 ธ.ค. 2011 ")
            st.markdown("- ลบข้อมูลซ้ำและรายการยกเลิกบิล\n- ตัดรายการที่จำนวนหรือราคาติดลบ/เป็นศูนย์\n"
                        "- แยกรายการที่ไม่ใช่สินค้า\n"
                        "- ไม่ตัดรายการที่ซื้อจำนวนมาก เพราะอาจเป็นการขายส่งจริง")
        st.markdown("### ข้อจำกัดที่ควรรู้")
        st.markdown("- **ไม่มีข้อมูลสต็อกคงเหลือและต้นทุน** จึงบอกไม่ได้ว่าสินค้าไหนขาดหรือค้างอยู่ในคลัง \n"
                    "- เดือนธันวาคมของข้อมูลโครงการไม่ครบเดือน\n"
                    "- ข้อมูลไม่ถึง 2 ปีเต็ม การคาดการณ์ตามฤดูกาลจึงเป็นเพียงแนวทาง")
    with b, st.container(border=True):
        st.markdown("### ระบบคิดอย่างไร")
        st.markdown(
            "**1. หาสินค้าที่ซื้อคู่กัน** — ดูว่าในบิลเดียวกันลูกค้าซื้ออะไรด้วยกันบ่อย เทียบกับที่ควรเป็นโดยบังเอิญ \n\n"
            "**2. คาดการณ์ยอดขาย** — ดูแนวโน้มและฤดูกาลจากยอดขายรายเดือนย้อนหลัง (Holt-Winters) "
            "แล้วทดสอบว่าถ้าให้ทายย้อนหลัง จะคลาดจากของจริงเท่าไร\n\n"
            )
        st.markdown("### ศัพท์ที่ใช้ในระบบ")
        st.markdown("- **ซื้อคู่กันบ่อยกว่าปกติ (เท่า)** — ค่า Lift ยิ่งสูงยิ่งไม่ใช่เรื่องบังเอิญ (มากกว่า 1 ถือว่ามีความสัมพันธ์)\n"
                    "- **ซื้อตัวแรกแล้วซื้อตัวที่สองด้วย (%)** — ค่า Confidence\n"
                    "- **คลาดเคลื่อน (WAPE)** — ผลรวมความผิดพลาดของค่าที่ทาย หารด้วยยอดขายจริงรวม ยิ่งต่ำยิ่งดี")