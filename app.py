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
# ตั้งค่าหน้าเว็บและธีม
# ======================================================================

# ======================================================================
# 1. ตั้งค่าหน้าเว็บและธีม (สไตล์ Schedule dashboard: เทาอ่อน / ขาว / ดำ / พาสเทล)
# ======================================================================
st.set_page_config(page_title="Demand Forecasting to Support Inventory Planning", page_icon="🛍️",
                   layout="wide", initial_sidebar_state="expanded")

# สีกราฟ: ม่วงพาสเทลเข้ม / ส้มพีช / เทา
ACCENT, FESTIVE, GREY = "#4A90E2", "#F4A38C", "#B8CDE0"

st.markdown("""
<style>
  #MainMenu, footer, .stAppDeployButton {display:none !important;}
  header[data-testid="stHeader"] {background:transparent;}
  .stApp {background:#D9DADC;}
  html, body, [class*="css"] {font-family:'Inter','Noto Sans Thai','Segoe UI',sans-serif;}

  /* แผงหลักสีขาวมุมโค้งเหมือนในภาพ */
  .block-container {max-width:1500px; padding:1.6rem 2rem 2rem; font-size:.98rem; background:#fff;
        border-radius:28px; margin-top:1.2rem; margin-bottom:1.2rem; box-shadow:0 20px 50px rgba(0,0,0,.08);}
  h1 {font-size:1.8rem !important; font-weight:700 !important; color:#111;}
  h2, h3 {font-weight:700 !important; color:#111;}
  [data-testid="stMetricValue"] {font-size:1.6rem; font-weight:700; color:#111;}
  [data-testid="stMetricLabel"] p {font-size:.9rem; color:#6B7280;}

  /* ----- Sidebar: ไล่สีมิ้นต์ -> เทาอ่อน ----- */
  section[data-testid="stSidebar"] {background:linear-gradient(160deg,#DDEBEC 0%,#EEF3F3 35%,#F5F5F6 70%);
        border-right:none;}
  section[data-testid="stSidebar"] [role="radiogroup"] {gap:.2rem;}
  section[data-testid="stSidebar"] [role="radiogroup"] label {padding:.6rem 1rem; border-radius:999px; width:100%; cursor:pointer;}
  section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
  section[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight:500; color:#374151; font-size:.95rem;}
  section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(255,255,255,.55);}
  section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#fff; box-shadow:0 2px 8px rgba(0,0,0,.05);}
  section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#111; font-weight:700;}
  .brand {display:flex; align-items:center; gap:.7rem; margin:.3rem 0 1.4rem;}
  .brand .logo {width:44px;height:44px;border-radius:14px;background:#2B2F36;color:#fff;
                display:flex;align-items:center;justify-content:center;font-size:1.4rem;}
  .brand b {font-size:1.2rem; color:#111;} .brand small {display:block; color:#6B7280; font-size:.72rem;}
  .st-key-upload_card {background:rgba(255,255,255,.6); border-radius:20px; padding:14px 14px 6px;}
  .st-key-upload_card [data-testid="stFileUploaderDropzone"] {background:#fff; border-radius:14px;}

  /* ----- Inputs / buttons: ปุ่มดำ + แท็บพิลล์ดำ ----- */
  [data-testid="stTextInput"] input {border-radius:999px; background:#fff; border:1px solid #E5E7EB; padding:.6rem 1.1rem;}
  div[data-baseweb="select"] > div {border-radius:999px !important; border-color:#E5E7EB !important;}
  .stButton > button, [data-testid="stBaseButton-secondary"] {border-radius:999px; font-weight:600; padding:.5rem 1.4rem;
        border:1px solid #E5E7EB; background:#fff; color:#111;}
  button[kind="primary"], [data-testid="stBaseButton-primary"] {background:#111 !important; color:#fff !important; border:none !important;}
  [data-testid="stMain"] [role="radiogroup"] label {background:#F3F4F6; padding:.15rem .9rem; border-radius:999px; border:none;}
  [data-testid="stMain"] [role="radiogroup"] label > div:first-child {display:none;}
  [data-testid="stMain"] [role="radiogroup"] label p {color:#6B7280; font-size:.88rem;}
  [data-testid="stMain"] [role="radiogroup"] label:has(input:checked) {background:#111;}
  [data-testid="stMain"] [role="radiogroup"] label:has(input:checked) p {color:#fff !important;}

  /* ----- Cards ----- */
  [data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:22px; border:1px solid #EEF0F2;
        box-shadow:0 4px 18px rgba(0,0,0,.04);}
  .st-key-hero {background:linear-gradient(135deg,#E4E3FA 0%,#EEF0FB 55%,#F7F7FB 100%); border-radius:26px; padding:30px 34px;
        box-shadow:none;}
  .hero-h {font-size:2.2rem; font-weight:800; line-height:1.2; color:#111; margin:0;}
  .hero-h span {color:#7B83EB;}
  .hero-p {font-size:1rem; color:#4B5563; margin:.8rem 0 1.1rem;}
  .hero-art {font-size:7.5rem; text-align:center; line-height:1; filter:drop-shadow(0 12px 18px rgba(0,0,0,.12));}
  .hero-art small {display:block; font-size:2rem; margin-top:.6rem; letter-spacing:.4rem;}
  .pillbadge {display:inline-block; background:#fff; border:1px solid #E5E7EB; border-radius:999px; padding:.4rem 1rem;
              font-size:.88rem; font-weight:600; margin-left:.5rem;}
  .tile {border-radius:20px; padding:16px 16px 14px; min-height:112px;}
  .tile .ic {width:44px;height:44px;border-radius:14px;background:rgba(255,255,255,.75);display:flex;align-items:center;
             justify-content:center; font-size:1.35rem; margin-bottom:.5rem;}
  .tile b {display:block; font-size:1rem; color:#111;} .tile span {font-size:.86rem; color:#4B5563;}
  .chips {display:flex; flex-wrap:wrap; gap:.7rem;}
  .chip {display:flex; align-items:center; gap:.6rem; background:#fff; border:1px solid #EEF0F2; border-radius:16px; padding:.55rem .9rem;
         box-shadow:0 2px 10px rgba(0,0,0,.04); max-width:260px;}
 .chip .d {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 38px;
    color: #333;
}

.chip .d svg {
    width: 34px;
    height: 34px;
    display: block;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
} .chip b {display:block; font-size:.88rem; color:#111; white-space:nowrap; overflow:hidden;
         text-overflow:ellipsis; max-width:190px;} .chip small {color:#6B7280; font-size:.78rem;}
  .donut {width:118px;height:118px;border-radius:50%;display:flex;align-items:center;justify-content:center;}
  .donut-in {width:88px;height:88px;border-radius:50%;background:#fff;display:flex;flex-direction:column;
             align-items:center;justify-content:center;}
  .donut-in b {font-size:1.45rem; color:#111;} .donut-in span {font-size:.72rem; color:#6B7280;}
  .mini {display:flex; gap:.5rem; align-items:flex-end; height:74px; margin-top:.4rem;}
  .mini div {flex:1; text-align:center; font-size:.72rem; color:#6B7280;} .mini i {display:block; border-radius:7px 7px 3px 3px; margin-bottom:3px;}
  .lb {display:flex; align-items:center; gap:.6rem; padding:.55rem .6rem; border-radius:14px; margin-bottom:.2rem;}
  .lb .nm {flex:1; font-size:.86rem; font-weight:600; color:#111; line-height:1.25;}
  .lb .lf {font-size:.8rem; font-weight:800; color:#7B83EB;}
  .medal {width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;
          font-size:.85rem; color:#fff; flex-shrink:0;}
  /* Ranking Medal - Blue Gradient */
.m1 {
    background: #4A90E2;
    color: #FFFFFF;
}

.m2 {
    background: #78AFE8;
    color: #FFFFFF;
}

.m3 {
    background: #A6C9F0;
    color: #FFFFFF;
}

.m4, .m5 {
    background: #FFFFFF;
    color: #6B7280;
    border: 1px solid #DCE5EF;
}
  .gift {background:linear-gradient(135deg,#E4E3FA,#F3F3FC); border-radius:22px; padding:18px 18px 16px; border:none;}
  .pbar {height:9px; border-radius:99px; background:#DCDDF3; margin-top:.5rem;} .pbar div {height:9px; border-radius:99px; background:#111;}
  .card-t {font-weight:700; font-size:1.02rem; color:#111; margin-bottom:.5rem;}
  .bundle-title {font-size:1.08rem; font-weight:700; color:#111; margin-bottom:.2rem;}
  .badge {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.82rem; font-weight:700;}
  .b-high {background:#D5F5E3; color:#166534;} .b-mid {background:#FDEBC8; color:#92400E;}
  .lead {font-size:1rem; color:#6B7280; margin:.1rem 0 1rem;}
  .note {font-size:.86rem; color:#6B7280;}

/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {
    background-color: #E4EFF0;
}

/* SIDEBAR BRAND */

.brand {
    display: flex;
    align-items: center;
    margin: .5rem 0 1.5rem;
}

.brand b {
    display: block;
    font-size: 1.2rem !important;
    font-weight: 750;
    color: #111;
}

.brand small {
    display: block;
    color: #6B7280;
    font-size: .78rem !important;
}

/* MENU HEADING */
.menu-heading {
    color: #8A969A;
    font-size: 15px;
    margin: 0 0 22px 12px;
}

/* MENU BUTTON */

/* ----- Sidebar Menu Buttons ----- */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 16px !important;
    color: #202020 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    min-height: 40px !important;
    height: 40px !important;
    padding: 6px 16px !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    margin-bottom: 6px !important;
}

/* Hover */
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.55) !important;
}

/* เมนูที่เลือก */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: none !important;
    border-radius: 16px !important;
    color: #111111 !important;
    font-weight: 650 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04) !important;
}

/* ขนาดไอคอน */
[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {
    font-size: 20px !important;
}


/* SELECTED MENU */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #FFFFFF;
    color: #111111;
    box-shadow: 0 5px 18px rgba(0,0,0,0.04);
}

/* ICON SIZE */
[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {
    font-size: 29px;
}
/* บังคับตัวหนังสือเมนู Sidebar ให้หนาและดำทุกเมนู */
section[data-testid="stSidebar"] div.stButton > button,
section[data-testid="stSidebar"] div.stButton > button p,
section[data-testid="stSidebar"] div.stButton > button span {
    font-weight: 600 !important;
    color: #000000 !important;
}
/* จัดเมนู Sidebar ชิดซ้าย */
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 16px !important;
}

/* จัดไอคอนและข้อความให้อยู่ชิดซ้าย */
[data-testid="stSidebar"] .stButton > button div {
    justify-content: flex-start !important;
}

/* จัดข้อความเมนูชิดซ้าย */
[data-testid="stSidebar"] .stButton > button p {
    text-align: left !important;
    width: 100% !important;
}

/* Home dashboard: compact highlight and reference-style spacing */
/* Home Highlight - Vertical Center */
.st-key-home_highlight {
    min-height: 150px;
    padding: 24px 30px;
    margin: 0 0 8px;
    border-radius: 26px;

    background: linear-gradient(
        120deg,
        #DCEEF5 0%,
        #EAF5F9 40%,
        #F5FAFC 75%,
        #FFFFFF 100%
    ) !important;

    border: none !important;
    box-shadow: none !important;

    display: flex;
    flex-direction: column;
    justify-content: center !important;
}

/* Title */
.st-key-home_highlight .home-highlight-title {
    font-size: 1.1rem;
    line-height: 1.5;
    font-weight: 700;
    color: #454957;
    margin: 0 0 12px;
}

/* Progress Bar */
.st-key-home_highlight .pbar {
    height: 9px;
    border-radius: 99px;
    background: #D5E8F0;
    margin-top: 8px;
}

.st-key-home_highlight .pbar div {
    height: 9px;
    border-radius: 99px;
    background: #111111;
}

/* Description */
.st-key-home_highlight .home-highlight-note {
    font-size: 0.8rem;
    color: #778995;
    margin-top: 10px;
}

/* Progress Bar */
.st-key-home_highlight .pbar {
    height: 9px;
    border-radius: 99px;
    background: #D5E8F0;
    margin-top: 10px;
}

.st-key-home_highlight .pbar div {
    height: 9px;
    border-radius: 99px;
    background: #111111;
}
.home-highlight-title {font-size:1.04rem; line-height:1.55; font-weight:700; color:#454957; margin-bottom:12px;}
.home-highlight-note {font-size:.88rem; color:#777D8B; margin-top:9px;}

/* Search Bar - Minimal Clean Style */
.st-key-q {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* กรอบช่อง Search */
.st-key-q div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid #DADADA !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    height: 40px !important;
    min-height: 40px !important;
    overflow: hidden !important;
}

/* ลบพื้นหลังชั้นใน */
.st-key-q div[data-baseweb="input"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    height: 100% !important;
}

/* ช่องพิมพ์ */
.st-key-q input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;

    height: 40px !important;
    min-height: 40px !important;
    line-height: 40px !important;

    font-size: 15px !important;
    color: #333333 !important;

    padding: 0 12px 0 40px !important;
    box-sizing: border-box !important;

    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23999999' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");

    background-repeat: no-repeat !important;
    background-position: 12px center !important;
    background-size: 22px 22px !important;
}

.st-key-q input::placeholder {
    color: #85858D !important;
    opacity: 1 !important;
}

/* กรอบ*/
.st-key-q div[data-baseweb="input"]:focus-within {
    border: 1px solid #DADADA !important;
    box-shadow: none !important;
}
.tile {min-height:112px;}
.chips {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.7rem;}
.chip {max-width:none; min-width:0; min-height:66px;}
.chip b {max-width:100%;}
.home-side-card {padding:2px 0;}
@media (max-width: 900px) {
    .chips {grid-template-columns:repeat(2,minmax(0,1fr));}
}

/* Forecast Card */
.st-key-nextcard {
    background: #FFFFFF !important;
    border: 1.5px solid #D9D9D9 !important;
    border-radius: 8px !important;
    padding: 18px 22px !important;
    box-shadow: none !important;
}

/* Forecast Title */
.forecast-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #343541;
    line-height: 1.6;
    margin-bottom: 8px;
}

/* Forecast Subtitle */
.forecast-subtitle {
    font-size: 1.05rem;
    font-weight: 600;
    color: #737987;
    margin-bottom: 22px;
}

/* Progress */
.forecast-progress {
    height: 10px !important;
    border-radius: 20px !important;
    background: #DEDEF3 !important;
    overflow: hidden;
}

.forecast-progress div {
    height: 10px !important;
    background: #111111 !important;
    border-radius: 20px !important;
}

        st.markdown("### Next Month Forecast")
        # Forecast code ...

        st.markdown("### Key Metrics")
        # KPI code ...

        st.markdown("### Top Products")
        # Top Products code ...

    with side:
        # Forecast Accuracy


/* Key Metrics */
.metric-card {
    height: 144px;
    width: 100%;
    box-sizing: border-box;
    padding: 17px 20px;
    border: none !important;
    border-radius: 15px;
    display: flex;
    flex-direction: column;
}

.metric-card.blue {
    background: linear-gradient(
        180deg, #D7E8F2 0%, #F3F8FD 100%
    );
}

.metric-card.green {
    background: linear-gradient(
        180deg, #D5F0D7 0%, #F2FBF1 100%
    );
}

/* Heading */
.metric-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #111111;
    font-size: 16px;
    font-weight: 700;
    white-space: nowrap;
    line-height: 26px;
}


/* ===== KPI Cards ===== */
.metric-card {
    width: 100%;
    height: 120px;
    min-width: 0;
    box-sizing: border-box;
    padding: 16px 14px;
    border: none !important;
    border-radius: 10px;
    box-shadow: none !important;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.metric-card.blue {
    background: linear-gradient(180deg, #D7E8F2 0%, #F3F8FD 100%);
}

.metric-card.green {
    background: linear-gradient(180deg, #D5F0D7 0%, #F2FBF1 100%);
}

/* Heading */
.metric-heading {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 100%;
    min-width: 0;
    color: #111111;
    font-size: 14px;
    line-height: 22px;
    font-weight: 700;
}

.metric-heading .metric-label {
    flex: 1;
    min-width: 0;
    white-space: normal;
    overflow-wrap: anywhere;
}

.metric-icon {
    width: 23px;
    height: 23px;
    flex: 0 0 23px;
}

.metric-icon svg {
    display: block;
    width: 23px;
    height: 23px;
}

/* Value */
.metric-value {
    width: 100%;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    line-height: 24px;
    font-weight: 500;
    color: #111111;
    overflow-wrap: anywhere;
}

 
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

NAV = {
    "Home": "home",
    "Sales": "sales",
    "Bundle": "bundle",
    "Forecast": "forecast",
    "Product": "product",
    "About": "about"
}

MENU_ICONS = {
    "Home": ":material/home:",
    "Sales": ":material/monitoring:",
    "Bundle": ":material/link:",
    "Forecast": ":material/trending_up:",
    "Product": ":material/inventory_2:",
    "About": ":material/info:"
}

# Sidebar Logo
st.sidebar.markdown("""
<div class="brand">
    <div>
        <b>Retail Insight</b>
        <small>Demand Forecasting & Bundling</small>
    </div>
</div>
""", unsafe_allow_html=True)

if "nav" not in st.session_state:
    st.session_state.nav = "Home"

for label, value in NAV.items():
    if st.sidebar.button(
        label,
        key=f"menu_{value}",
        icon=MENU_ICONS[label],
        use_container_width=True,
        type="primary" if st.session_state.nav == label else "secondary"
    ):
        st.session_state.nav = label
        st.rerun()

nav = st.session_state.nav

if nav not in NAV:
    nav = "Home"
    st.session_state.nav = "Home"

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
q = t1.text_input("ค้นหา", key="q", placeholder="Search", label_visibility="collapsed")
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

        # Compact seasonal insight banner, matching the reference layout
        tip = (f"เดือนที่ขายดีที่สุดคือ {thm(peak)} — ช่วงเทศกาล ควรเตรียมสินค้าล่วงหน้า"
               if festive_share > 22 else f"เดือนที่ขายดีที่สุดคือ {thm(peak)} — วางแผนสินค้าให้พร้อมก่อนช่วงยอดขายสูง")
        progress = min(max(float(festive_share), 0), 100)
        with st.container(key="home_highlight"):
            hi_left, hi_right = st.columns([3.2, 1.35], vertical_alignment="center")
            with hi_left:
                st.markdown(f'<div class="home-highlight-title">{html.escape(tip)}</div>'
                            f'<div class="pbar"><div style="width:{progress:.0f}%"></div></div>'
                            f'<div class="home-highlight-note">ยอดขายช่วงเทศกาล พ.ย.–ธ.ค. คิดเป็น {festive_share:.1f}% ของยอดขายทั้งหมด</div>',
                            unsafe_allow_html=True)
            with hi_right:
                if st.button("ดูสินค้าที่ควรขายคู่กัน", key="home_bundle_link", type="primary", use_container_width=True):
                    st.session_state.nav = "Bundle"
                    st.rerun()

        st.markdown("### Next Month Forecast")
        with st.container(key="nextcard", border=True):
            if fc is None:
                st.info("ข้อมูลน้อยกว่า 8 เดือน จึงยังคาดการณ์ไม่ได้")
            else:
                mq, _ = monthly_qty(df)
                nxt = fc["fut"].iloc[0]
                prog = float(min(max(nxt["Forecast"] / mq.max(), 0), 1))
                n2, n3 = st.columns([4, 1.2], vertical_alignment="center")
                
                n2.markdown(f"**{thm(nxt['Month'])}** — คาดว่าจะขายได้ประมาณ **{nxt['Forecast']:,.0f} ชิ้น**<br>"
                            f"<span class='note'>เทียบกับเดือนที่ขายสูงสุด ({prog:.0%})</span>"
                            f"<div class='pbar'><div style='width:{prog*100:.0f}%'></div></div>", unsafe_allow_html=True)
                if n3.button("ดูรายละเอียด", type="primary", key="go_fc"):
                    st.session_state.nav = "Forecast"
                    st.rerun()


        
        st.markdown("### Key Metrics")

        k = st.columns(4, gap="small")

        icons = [
            # Dollar
            """<svg viewBox="0 0 24 24" fill="none"
                stroke="#333333" stroke-width="1.8"
                stroke-linecap="round">
                <path d="M12 2v20 M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>""",

            # Box / Quantity
            """<svg viewBox="0 0 24 24" fill="none"
                stroke="#333333" stroke-width="1.8"
                stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18"/>
                <rect x="7" y="7" width="10" height="10"/>
            </svg>""",

            # Shopping Cart
            """<svg viewBox="0 0 24 24" fill="none"
                stroke="#333333" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 3h2l3 13h12l3-9H5"/>
                <circle cx="9" cy="20" r="1.5"/>
                <circle cx="18" cy="20" r="1.5"/>
                <path d="M13 4v7 M10 7.5h6"/>
            </svg>""",

            # Calendar Clock
            """<svg viewBox="0 0 24 24" fill="none"
                stroke="#333333" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="5" width="18" height="16" rx="1.5"/>
                <path d="M7 2v6 M17 2v6 M3 10h18"/>
                <circle cx="16.5" cy="16" r="3"/>
                <path d="M16.5 14.5v1.5l1.2.8"/>
            </svg>"""
        ]

        tiles = [
            ("blue", "Total Revenue", money_short(total_sales)),
            ("green", "Total Quantity Sold", f"{total_qty:,.0f} ชิ้น"),
            ("blue", "Total Products", f"{n_products:,} รายการ"),
            ("green", "Peak Month", thm(peak))
        ]

        for col, (icon, (color, title, value)) in zip(k, zip(icons, tiles)):
            col.markdown(f"""
                <div class="metric-card {color}">
                    <div class="metric-heading">
                        <span class="metric-icon">{icon}</span>
                        <span class="metric-label">{title}</span>
                    </div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)


        
        st.markdown("### Top Products")
        topc = df.groupby("Description")["Quantity"].sum().nlargest(5)

        # Package Icon (SVG)
        package_icon = """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m12 3 8.5 4.7v8.6L12 21l-8.5-4.7V7.7L12 3Z"/>
            <path d="m3.5 7.7 8.5 4.8 8.5-4.8"/>
            <path d="M12 12.5V21"/>
            <path d="m7.7 5.4 8.6 4.8"/>
        </svg>
        """

        chips = "".join(
            f'<div class="chip"><span class="d">{package_icon}</span><div style="min-width:0"><b title="{html.escape(str(name))}">{html.escape(str(name))}</b>'
            f'<small>ขายได้ {qty:,.0f} ชิ้น</small></div></div>'
            for name, qty in topc.items())

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
                d1.markdown(f'<div class="donut" style="background:conic-gradient({ACCENT} {pct}%, #DCEBFA 0)"><div class="donut-in"><b>{pct:.0f}%</b><span>Accuracy</span></div></div>', unsafe_allow_html=True)
                d2.markdown(f"**{q_txt}**<br><span class='note'>คลาดเคลื่อนจากของจริง {w:.1f}%<br>ความแม่นยำ = 100 − WAPE</span>", unsafe_allow_html=True)
            st.markdown('<div class="card-t" style="margin-top:1rem">Sales by Day of Week</div>', unsafe_allow_html=True)
            dv = dow.reindex(DOW_ORDER).fillna(0)
            mx = max(dv.max(), 1)
            bars = "".join(
                f'<div><i style="height:{8 + 52 * v / mx:.0f}px;background:{ACCENT if v == dv.max() else "#D6E8FA"}"></i>'
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