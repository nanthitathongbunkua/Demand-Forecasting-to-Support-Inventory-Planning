import os
import warnings
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 1. โหลดและเตรียมข้อมูล
# ---------------------------------------------------------------
xl = pd.ExcelFile('online_retail_II_clean_final_v2.xlsx')
df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['StockCode'] = df['StockCode'].astype(str)

os.makedirs('app_data', exist_ok=True)
df['Sales_Value'] = df['Quantity'] * df['Price']
desc = df.drop_duplicates('StockCode').set_index('StockCode')['Description'].str.strip().to_dict()

# ---------------------------------------------------------------
# 2. กฎคู่สินค้า Regular vs Festive (FP-Growth)
# ---------------------------------------------------------------
mon = df['InvoiceDate'].dt.month
periods = {'Regular': df[mon <= 10], 'Festive': df[mon >= 11]}
out = []
for name, sub in periods.items():
    tx = sub.groupby('Invoice')['StockCode'].apply(lambda x: list(set(x))).tolist()
    te = TransactionEncoder()
    arr = te.fit(tx).transform(tx, sparse=True)
    enc = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_)
    its = fpgrowth(enc, min_support=0.02, use_colnames=True)
    r = association_rules(its, metric='confidence', min_threshold=0.2)
    r = r[r['lift'] > 1].copy()
    r['period'] = name
    r['antecedents'] = r['antecedents'].apply(lambda s: ' + '.join(desc.get(c, c) for c in sorted(s)))
    r['consequents'] = r['consequents'].apply(lambda s: ' + '.join(desc.get(c, c) for c in sorted(s)))
    out.append(r[['period', 'antecedents', 'consequents', 'support', 'confidence', 'lift']])
rules = pd.concat(out).sort_values(['period', 'lift'], ascending=[True, False])
rules.to_csv('app_data/rules.csv', index=False)

# ---------------------------------------------------------------
# 3. ภาพรวมยอดขาย
# ---------------------------------------------------------------
ym = df['InvoiceDate'].dt.to_period('M').astype(str)
monthly = df.groupby(ym).agg(Sales_Value=('Sales_Value', 'sum'), Quantity=('Quantity', 'sum')).reset_index()
monthly.columns = ['Month', 'Sales_Value', 'Quantity']
monthly.to_csv('app_data/monthly_sales.csv', index=False)

dow = df.groupby(df['InvoiceDate'].dt.day_name())['Sales_Value'].sum().reset_index()
dow.columns = ['Day', 'Sales_Value']
dow.to_csv('app_data/day_of_week.csv', index=False)

topq = df.groupby('StockCode').agg(Quantity=('Quantity', 'sum')).sort_values('Quantity', ascending=False).head(10).reset_index()
topq['Description'] = topq['StockCode'].map(desc)
topq.to_csv('app_data/top10_quantity.csv', index=False)

# ---------------------------------------------------------------
# 4. พยากรณ์ (Holt-Winters & Moving Average)
# ---------------------------------------------------------------
dfx = df[df['StockCode'] != '23843']
mq = dfx.groupby([ym[dfx.index], 'StockCode'])['Quantity'].sum().unstack(fill_value=0).sort_index().iloc[:-1]
total = mq.sum(axis=1).astype(float)
TEST = 6

def hw(train, steps):
    return ExponentialSmoothing(train.clip(lower=0.1), seasonal='add', seasonal_periods=12,
                                initialization_method='legacy-heuristic').fit().forecast(steps).values

# 4.1 Backtest ยอดรวม
tr = len(total) - TEST
bt = pd.DataFrame({'Month': total.index[tr:], 'Actual': total.values[tr:],
                   'Forecast': np.clip(hw(total.iloc[:tr], TEST), 0, None)})
bt.to_csv('app_data/forecast_backtest.csv', index=False)

# คำนวณ Metrics ทั้งหมดสำหรับแสดงหน้าเว็บ
mae_total = float(np.mean(np.abs(bt.Actual - bt.Forecast)))
rmse_total = float(np.sqrt(np.mean((bt.Actual - bt.Forecast) ** 2)))
mape_total = float(np.mean(np.abs((bt.Actual - bt.Forecast) / bt.Actual)) * 100)
wape_total = float(np.abs(bt.Actual - bt.Forecast).sum() / bt.Actual.sum() * 100)

# 4.2 พยากรณ์ล่วงหน้า 3 เดือน
fut_idx = pd.period_range(pd.Period(total.index[-1]) + 1, periods=3, freq='M').astype(str)
fut = pd.DataFrame({'Month': fut_idx, 'Forecast': np.clip(hw(total, 3), 0, None)})
fut.to_csv('app_data/forecast_future.csv', index=False)

# 4.3 Top 10 สินค้า: Moving Average 3 เดือน
top10 = [c for c in mq.sum().sort_values(ascending=False).index][:10]
rows = []
for c in top10:
    s = mq[c].astype(float)
    t = s.iloc[tr:].values
    f = s.iloc[tr-3:tr].mean()
    wape = np.abs(t - f).sum() / t.sum() * 100
    rows.append([c, desc.get(c, c), s.iloc[-3:].tolist()[0], s.iloc[-3:].tolist()[1], s.iloc[-3:].tolist()[2],
                 round(s.iloc[-3:].mean()), round(wape, 1)])
sku = pd.DataFrame(rows, columns=['StockCode', 'Description', 'M-2', 'M-1', 'M0', 'Forecast_next_month', 'WAPE_backtest_%'])
sku.to_csv('app_data/forecast_top10_sku.csv', index=False)

# ---------------------------------------------------------------
# 5. ข้อมูลรายวันต่อ SKU และสร้าง meta.csv
# ---------------------------------------------------------------
daily = df.assign(Date=df['InvoiceDate'].dt.normalize()).groupby(['Date', 'StockCode']).agg(
    Quantity=('Quantity', 'sum'), Sales_Value=('Sales_Value', 'sum')).reset_index()
daily['Description'] = daily['StockCode'].map(desc)
daily.to_csv('app_data/daily_sales.csv.gz', index=False, compression='gzip')

pd.DataFrame([{
    'mae_total': round(mae_total, 2),
    'rmse_total': round(rmse_total, 2),
    'mape_total': round(mape_total, 1),
    'wape_total': round(wape_total, 1),
    'last_month': total.index[-1]
}]).to_csv('app_data/meta.csv', index=False)

print('เสร็จสมบูรณ์! ไฟล์ทั้งหมดถูกสร้างขึ้นในโฟลเดอร์ app_data/:', os.listdir('app_data'))