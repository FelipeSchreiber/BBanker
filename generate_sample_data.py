import yfinance as yf
import pandas as pd
from datetime import date, timedelta

# 1) Define tickers
tickers = [
    "AAPL","MSFT","AMZN","GOOGL","META",
    "TSLA","NVDA","JPM","BAC","XOM",
    "CVX","WMT","DIS","PFE","KO"
]

# 2) Define date range (last 3 years)
end_date = date.today()
start_date = end_date - timedelta(days=3*365)

# 3) Fetch historical data for all tickers
print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# 4) Extract Close prices (auto_adjust=True makes Close equivalent to Adj Close)
print("Processing data...")
print(f"Data columns: {data.columns.tolist()[:10]}...")
print(f"Data shape: {data.shape}")

# Check if data has MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    # Get Close from multi-index (which is adjusted when auto_adjust=True)
    adj_close = data.xs('Close', axis=1, level=0)
else:
    # Single ticker or different format
    if 'Close' in data.columns:
        adj_close = data[['Close']].copy()
        adj_close.columns = tickers
    else:
        adj_close = data

# 5) Save to Excel with multiple sheets
excel_file = "15_US_Stocks_3Y_Closing_Data.xlsx"
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    # Consolidated sheet
    adj_close.to_excel(writer, sheet_name="All_Tickers")

print(f"Excel file generated: {excel_file}")
