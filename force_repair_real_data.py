
import yfinance as yf
import sqlite3
from pathlib import Path

DB_PATH = "data/Live/topstepx_buffer.db"

def force_repair():
    print("Fetching fresh 1m data from yfinance...")
    data = yf.download("MGC=F", interval="1m", period="1d")
    if data.empty:
        print("No data found.")
        return

    conn = sqlite3.connect(DB_PATH)
    count = 0
    for ts, row in data.iterrows():
        ts_utc = ts.strftime('%Y-%m-%d %H:%M:%S')
        epoch = int(ts.timestamp() * 1000)
        
        o = float(row['Open'].iloc[0]) if hasattr(row['Open'], 'iloc') else float(row['Open'])
        h = float(row['High'].iloc[0]) if hasattr(row['High'], 'iloc') else float(row['High'])
        l = float(row['Low'].iloc[0]) if hasattr(row['Low'], 'iloc') else float(row['Low'])
        c = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
        v = float(row['Volume'].iloc[0]) if hasattr(row['Volume'], 'iloc') else float(row['Volume'])

        # Using INSERT OR REPLACE to update those synthetic bars with REAL data
        conn.execute(
            "INSERT OR REPLACE INTO ohlcv_1m (symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("MICRO_GOLD", "1m", epoch, ts_utc, o, h, l, c, v)
        )
        count += 1
    
    conn.commit()
    conn.close()
    print(f"✅ Success: Updated database with REAL market data. Processed {count} rows.")

if __name__ == "__main__":
    force_repair()
