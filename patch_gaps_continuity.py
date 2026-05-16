
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "data/Live/topstepx_buffer.db"

def fill_physical_gaps():
    conn = sqlite3.connect(DB_PATH)
    # 1. Load data into DataFrame
    df = pd.read_sql("SELECT * FROM ohlcv_1m ORDER BY epoch_ms ASC", conn)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    if df.empty:
        return

    # 2. Identify expected range
    start_dt = df['timestamp_utc'].min()
    end_dt = df['timestamp_utc'].max()
    expected_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')
    
    # 3. Reindex to find missing minutes
    df.set_index('timestamp_utc', inplace=True)
    df_filled = df.reindex(expected_range)
    
    # 4. Fill missing values (Forward Fill prices, 0 for volume)
    missing_mask = df_filled['close'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("No physical gaps detected. Continuity is perfect.")
        conn.close()
        return

    print(f"Detected {missing_count} missing minutes. Interpolating...")
    
    # Forward fill prices (Open, High, Low, Close get the last known Close)
    df_filled['close'] = df_filled['close'].ffill()
    df_filled['open'] = df_filled['open'].fillna(df_filled['close'])
    df_filled['high'] = df_filled['high'].fillna(df_filled['close'])
    df_filled['low'] = df_filled['low'].fillna(df_filled['close'])
    df_filled['volume'] = df_filled['volume'].fillna(0)
    df_filled['symbol'] = df_filled['symbol'].ffill()
    df_filled['timeframe'] = df_filled['timeframe'].ffill()
    
    # 5. Insert only the missing rows back to DB
    new_rows = df_filled[missing_mask].copy()
    new_rows.index.name = 'timestamp_utc'
    new_rows.reset_index(inplace=True)
    
    added = 0
    for _, row in new_rows.iterrows():
        ts_str = row['timestamp_utc'].strftime('%Y-%m-%d %H:%M:%S')
        epoch = int(row['timestamp_utc'].timestamp() * 1000)
        
        try:
            conn.execute(
                "INSERT OR IGNORE INTO ohlcv_1m (symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row['symbol'], row['timeframe'], epoch, ts_str, row['open'], row['high'], row['low'], row['close'], row['volume'])
            )
            added += 1
        except Exception as e:
            print(f"Error inserting {ts_str}: {e}")
            
    conn.commit()
    conn.close()
    print(f"✅ Success: Closed {added} gaps physically in the database.")

if __name__ == "__main__":
    fill_physical_gaps()
