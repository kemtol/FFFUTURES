import databento as db
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DBN_PATH = "data/Level_0_Raw/databento/glbx-mdp3-20100606-20260415.ohlcv-1m.dbn.zst"
DB_PATH = "data/Level_0_Raw/MGC_1m.db"
TABLE = "investing_ohlcv_1m"

def ingest():
    if not Path(DBN_PATH).exists():
        print(f"Error: File {DBN_PATH} tidak ditemukan.")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Membersihkan data lama untuk MICRO_GOLD...")
    conn.execute(f"DELETE FROM {TABLE} WHERE symbol = 'MICRO_GOLD'")
    conn.commit()

    print(f"Membuka file Databento: {DBN_PATH}")
    store = db.DBNStore.from_file(DBN_PATH)
    
    # Mapping instrument_id (int) ke symbol string (kontrak)
    # Metadata 'symbol' adalah instrument_id dalam bentuk string
    valid_instrument_ids = set()
    for sym, mapping_list in store.metadata.mappings.items():
        if '-' in sym: # Abaikan spread
            continue
        for entry in mapping_list:
            if 'symbol' in entry:
                valid_instrument_ids.add(int(entry['symbol']))

    print(f"Ditemukan {len(valid_instrument_ids)} ID instrumen individu yang valid.")

    count = 0
    batch = []
    batch_size = 50000 # Batch lebih besar untuk kecepatan
    
    current_epoch_ms = None
    best_candle = None

    print("Memulai proses migrasi data (Highest Volume per Minute)...")
    
    def get_row(epoch_ms, candle):
        ts_utc = datetime.fromtimestamp(epoch_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        # Databento price is fixed precision 1e9
        return ("MICRO_GOLD", "1m", epoch_ms, ts_utc, 
                candle.open / 1_000_000_000.0, 
                candle.high / 1_000_000_000.0, 
                candle.low / 1_000_000_000.0, 
                candle.close / 1_000_000_000.0, 
                float(candle.volume))

    try:
        for record in store:
            if record.instrument_id not in valid_instrument_ids:
                continue
                
            epoch_ms = record.ts_event // 1_000_000
            
            if epoch_ms != current_epoch_ms:
                if best_candle is not None:
                    batch.append(get_row(current_epoch_ms, best_candle))
                    
                    if len(batch) >= batch_size:
                        conn.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                        conn.commit()
                        count += len(batch)
                        print(f"Telah menyimpan {count} menit...")
                        batch = []
                
                current_epoch_ms = epoch_ms
                best_candle = record
            else:
                if record.volume > best_candle.volume:
                    best_candle = record
                    
        if best_candle is not None:
            batch.append(get_row(current_epoch_ms, best_candle))
        if batch:
            conn.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()
            count += len(batch)
            
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
    finally:
        print(f"Selesai. Total menit unik (front month) yang disimpan: {count}")
        conn.close()

if __name__ == "__main__":
    ingest()
