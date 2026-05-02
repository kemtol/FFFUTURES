from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import uvicorn
import pandas as pd
import os
import importlib.util
import vectorbt as vbt
from datetime import datetime
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/MGC_1m.db"

def get_db_info():
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM investing_ohlcv_1m").fetchall()]
    bounds = conn.execute("SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM investing_ohlcv_1m").fetchone()
    conn.close()
    return symbols, bounds

@app.get("/")
def home():
    symbols, bounds = get_db_info()
    symbol_options = "".join([f'<option value="{s}">{s}</option>' for s in symbols])
    db_min = bounds[0].split(' ')[0] if bounds[0] else "2010-01-01"
    db_max = bounds[1].split(' ')[0] if bounds[1] else "2026-04-17"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Professional Backtest Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.23.4/ace.js"></script>
    <style>
        body {{ background: #0b0e11; color: #d1d4dc; font-family: 'Inter', sans-serif; overflow: hidden; }}
        .sidebar {{ background: #161a1e; border-right: 1px solid #2b2f3a; height: 100vh; padding: 20px; overflow-y: auto; }}
        .main-content {{ height: 100vh; display: flex; flex-direction: column; }}
        #chart {{ flex: 1; background: #0b0e11; }}
        #editor {{ height: 350px; border: 1px solid #2b2f3a; border-radius: 4px; }}
        .report-section {{ height: 250px; overflow-y: auto; background: #161a1e; border-top: 1px solid #2b2f3a; padding: 15px; }}
        .form-control, .form-select {{ background: #1e222d !important; border: 1px solid #2b2f3a !important; color: #d1d4dc !important; font-size: 0.85rem; }}
        .btn-primary {{ background: #2962ff; border: none; font-weight: 600; }}
        .metric-box {{ background: #1c2127; border: 1px solid #2b2f3a; padding: 10px; border-radius: 4px; text-align: center; flex: 1; }}
        .metric-label {{ font-size: 10px; color: #848e9c; text-transform: uppercase; }}
        .metric-value {{ font-size: 15px; font-weight: bold; color: #fff; }}
        .table {{ color: #d1d4dc; font-size: 12px; }}
        .win {{ color: #26a69a; }} .loss {{ color: #ef5350; }}
    </style>
</head>
<body>
    <div class="container-fluid p-0">
        <div class="row g-0">
            <div class="col-md-3 sidebar">
                <h5 class="mb-4 text-white">Backtest Engine</h5>
                <div class="mb-3">
                    <label class="small text-secondary fw-bold">1. SYMBOL</label>
                    <select id="symbol" class="form-select">{symbol_options}</select>
                </div>
                <div class="mb-3">
                    <label class="small text-secondary fw-bold">2. DATE RANGE</label>
                    <input type="date" id="start" class="form-control mb-1" value="2026-04-10" min="{db_min}" max="{db_max}">
                    <input type="date" id="end" class="form-control" value="{db_max}" min="{db_min}" max="{db_max}">
                </div>
                <div class="mb-3">
                    <label class="small text-secondary fw-bold">3. CODE</label>
                    <div id="editor">from service.indicators import Indicators as ind

def apply_strategy(df):
    close = df['close']
    df['fast'] = ind.sma(close, 20)
    df['slow'] = ind.sma(close, 50)
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1
    df.loc[df['fast'] < df['slow'], 'signal'] = -1
    return df</div>
                </div>
                <button onclick="run()" id="btn-run" class="btn btn-primary w-100 py-2">RUN BACKTEST</button>
            </div>
            <div class="col-md-9 main-content">
                <div class="p-2 d-flex gap-2 bg-dark shadow-sm">
                    <div class="metric-box"><div class="metric-label">Net Profit</div><div id="m-p" class="metric-value">-</div></div>
                    <div class="metric-box"><div class="metric-label">Win Rate</div><div id="m-w" class="metric-value">-</div></div>
                    <div class="metric-box"><div class="metric-label">Trades</div><div id="m-t" class="metric-value">-</div></div>
                    <div class="metric-box"><div class="metric-label">Max DD</div><div id="m-d" class="metric-value">-</div></div>
                </div>
                <div id="chart"></div>
                <div class="report-section">
                    <table class="table table-dark table-sm">
                        <thead><tr><th>Entry</th><th>Exit</th><th>Price In</th><th>Price Out</th><th>PnL</th></tr></thead>
                        <tbody id="trades"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    <script>
        let chart, series, editor;
        function init() {{
            editor = ace.edit("editor");
            editor.setTheme("ace/theme/monokai");
            editor.session.setMode("ace/mode/python");
            const chartEl = document.getElementById('chart');
            chart = LightweightCharts.createChart(chartEl, {{
                layout: {{ background: {{ color: '#0b0e11' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ visible: false }}, horzLines: {{ visible: false }} }},
                timeScale: {{ timeVisible: true }}
            }});
            series = chart.addCandlestickSeries();
            window.addEventListener('resize', () => chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }}));
            loadData();
        }}
        async function loadData() {{
            const symbol = document.getElementById('symbol').value;
            const start = document.getElementById('start').value;
            const end = document.getElementById('end').value;
            const data = await fetch(`/api/data?symbol=${{symbol}}&start=${{start}}&end=${{end}}`).then(r => r.json());
            if (data.length > 0) series.setData(data);
            chart.timeScale().fitContent();
        }}
        async function run() {{
            const btn = document.getElementById('btn-run');
            btn.disabled = true; btn.innerText = "RUNNING...";
            const payload = {{
                symbol: document.getElementById('symbol').value,
                start: document.getElementById('start').value,
                end: document.getElementById('end').value,
                code: editor.getValue()
            }};
            try {{
                await loadData();
                const resp = await fetch('/api/run', {{
                    method: 'POST', headers: {{'Content-Type':'application/json'}},
                    body: JSON.stringify(payload)
                }});
                const res = await resp.json();
                if (res.error) return alert(res.error);
                document.getElementById('m-p').innerText = `$${{res.metrics.profit.toFixed(2)}}`;
                document.getElementById('m-w').innerText = `${{res.metrics.win_rate.toFixed(1)}}%`;
                document.getElementById('m-t').innerText = res.metrics.trades;
                document.getElementById('m-d').innerText = `${{res.metrics.max_dd.toFixed(1)}}%`;
                document.getElementById('trades').innerHTML = res.trades.map(t => `
                    <tr><td>${{t.in_t}}</td><td>${{t.out_t}}</td><td>${{t.in_p.toFixed(2)}}</td><td>${{t.out_p.toFixed(2)}}</td><td class="${{t.pnl>=0?'win':'loss'}}">${{t.pnl.toFixed(2)}}</td></tr>
                `).join('');
            }} catch(e) {{ alert("Error connection"); }}
            finally {{ btn.disabled = false; btn.innerText = "RUN BACKTEST"; }}
        }}
        window.onload = init;
    </script>
</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/api/data")
def api_data(symbol: str, start: str, end: str):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT (epoch_ms/1000) as time, timestamp_utc, open, high, low, close FROM investing_ohlcv_1m WHERE symbol=? AND timestamp_utc>=? AND timestamp_utc<=? ORDER BY epoch_ms ASC LIMIT 2000", (symbol, f"{start} 00:00:00", f"{end} 23:59:59")).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/run")
async def api_run(data: dict = Body(...)):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT open, high, low, close, timestamp_utc as date FROM investing_ohlcv_1m WHERE symbol=? AND timestamp_utc>=? AND timestamp_utc<=? ORDER BY epoch_ms ASC", conn, params=(data['symbol'], f"{data['start']} 00:00:00", f"{data['end']} 23:59:59"))
        conn.close()
        if df.empty: return {"error": "No data found"}
        with open("temp_strat.py", "w") as f: f.write(data['code'])
        spec = importlib.util.spec_from_file_location("mod", "temp_strat.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        df = mod.apply_strategy(df)
        pf = vbt.Portfolio.from_signals(df['close'], entries=(df['signal']==1), exits=(df['signal']==-1), init_cash=10000, freq='1m')
        trades = []
        if not pf.trades.records.empty:
            for _, r in pf.trades.records_readable.iterrows():
                trades.append({{"in_t": str(df['date'].iloc[int(r['Entry Index'])]), "out_t": str(df['date'].iloc[int(r['Exit Index'])]), "in_p": float(r['Entry Price']), "out_p": float(r['Exit Price']), "pnl": float(r['PnL'])}})
        return {{ "metrics": {{"profit": float(pf.total_profit()), "win_rate": float(pf.win_rate()*100) if not pd.isna(pf.win_rate()) else 0, "trades": int(pf.total_trades()), "max_dd": float(pf.max_drawdown()*100)}}, "trades": trades[::-1] }}
    except Exception as e: return {{"error": str(e)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=45678)
