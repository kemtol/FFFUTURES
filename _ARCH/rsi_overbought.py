import pandas as pd
import numpy as np

def apply_strategy(df, params):
    period = int(params.get('rsi_period', 14))
    overbought = int(params.get('overbought', 70))
    oversold = int(params.get('oversold', 30))
    
    # Hitung RSI sederhana
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['signal'] = 0
    # Buy jika RSI di bawah oversold (oversold reversal)
    df.loc[(df['rsi'] < oversold), 'signal'] = 1
    # Sell jika RSI di atas overbought
    df.loc[(df['rsi'] > overbought), 'signal'] = -1
    
    return df
