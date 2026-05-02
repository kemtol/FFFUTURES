from service.indicators import Indicators as ind

def apply_strategy(df, params):
    fast_p = int(params.get('fast', 20))
    slow_p = int(params.get('slow', 50))
    
    # Memanggil dari library yang sudah dinormalisasi
    df['fast_sma'] = ind.sma(df, period=fast_p)
    df['slow_sma'] = ind.sma(df, period=slow_p)
    
    df['signal'] = 0
    df.loc[(df['fast_sma'] > df['slow_sma']) & (df['fast_sma'].shift(1) <= df['slow_sma'].shift(1)), 'signal'] = 1
    df.loc[(df['fast_sma'] < df['slow_sma']) & (df['fast_sma'].shift(1) >= df['slow_sma'].shift(1)), 'signal'] = -1
    
    return df
