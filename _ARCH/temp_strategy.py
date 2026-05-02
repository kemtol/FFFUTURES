from service.indicators import Indicators as ind

def apply_strategy(df):
    # Strategy: Simple SMA Crossover
    close = df['close']
    df['fast'] = ind.sma(close, 20)
    df['slow'] = ind.sma(close, 50)
    
    df['signal'] = 0
    # Buy (1) when fast > slow
    df.loc[df['fast'] > df['slow'], 'signal'] = 1
    # Exit (-1) when fast < slow
    df.loc[df['fast'] < df['slow'], 'signal'] = -1
    return df