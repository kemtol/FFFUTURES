import vectorbt as vbt
import pandas as pd

class Indicators:
    @staticmethod
    def sma(close, period=20):
        return vbt.MA.run(close, window=period, ema=False).ma

    @staticmethod
    def ema(close, period=20):
        return vbt.MA.run(close, window=period, ema=True).ma

    @staticmethod
    def rsi(close, period=14):
        return vbt.RSI.run(close, window=period).rsi

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        macd_ind = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
        return macd_ind.macd, macd_ind.signal, macd_ind.histogram

    @staticmethod
    def bbands(close, period=20, std_dev=2):
        bb = vbt.BBANDS.run(close, window=period, alpha=std_dev)
        return bb.upper, bb.middle, bb.lower
