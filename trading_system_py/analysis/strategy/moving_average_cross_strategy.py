import pandas as pd
import numpy as np
from trading_system_py.analysis.strategy.strategy import Strategy
from trading_system_py.utils.technical_indicators import TechnicalIndicators


class MovingAverageCrossStrategy(Strategy):
    """
    Strategia di trading basata sull'incrocio delle medie mobili.

    Questa strategia utilizza due medie mobili semplici (SMA), una a breve termine e una a lungo termine, per generare segnali di trading.
    Quando la media mobile a breve termine supera quella a lungo termine, viene generato un segnale di acquisto (long).
    Quando la media mobile a breve termine scende al di sotto di quella a lungo termine, viene generato un segnale di vendita (short).

    Attributi:
        short_window (int): La finestra temporale della media mobile a breve termine. Default: 20.
        long_window (int): La finestra temporale della media mobile a lungo termine. Default: 50.

    Metodi:
        generate_signals(data):
            Genera i segnali di trading basati sull'incrocio delle medie mobili.
            - Calcola le medie mobili a breve e lungo termine.
            - Genera un segnale di acquisto (1.0) quando la media a breve termine supera quella a lungo termine.
            - Genera un segnale di vendita (-1.0) quando la media a breve termine scende sotto quella a lungo termine.
            - Registra i cambiamenti di posizione in una colonna separata ('positions').

    Parametri:
        short_window (int, opzionale): Periodo per la media mobile a breve termine. Default: 20.
        long_window (int, opzionale): Periodo per la media mobile a lungo termine. Default: 50.
    """
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = TechnicalIndicators.calculate_indicator('SMA', data['Close'], timeperiod=self.short_window)
        signals['long_mavg'] = TechnicalIndicators.calculate_indicator('SMA', data['Close'], timeperiod=self.long_window)
        
        signals['signal'][self.short_window:] = np.where(
            signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)
        
        # Cambio di posizione solo su cambio di segnale
        signals['positions'] = signals['signal'].diff().fillna(0.0)
        
        return signals