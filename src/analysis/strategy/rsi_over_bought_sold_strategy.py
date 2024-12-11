import pandas as pd
import numpy as np
from src.analysis.strategy.strategy import Strategy
from src.utils.technical_indicators import TechnicalIndicators


class RSIOverboughtOversoldStrategy(Strategy):
    """
        Genera segnali di trading basati sui livelli di ipercomprato e ipervenduto dell'RSI (Relative Strength Index).

        Questa funzione utilizza l'indicatore RSI per determinare condizioni di ipercomprato e ipervenduto su un asset:
        - Genera un segnale di acquisto (1.0) quando l'RSI scende al di sotto del livello di ipervenduto.
        - Genera un segnale di vendita (-1.0) quando l'RSI supera il livello di ipercomprato.
        - Non genera segnali (0.0) quando l'RSI è compreso tra i livelli di ipercomprato e ipervenduto.

        I segnali sono calcolati per ogni giorno e vengono mantenuti fino a quando non viene generato un nuovo segnale.

        Parametri:
        - data (pd.DataFrame): Un DataFrame contenente i dati storici dell'asset, con una colonna 'Close' per i prezzi di chiusura.
    """
    def __init__(self, rsi_period=14, overbought=70, oversold=30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['RSI'] = TechnicalIndicators.calculate_indicator('RSI', data['Close'], timeperiod=self.rsi_period)
        
        signals['signal'] = np.where(signals['RSI'] < self.oversold, 1.0, 
                                     np.where(signals['RSI'] > self.overbought, -1.0, 0.0))
        
        # Manteniamo la posizione fino a un nuovo segnale
        signals['positions'] = signals['signal']
        
        return signals