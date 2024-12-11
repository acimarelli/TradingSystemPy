import pandas as pd
import numpy as np
from src.analysis.strategy.strategy import Strategy
from src.utils.technical_indicators import TechnicalIndicators


class BollingerBandsStrategy(Strategy):
    """
        Genera segnali di trading basati sulle Bande di Bollinger.

        Questa funzione utilizza le Bande di Bollinger (Bollinger Bands, BBANDS) per identificare opportunità di trading:
        - Genera un segnale di acquisto (1.0) quando il prezzo di chiusura scende al di sotto della banda inferiore.
        - Genera un segnale di vendita (-1.0) quando il prezzo di chiusura supera la banda superiore.
        - Non genera segnali (0.0) quando il prezzo si trova tra le bande.

        I segnali sono calcolati per ogni giorno e vengono mantenuti fino a quando non viene generato un nuovo segnale.

        Parametri:
        - data (pd.DataFrame): Un DataFrame contenente i dati storici dell'asset, con una colonna 'Close' per i prezzi di chiusura.

        Ritorna:
        - signals (pd.DataFrame): Un DataFrame con le seguenti colonne:
            - 'signal': Segnale di trading (1.0 per acquisto, -1.0 per vendita, 0.0 per nessuna azione).
            - 'positions': Differenza tra il segnale corrente e il precedente (mostra i cambiamenti di posizione).
        """
    def __init__(self, timeperiod=20, nbdevup=2, nbdevdn=2):
        self.timeperiod = timeperiod
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_indicator(
            'BBANDS', data['Close'], timeperiod=self.timeperiod, nbdevup=self.nbdevup, nbdevdn=self.nbdevdn)
        
        signals['signal'] = np.where(data['Close'] < lower_band, 1.0, 
                                     np.where(data['Close'] > upper_band, -1.0, 0.0))
        
        # Manteniamo la posizione fino al successivo segnale
        signals['positions'] = signals['signal']

        return signals