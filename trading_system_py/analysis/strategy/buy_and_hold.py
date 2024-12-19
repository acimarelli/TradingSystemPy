import pandas as pd
from src.analysis.strategy.strategy import Strategy


class BuyAndHold(Strategy):
    """
    Strategia di trading che prevede l'acquisto all'inizio per poi mantenere la posizione long. 
    """
    def generate_signals(self, data):
        """
        Genera segnali per una strategia di acquisto e mantenimento.
        
        Parametri:
        - data: pd.DataFrame
            Dati storici dell'asset. È necessario che contenga almeno una colonna 'Close'.
        
        Ritorna:
        - signals: pd.DataFrame
            DataFrame contenente le colonne 'signal' e 'positions'.
            'signal' è 1.0 all'inizio e 0.0 per il resto del periodo.
            'positions' è 1.0 per mantenere la posizione acquistata.
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['signal'].iloc[0] = 1.0  # Compra all'inizio

        # Mantieni la posizione fino alla fine
        signals['positions'] = signals['signal']
        
        return signals  