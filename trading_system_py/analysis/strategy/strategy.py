class Strategy:
    def generate_signals(self):
        """
        Genera segnali di trading basati sulla strategia.

        Parametri:
        - data: pd.DataFrame
            Dati storici dell'asset.

        Ritorna un pd.DataFrame che deve avere tra le sue colonne:
            - signals: pd.Series contenente i segnali generati.
            - positions: pd.Series contenente le posizioni (acquisto/vendita).
        """
        raise NotImplementedError("Implementare il metodo generate_signals.")
    
    def get_strategy_name(self):
        return self.__class__.__name__
    
    