import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

from src.analysis.strategy.strategy import Strategy
from src.utils.technical_indicators import TechnicalIndicators


class LSTMNeuralNetworkStrategy(Strategy):
    """
        Genera segnali di trading utilizzando una rete neurale LSTM.

        Questa funzione utilizza una rete neurale ricorrente (LSTM) per prevedere il movimento dei prezzi basandosi su una finestra temporale fissa (lookback). 
        La strategia è costruita per identificare segnali di acquisto o nessuna azione:
        - Genera un segnale di acquisto (1.0) se il modello prevede un incremento di prezzo.
        - Genera un segnale di nessuna azione (0.0) altrimenti.

        La rete viene addestrata internamente sui dati storici forniti, utilizzando il prezzo di chiusura normalizzato per il calcolo.

        ### Parametri:
        - `data` (pd.DataFrame): DataFrame con i dati storici dell'asset, contenente almeno la colonna 'Close'.

        ### Ritorna:
        - `signals` (pd.DataFrame): Un DataFrame con le seguenti colonne:
            - `'signal'`: Segnale di trading generato dal modello LSTM (1.0 per acquisto, 0.0 per nessuna azione).
            - `'positions'`: Differenza tra il segnale corrente e il precedente (mostra i cambiamenti di posizione).

        ### Processo:
        1. I dati vengono normalizzati nell'intervallo [0, 1] utilizzando `MinMaxScaler`.
        2. I dati di training sono creati:
           - `x_train`: Sequenze temporali di lunghezza pari a `lookback`.
           - `y_train`: Etichetta binaria che indica se il prezzo di chiusura corrente è maggiore del precedente.
        3. Il modello LSTM viene addestrato per una singola epoca utilizzando i dati di training.
        4. I segnali sono generati facendo previsioni sul set di dati completo, basandosi sulle sequenze di input.

        ### Note:
        - Il modello è progettato per essere minimale e addestrato rapidamente, ma il numero di epoche e il batch size possono essere regolati per migliorare la performance.
        - La colonna 'Close' è necessaria per i calcoli. Altri dati nel DataFrame verranno ignorati.
        - Per risultati migliori, si consiglia di utilizzare un dataset sufficientemente ampio per il training.
        """
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def generate_signals(self, data):
        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        x_train, y_train = [], []
        for i in range(self.lookback, len(scaled_data)):
            x_train.append(scaled_data[i-self.lookback:i, 0])
            y_train.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        for i in range(self.lookback, len(scaled_data)):
            input_data = scaled_data[i-self.lookback:i, 0].reshape(1, -1, 1)
            prediction = self.model.predict(input_data, verbose=0)
            signals['signal'].iloc[i] = 1.0 if prediction > 0.5 else 0.0
        
        # La posizione segue il segnale predittivo
        signals['positions'] = signals['signal'].diff().fillna(0.0)
        
        return signals