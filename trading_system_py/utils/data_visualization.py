import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import talib as ta

from trading_system_py.utils.technical_indicators import TechnicalIndicatorsDataObject



class DataVisualizationStock:
    def __init__(self, data: pd.DataFrame):
        """
        Inizializza l'oggetto DataVisualization.

        :param data: Un DataFrame contenente i dati OHLC e i volumi.
                     Deve includere le colonne 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        self.data = data
        self.timestamp = self.data.index
        
        # Creiamo un oggetto TechnicalIndicatorsDataObject per calcolare gli indicatori
        self.ti_data_object = TechnicalIndicatorsDataObject(
            open_prices=self.data['Open'], 
            high_prices=self.data['High'], 
            low_prices=self.data['Low'], 
            close_prices=self.data['Close'], 
            volumes=self.data['Volume']
        )

    def plot_ohlc(self, title: str = "OHLC Chart"):
        """
        Visualizza un grafico OHLC (Open-High-Low-Close).

        :param title: Titolo del grafico.
        :return: Visualizza il grafico OHLC.
        """
        fig = go.Figure(data=[go.Candlestick(x=self.data.index,
                                             open=self.data['Open'],
                                             high=self.data['High'],
                                             low=self.data['Low'],
                                             close=self.data['Close'],
                                             name='OHLC')])

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()

    def plot_ohlc_with_sma(self, windows: list[int], title: str = "OHLC with SMAs"):
        """
        Visualizza un grafico OHLC con una o più medie mobili semplici (SMA).

        :param windows: Una lista di finestre temporali per calcolare le medie mobili.
        :param title: Titolo del grafico.
        :return: Visualizza il grafico OHLC con SMA.
        """
        fig = go.Figure()

        # Aggiungi il grafico OHLC
        fig.add_trace(go.Candlestick(x=self.data.index,
                                     open=self.data['Open'],
                                     high=self.data['High'],
                                     low=self.data['Low'],
                                     close=self.data['Close'],
                                     name='OHLC'))

        # Calcola e aggiungi ogni SMA
        for window in windows:
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[f'SMA_{window}'],
                                     mode='lines', name=f'SMA {window}'))

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()

    def plot_sma(self, windows: list[int], title: str = "Simple Moving Averages (SMA)"):
        """
        Visualizza una o più medie mobili semplici (SMA) sui dati di chiusura.

        :param windows: Una lista di finestre temporali per calcolare le medie mobili.
        :param title: Titolo del grafico.
        :return: Visualizza le SMA.
        """
        fig = go.Figure()

        # Aggiungi il grafico dei prezzi
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Close Price'))

        # Calcola e aggiungi ogni SMA
        for window in windows:
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[f'SMA_{window}'],
                                     mode='lines', name=f'SMA {window}'))

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        fig.show()

    def plot_ema(self, windows: list[int], title: str = "Exponential Moving Averages (EMA)"):
        """
        Visualizza una o più medie mobili esponenziali (EMA) sui dati di chiusura.

        :param windows: Una lista di finestre temporali per calcolare le EMA.
        :param title: Titolo del grafico.
        :return: Visualizza le EMA.
        """
        fig = go.Figure()

        # Aggiungi il grafico dei prezzi
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Close Price'))

        # Calcola e aggiungi ogni EMA
        for window in windows:
            self.data[f'EMA_{window}'] = ta.EMA(self.data['Close'], timeperiod=window)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[f'EMA_{window}'],
                                     mode='lines', name=f'EMA {window}'))

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        fig.show()

    def plot_macd(self, title: str = "MACD"):
        """
        Visualizza il MACD (Moving Average Convergence Divergence).

        :param title: Titolo del grafico.
        :return: Visualizza il grafico del MACD.
        """
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = ta.MACD(self.data['Close'])

        fig = go.Figure()

        # MACD line
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD'], mode='lines', name='MACD Line'))
        # Signal line
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD_signal'], mode='lines', name='Signal Line'))
        # MACD histogram
        fig.add_trace(go.Bar(x=self.data.index, y=self.data['MACD_hist'], name='MACD Histogram'))

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="MACD")
        fig.show()
    
    def plot_rsi(self, title: str = "RSI (Relative Strength Index)", period: int = 14):
        """
        Visualizza l'RSI (Relative Strength Index).

        :param title: Titolo del grafico.
        :param period: Il numero di giorni per calcolare l'RSI (default=14).
        :return: Visualizza il grafico dell'RSI.
        """
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=period)

        fig = go.Figure()

        # RSI line
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['RSI'], mode='lines', name='RSI'))

        # Aggiungi le linee di soglia per ipercomprato e ipervenduto
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought (70)')
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold (30)')

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="RSI")
        fig.show()

    def plot_bollinger_bands(self, window: int = 20, num_std_dev: float = 2, title: str = "Bollinger Bands"):
        """
        Visualizza un grafico OHLC con le Bande di Bollinger.

        :param window: Finestra temporale per calcolare la media mobile semplice (SMA).
        :param num_std_dev: Numero di deviazioni standard da utilizzare per le bande superiore e inferiore.
        :param title: Titolo del grafico.
        :return: Visualizza il grafico OHLC con le Bande di Bollinger.
        """
        # Calcolo delle Bande di Bollinger
        self.data['SMA'] = self.data['Close'].rolling(window=window).mean()
        self.data['Upper Band'], self.data['Middle Band'], self.data['Lower Band'] = ta.BBANDS(
            self.data['Close'], timeperiod=window, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0)

        fig = go.Figure()

        # Aggiungi il grafico OHLC
        fig.add_trace(go.Candlestick(x=self.data.index,
                                     open=self.data['Open'],
                                     high=self.data['High'],
                                     low=self.data['Low'],
                                     close=self.data['Close'],
                                     name='OHLC'))

        # Aggiungi le bande di Bollinger
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Upper Band'],
                                 mode='lines', line=dict(color='red', width=1), name='Upper Band'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Middle Band'],
                                 mode='lines', line=dict(color='blue', width=1), name='Middle Band (SMA)'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Lower Band'],
                                 mode='lines', line=dict(color='green', width=1), name='Lower Band'))

        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()

    def plot_combined(self, plots: list[tuple] = [('OHLC', {}), ('SMA', {'timeperiod': 50})], title: str = "Combined Plots", row_heights: list = None):
        # Separiamo i grafici OHLC/SMA/EMA/BBANDS dagli altri indicatori
        ohlc_plots = [plot for plot in plots if plot[0] in ['OHLC', 'DEMA', 'EMA', 'MA', 'SMA', 'KAMA', 'MAMA', 'MAVP', 'T3', 'TEMA', 'TRIMA', 'WMA', 'BBANDS']]
        indicator_plots = [plot for plot in plots if plot[0] not in ['OHLC', 'DEMA', 'EMA', 'MA', 'SMA', 'KAMA', 'MAMA', 'MAVP', 'T3', 'TEMA', 'TRIMA', 'WMA', 'BBANDS']]

        # Numero di subplot: uno per OHLC con SMA/EMA e un subplot per ciascun indicatore
        num_plots = 1 + len(indicator_plots)

        # Creazione del layout per i subplot con row_heights predefiniti
        default_row_heights = [0.5] + [0.15] * len(indicator_plots)
        if row_heights is None:
            row_heights = default_row_heights

        # Creiamo una figura con subplot
        fig = make_subplots(
            rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=[p[0].upper() for p in plots if p[0] not in ['DEMA', 'EMA', 'MA', 'SMA', 'KAMA', 'MAMA', 'MAVP', 'T3', 'TEMA', 'TRIMA', 'WMA', 'BBANDS']],
            row_heights=row_heights
        )

        ohlc_added = False
        row_id_ohlc = 1

        # Iterazione attraverso i plot specificati
        for plot_type, params in ohlc_plots:
            if plot_type == 'OHLC' and not ohlc_added:
                # Aggiungiamo il grafico OHLC alla prima riga
                fig.add_trace(go.Candlestick(x=self.data.index, open=self.data['Open'], high=self.data['High'],
                                            low=self.data['Low'], close=self.data['Close'], name='OHLC'), row=row_id_ohlc, col=1)
                ohlc_added = True
            
            elif plot_type in ['DEMA', 'EMA', 'MA', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA', 'KAMA', 'MAMA', 'MAVP']:
                # Calcoliamo l'indicatore tramite la funzione `calculate_indicator`
                indicator_values = self.ti_data_object.calculate_indicator(plot_type, **params)
                fig.add_trace(go.Scatter(x=self.data.index, y=indicator_values,
                                         mode='lines', name=f'{plot_type} {params.get("timeperiod", "")}'),
                              row=row_id_ohlc, col=1)

            elif plot_type == 'BBANDS':
                # Calcoliamo le Bande di Bollinger tramite la funzione `calculate_indicator`
                upper_band, middle_band, lower_band = self.ti_data_object.calculate_indicator('BBANDS', **params)
                fig.add_trace(go.Scatter(x=self.data.index, y=upper_band, mode='lines', name='Upper Band',
                                        line=dict(color='red')), row=row_id_ohlc, col=1)
                fig.add_trace(go.Scatter(x=self.data.index, y=middle_band, mode='lines', name='Middle Band',
                                        line=dict(color='blue')), row=row_id_ohlc, col=1)
                fig.add_trace(go.Scatter(x=self.data.index, y=lower_band, mode='lines', name='Lower Band',
                                        line=dict(color='green')), row=row_id_ohlc, col=1)

        # Aggiungiamo gli indicatori (RSI, MACD, Volume, ecc.) nei subplot separati
        for i, (plot_type, params) in enumerate(indicator_plots, start=2):  
            if plot_type == 'RSI':
                rsi_values = self.ti_data_object.calculate_indicator('RSI', **params)
                fig.add_trace(go.Scatter(x=self.data.index, y=rsi_values, mode='lines', name='RSI'), row=i, col=1)
                fig.add_hline(y=70, row=i, col=1, line=dict(color='red', dash='dash'))
                fig.add_hline(y=30, row=i, col=1, line=dict(color='green', dash='dash'))
            
            elif plot_type == 'MACD':
                macd_values, macd_signal, macd_hist = self.ti_data_object.calculate_indicator('MACD', **params)
                fig.add_trace(go.Scatter(x=self.data.index, y=macd_values, mode='lines', name='MACD Line'), row=i, col=1)
                fig.add_trace(go.Scatter(x=self.data.index, y=macd_signal, mode='lines', name='Signal Line'), row=i, col=1)
                fig.add_trace(go.Bar(x=self.data.index, y=macd_hist, name='MACD Histogram'), row=i, col=1)

            elif plot_type == 'VOLUME':
                bar_color = params.get('bar_color', 'blue')  # Default color is blue
                fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], name='Volume', marker_color=bar_color), row=i, col=1)

        # Aggiornamento del layout
        fig.update_layout(title_text=title, showlegend=True, height=450*num_plots)
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()

