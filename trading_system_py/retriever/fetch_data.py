import os
import yfinance as yf
import pandas as pd
import pickle
import plotly.graph_objects as go

import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from trading_system_py.retriever.data_object_manipulation import DataObjectManipulation
from trading_system_py.retriever.risk_free_object_manipulation import RiskFreeObjectManipulation
from trading_system_py.utils.data_visualization import DataVisualizationStock


class FetchSingleStock:
    
    # DOWNLOAD TIME SERIES FROM YAHOO FINANCE API
    @staticmethod
    def download_history(ticker: str, date_range: list[str] = None) -> pd.DataFrame:
        """
        Scarica la serie storica dei prezzi di un singolo titolo da Yahoo Finance.

        :param ticker: Il simbolo del titolo da scaricare.
        :param date_range: Un intervallo opzionale di date sotto forma di lista [start, end]. 
                           Se non viene fornito, scarica i dati per l'intero periodo disponibile.
        :return: Un pd.DataFrame con i dati storici del titolo.
        """
        return yf.Ticker(ticker).history(start=date_range[0], end=date_range[1]) if date_range is not None else \
               yf.Ticker(ticker).history(period='max')
    
    @staticmethod
    def resample_history_by_custom_bar_time_interval(data: pd.DataFrame, time_interval: str = None) -> pd.DataFrame:
        """
        Permette di ricostruire la serie storica dei prezzi di un singolo titolo secondo l'intervallo deciso dall'utente.

        :param data: Un pd.DataFrame con i dati storici del titolo presi mediante l'API di Yahoo Finance.
        :param time_interval: stringa che identifica l'ampiezza della barra. Se non viene fornito lascia immutata la base dati.
                              https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        :return: Un pd.DataFrame con i dati storici del titolo gruppati secondo l'intervallo scelto.
        """
        if time_interval is None:
            return data
        else:
            return data.groupby(pd.Grouper(freq=time_interval)).agg({"Open": "first",  \
                                                                     "High": "max",    \
                                                                     "Low": "min",     \
                                                                     "Close": "last",   \
                                                                     "Volume": "sum"})
    
    # DOWNLOAD OPTIONS DATA
    @staticmethod
    def download_options_put_call_ratio(ticker: str) -> pd.DataFrame:
        """
        Calcola il rapporto Put/Call (Put/Call Ratio) per un determinato titolo basato sulle opzioni disponibili.
        Questo metodo scarica le opzioni disponibili per il titolo specificato, calcola il volume totale delle call
        e delle put per ogni data di scadenza e determina il rapporto Put/Call per ciascuna data.

        :param ticker: Il ticker del titolo per il quale calcolare il rapporto Put/Call.
        :return:pd.DataFrame: Un DataFrame contenente le date di scadenza come indice e il rapporto Put/Call 
                          (colonna 'Put/Call Ratio') per ciascuna data di scadenza.
                          Se il volume totale delle call è zero, il rapporto sarà impostato a `None`.
        """
        stock = yf.Ticker(ticker)

        # Ottieni le date di scadenza disponibili
        expiration_dates = stock.options
        put_call_ratios = {}
        for exp_date in expiration_dates:
            # Ottieni la chain delle opzioni per la data di scadenza specifica
            option_chain = stock.option_chain(exp_date)
            calls = option_chain.calls
            puts = option_chain.puts

            # Calcola il volume totale di call e put
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()

            # Calcola il rapporto put/call
            if total_call_volume > 0:
                put_call_ratio = total_put_volume / total_call_volume
            else:
                put_call_ratio = None  # Evita divisioni per zero

            put_call_ratios[exp_date] = put_call_ratio

        # Converti in DataFrame per una visualizzazione più semplice
        df_put_call_ratios = pd.DataFrame.from_dict(put_call_ratios, orient='index', columns=['Put/Call Ratio'])

        return put_call_ratios
    
    # SCRAPE FUNDAMENTAL DATA
    @staticmethod
    def scrape_fundamental_data(ticker: str, 
                                url_str_no_isin: str = "http://finviz.com/quote.ashx?t=",
                                metrics: tuple = ('P/B', 'P/E', 'Forward P/E', 'PEG', 'Debt/Eq', 'EPS (ttm)', 
                                                  'Dividend %', 'ROE', 'ROI', 'EPS Q/Q', 'Insider Own'),
                                func: object = lambda webpage, metric: webpage.find('div').find_next(text=metric).find_next(class_='snapshot-td2').text,
                                delay: float = 2.0) -> dict:
        """
        Esegue il web scraping di dati fondamentali di un titolo da Finviz.

        :param ticker: Il simbolo del titolo.
        :param url_str_no_isin: L'URL di base per il scraping (di default punta a Finviz).
        :param metrics: Un tuple di metriche fondamentali da estrarre.
        :param func: Una funzione per estrarre una metrica specifica dal contenuto HTML.
        :param delay: Ritardo (in secondi) tra le richieste per evitare di sovraccaricare il server.
        :return: Un dizionario con le metriche fondamentali richieste.
        """
        
        def scrape_fundamental_metric(soup: BeautifulSoup, metric: str, extraction_func: object) -> str:
            """
            Estrae una metrica fondamentale utilizzando la funzione fornita.
            """
            try:
                return extraction_func(soup, metric)
            except Exception as e:
                print(f"Errore nell'estrazione della metrica '{metric}': {e}")
                return None

        # Costruzione dell'URL completo
        url = f"{url_str_no_isin}{ticker.lower()}"
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            # Pausa per evitare sovraccarico
            print(f"Attesa di {delay} secondi prima di inviare la richiesta per il ticker {ticker}...")
            time.sleep(delay)

            # Richiesta HTTP con un header personalizzato
            req = Request(url, headers=headers)
            webpage = urlopen(req, timeout=10).read()
            soup = BeautifulSoup(webpage, "html.parser")

            # Estrarre tutte le metriche richieste
            results = {metric: scrape_fundamental_metric(soup, metric, func) for metric in metrics}
            
            return results

        except HTTPError as e:
            print(f"HTTPError: {e.code} - {e.reason} per il ticker {ticker}")
        except URLError as e:
            print(f"URLError: {e.reason} per il ticker {ticker}")
        except Exception as e:
            print(f"Errore generico durante la richiesta: {e}")

        return {el: None for el in metrics}
    
    @staticmethod
    def deannualize(annual_rate: float, periods: int = 252) -> float:
        """
        Converte un tasso annuale in un tasso giornaliero.

        :param annual_rate: Il tasso di interesse annuale.
        :param periods: Il numero di periodi in un anno (di default 252, che rappresenta il numero di giorni di trading).
        :return: Il tasso de-annualizzato.
        """
        # DE-ANNUALIZE YEARLY INTERREST RATES
        return (1 + annual_rate) ** (1 / periods) - 1

    @staticmethod
    def download_risk_free_rate_curve(ticker: str, date_range: list[str] = None) -> pd.DataFrame:
        """
        Scarica la curva dei tassi risk-free da Yahoo Finance e calcola i tassi giornalieri de-annualizzati.

        :param ticker: Il simbolo del titolo che rappresenta il tasso risk-free (ad esempio, "^IRX" per il Treasury Bill a 3 mesi).
        :param date_range: Un intervallo opzionale di date per scaricare i dati.
        :return: Un pd.DataFrame con i tassi annualizzati e de-annualizzati.
        """
        all_data = yf.download(ticker, start=date_range[0], end=date_range[1]) if date_range is not None else \
                   yf.download(ticker, period='max')
        annualized = all_data["Close"]
        daily = annualized.apply(FetchSingleStock.deannualize)
        calculated_data = pd.DataFrame({"index": all_data.index, 
                                        "annualized": annualized.values.flatten(), 
                                        "daily": daily.values.flatten()})
        out = pd.concat([all_data, calculated_data.set_index("index")], axis=1)
        out["ticker"] = ticker
        return out[["ticker"] + [c for c in all_data.columns] + ["annualized", "daily"]]


class FetchData(FetchSingleStock, DataObjectManipulation, RiskFreeObjectManipulation):

    def __init__(self, ticker_list: list[str], risk_free_ticker: str = '^IRX', date_range: list[str] = None, bar_time_interval: str = None):
        self.ticker = ticker_list
        self.risk_free_ticker = risk_free_ticker
        self.date_range = date_range
        self.bar_time_interval = bar_time_interval
        self.risk_free_rate_curve = self.resample_history_by_custom_bar_time_interval(
            data=self.download_risk_free_rate_curve(ticker=self.risk_free_ticker, date_range=self.date_range),
            time_interval=self.bar_time_interval)
        self.data = {}

        for isin in self.ticker:
            try:
                ticker_info = yf.Ticker(isin).info
            except Exception as e:
                ticker_info = None

            self.data.update({isin: {"Description": ticker_info.get("longBusinessSummary") if ticker_info is not None else None,
                                     "Industry": ticker_info.get("industry") if ticker_info is not None else None,
                                     "Sector": ticker_info.get("sector") if ticker_info is not None else None,
                                     "Fundamentals": self.scrape_fundamental_data(ticker=isin),
                                     "History": self.resample_history_by_custom_bar_time_interval(
                                         data=self.download_history(ticker=isin, date_range=self.date_range),
                                         time_interval=self.bar_time_interval),
                                     "PutCallRatio": self.download_options_put_call_ratio(ticker=isin)}
                             })
        self.calculate_over_under_valuated()

        # CALL DATA MANIPULATION OBJECT TO USE ITS FUNCTIONS
        DataObjectManipulation.__init__(self, data=self.data)
        RiskFreeObjectManipulation.__init__(self, risk_free_rate_curve=self.risk_free_rate_curve)
    

    def calculate_over_under_valuated(self, price_to_earning_name: str = "P/E"):
        """
        Calcola se un titolo è sottovalutato, sopravvalutato o equamente valutato basandosi sul rapporto P/E rispetto alla media dei titoli.

        :param price_to_earning_name: Il nome della metrica P/E (di default "P/E").
        :return: Aggiorna il dizionario data con le informazioni sul valore relativo del titolo.
        """
        def over_under_cluster(value):
            """
            Classifica il valore relativo del P/E in under_valued, over_valued o fair_valued.
            """
            if value < 1:
                return "under_valued"
            elif value > 1:
                return "over_valued"
            elif value == 1:
                return "fair_valued"
            else:
                return None

        try:
            # Recupera i dati fondamentali
            _ = self.get_all_fundamental()
            
            # Converte i valori della metrica P/E in numerici (NaN per valori non validi)
            _[price_to_earning_name] = pd.to_numeric(_.loc[_.index == price_to_earning_name].values[0], errors='coerce')
            
            # Calcola la media ignorando i NaN
            mean_pe_all_stocks = _[price_to_earning_name].mean()
            
            # Verifica se la media è calcolabile
            if pd.isna(mean_pe_all_stocks) or mean_pe_all_stocks == 0:
                raise ValueError("La media del P/E non è valida (NaN o zero). Controllare i dati.")

            # Calcola i valori relativi e aggiorna i dati
            for isin in self.ticker:
                try:
                    pe_value = float(self.data[isin]["Fundamentals"][price_to_earning_name])
                    over_under_value = pe_value / mean_pe_all_stocks
                    self.data[isin]["Fundamentals"].update({
                        "Over/Under PE Value": over_under_value,
                        "Over/Under PE Cluster": over_under_cluster(over_under_value)
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Errore nel calcolo per il ticker {isin}: {e}")
                    self.data[isin]["Fundamentals"].update({
                        "Over/Under PE Value": None,
                        "Over/Under PE Cluster": None
                    })

        except Exception as e:
            print(f"Errore generale durante il calcolo Over/Under PE: {e}")

    def plot_ticker_data(self, ticker: str, plots: list = None):
        """
        Visualizza i dati per un particolare ticker utilizzando le funzioni di DataVisualization.

        :param ticker: Il simbolo del titolo da visualizzare.
        :param plots: Una lista di tuple con il nome della funzione e i suoi parametri. Esempio: [('OHLC', {}), ('VOLUME', {})]
        :return: Nessun valore di ritorno. Visualizza i grafici richiesti.
        """
        if plots is None:
            plots = [
                ('OHLC', {}),
                ('VOLUME', {})
            ]

        if ticker in self.data:
            stock_data = self.data[ticker]["History"]

            # Controlla che i dati OHLC esistano
            if not stock_data.empty and all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                # Crea un'istanza di DataVisualization e richiama i metodi di visualizzazione
                visualizer = DataVisualizationStock(stock_data)

                # Visualizza i grafici definiti in plots
                visualizer.plot_combined(plots)
            else:
                print(f"Dati insufficienti per il ticker {ticker}")
        else:
            print(f"Ticker {ticker} non trovato nei dati")

    def plot_put_call_ratio(self, ticker: str):
        """
        Visualizza i dati del Put/Call Ratio per uno specifico isin.

        :param ticker: Il simbolo del titolo.
        :return: Nessun valore di ritorno. Visualizza il grafico richiesto.
        """
        df_put_call_ratios = pd.DataFrame.from_dict(self.data[ticker]['PutCallRatio'], orient='index', columns=['Put/Call Ratio'])

        # Creare il grafico
        fig = go.Figure()

        # Aggiungere la linea del put/call ratio
        fig.add_trace(go.Scatter(
            x=df_put_call_ratios.index, 
            y=df_put_call_ratios['Put/Call Ratio'], 
            mode='lines',
            name='Put/Call Ratio',
            line=dict(color='blue')
        ))

        # Linea a y=0.8 
        fig.add_trace(go.Scatter(
            x=[df_put_call_ratios.index.min(), df_put_call_ratios.index.max()],
            y=[0.8, 0.8],
            mode='lines',
            name='Bearish Threshold',
            line=dict(color='red', dash='dash')
        ))

        # Linea a y=0.3 
        fig.add_trace(go.Scatter(
            x=[df_put_call_ratios.index.min(), df_put_call_ratios.index.max()],
            y=[0.3, 0.3],
            mode='lines',
            name='Bullish Threshold',
            line=dict(color='green', dash='dash')
        ))

        # Evidenziare l'area tra 0.4 e 0.5 (grigio chiaro)
        fig.add_trace(go.Scatter(
            x=list(df_put_call_ratios.index) + list(df_put_call_ratios.index)[::-1],
            y=[0.4] * len(df_put_call_ratios) + [0.5] * len(df_put_call_ratios[::-1]),
            fill='toself',
            fillcolor='lightgrey',
            line=dict(color='rgba(255,255,255,0)'),
            name='Neutral Zone',
            opacity=0.5
        ))

        # Configurare il titolo e il layout
        fig.update_layout(
            title=f'Put/Call Ratio - {ticker}',
            xaxis_title='Date',
            yaxis_title='Put/Call Ratio',
            template='plotly_white',
            height=600,
            width=1000
        )

        # Mostrare il grafico
        fig.show()

    # EXPORT FUNCTION
    def export(self, path_out: str, filename: str):
        """
        Esporta i dati del portafoglio e dei titoli in un file pickle.

        :param path_out: Il percorso di output.
        :param filename: Il nome del file pickle.
        :return: Nessun valore di ritorno. Salva i dati su file.
        """
        os.makedirs(path_out, exist_ok=True)

        with open(os.path.join(path_out, filename), 'wb') as obj:
            pickle.dump({'tickers': self.ticker, 'data': self.data,      
                         'risk_free_ticker': self.risk_free_ticker, 'risk_free_rate_curve': self.risk_free_rate_curve}, obj)