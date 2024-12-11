import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce
from src.analysis.portfolio.portfolio import Portfolio


class Backtest:
    def __init__(self, portfolio, strategy_combination, allow_short_selling=False, verbose=False):
        """
        Inizializza il Backtest per una combinazione di strategie su un portafoglio.

        Parametri:
        - portfolio: Portfolio
            Il portafoglio contenente gli asset e il capitale iniziale.
        - strategy_combination: dict
            Un dizionario {ISIN: strategia} contenente la strategia per ciascun asset.
        - allow_short_selling: bool
            Se True, consente la vendita allo scoperto durante il backtest.
        - verbose: bool
            Se True, stampa i dettagli delle transazioni.
        """
        self.portfolio = portfolio
        self.portfolio.verbose = verbose  # Imposta il verbose tramite il setter
        self.strategy_combination = strategy_combination
        self.allow_short_selling = allow_short_selling
        self.verbose = verbose
        self.signals_dict = {}  # {ISIN: DataFrame con i segnali}
        self.simulations = {}
        self.simulation_results = {}  # Per salvare i risultati delle simulazioni (es. valore portafoglio)

    def run(self, make_simulation: bool = False, num_simulation: int = 1000):
        """
        Esegue il backtest per ogni combinazione di strategie.
        
        Parametri:
        - make_simulation: bool
            Se True, esegue N simulazioni per ogni ISIN.
        - num_simulation: int
            Numero di simulazioni da eseguire (solo se make_simulation è True).
        """
        if make_simulation:
            # Eseguo il backtest per il dato originale passato in init
            backtest_orig = Backtest(portfolio=self.portfolio, strategy_combination=self.strategy_combination)
            backtest_orig.run(make_simulation=False)
            self.signals_dict = backtest_orig.signals_dict

            # Esegui le simulazioni e il backtest su ciascuna
            self.generate_simulations(N=num_simulation, T=len(self.get_all_dates()))
            for isin in self.portfolio.assets:
                self.simulations[isin].set_index(self.get_all_dates(), inplace=True)

            # Definisco il dato della i-sima simulazione da backtestare
            for idx in range(num_simulation):
                sim_data_object = {}
                for isin in self.portfolio.assets:
                    sim_data = self.simulations[isin][f'Simulation_{idx}']
                    simulated_history = pd.DataFrame({'Close': sim_data}).set_index(self.get_all_dates())
                    sim_data_object.update({isin: {'History': simulated_history}})
                
                # Crea una copia del portafoglio per la combinazione corrente
                portfolio_copy = Portfolio(
                    data=sim_data_object,
                    weights=self.portfolio.weights,
                    init_cash=self.portfolio.init_cash,
                    fee_plus=self.portfolio.fee_plus,
                    broker_fee=self.portfolio.broker_fee,
                    verbose=self.verbose
                )

                # Esegui il backtest per la i-sima simulazione 
                backtest = Backtest(portfolio=portfolio_copy, strategy_combination=self.strategy_combination)
                backtest.run(make_simulation=False, num_simulation=num_simulation)  # Esegui il backtest sul singolo set di dati simulati

                # Estrae date e valori totali del portafoglio relativo alla i-sima simulazione
                dt = [record['date'] for record in backtest.portfolio.portfolio_history]
                simulated_ptf_value = [record['total_value'] for record in backtest.portfolio.portfolio_history]

                # Salva i risultati del backtest della i-sima simulazione
                self.simulation_results[idx] = {
                    'portfolio_value': pd.Series(simulated_ptf_value, index=dt, name=f'Portfolio_Simulation_{idx}'),
                    'portfolio_history': backtest.portfolio.portfolio_history,
                    'transaction_history': backtest.portfolio.transaction_history,
                    'signals_dict': backtest.signals_dict
                }
        else:
            # Backtest standard senza simulazioni
            dates = self.get_all_dates()

            # Genera i segnali per ciascun ISIN
            for isin, strategy in self.strategy_combination.items():
                data = self.portfolio.data[isin]['History']
                signals = strategy.generate_signals(data)
                self.signals_dict[isin] = signals

            # Esegui il backtest
            for date in dates:
                for isin, strategy in self.strategy_combination.items():
                    signals = self.signals_dict[isin]
                    if date not in signals.index:
                        continue

                    position = signals.loc[date, 'positions']
                    price = self.portfolio.get_price(isin, date)

                    # Verifica lo stop loss e il take profit per le posizioni long
                    if isin in self.portfolio.positions:
                        position_info = self.portfolio.positions[isin]
                        
                        # Controlla stop loss e take profit solo se definiti
                        if position_info['shares'] > 0:
                            # Verifica se lo stop loss è definito e applicabile
                            if position_info['stop_loss_price'] is not None and price <= position_info['stop_loss_price']:
                                shares_to_sell = position_info['shares']
                                self.portfolio.sell(date, isin, price, shares_to_sell)
                                continue
                            
                            # Verifica se il take profit è definito e applicabile
                            elif position_info['take_profit_price'] is not None and price >= position_info['take_profit_price']:
                                shares_to_sell = position_info['shares']
                                self.portfolio.sell(date, isin, price, shares_to_sell)
                                continue

                        # Verifica lo stop loss e il take profit per le posizioni short
                        elif position_info['shares'] < 0:
                            # Verifica se lo stop loss è definito e applicabile per short
                            if position_info['stop_loss_price'] is not None and price >= position_info['stop_loss_price']:
                                shares_to_cover = abs(position_info['shares'])
                                self.portfolio.cover_short(date, isin, price, shares_to_cover)
                                continue

                            # Verifica se il take profit è definito e applicabile per short
                            elif position_info['take_profit_price'] is not None and price <= position_info['take_profit_price']:
                                shares_to_cover = abs(position_info['shares'])
                                self.portfolio.cover_short(date, isin, price, shares_to_cover)
                                continue

                    # Gestisci nuovi acquisti e short in base ai segnali
                    if position == 1.0:
                        # Gestisci posizioni long e short
                        if isin in self.portfolio.positions and self.portfolio.positions[isin]['shares'] < 0:
                            shares_to_cover = abs(self.portfolio.positions[isin]['shares'])
                            self.portfolio.cover_short(date, isin, price, shares_to_cover)
                        else:
                            target_cash = self.portfolio.cash_allocation[isin]
                            invested_shares = self.portfolio.positions.get(isin, {}).get('shares', 0)
                            invested_cash = invested_shares * price
                            investment_needed = target_cash - invested_cash
                            investment = min(investment_needed, self.portfolio.cash)
                            shares_to_buy = int(investment / price)
                            if shares_to_buy > 0:
                                self.portfolio.buy(date, isin, price, shares_to_buy)

                    elif position == -1.0:
                        # Gestisci le vendite e posizioni short
                        if isin in self.portfolio.positions and self.portfolio.positions[isin]['shares'] > 0:
                            shares_to_sell = abs(self.portfolio.positions[isin]['shares'])
                            self.portfolio.sell(date, isin, price, shares_to_sell)

                        # Usa il target_cash per shortare, non tutto il cash del portafoglio
                        target_cash = self.portfolio.cash_allocation[isin]
                        invested_shares = self.portfolio.positions.get(isin, {}).get('shares', 0)
                        invested_cash = invested_shares * price
                        shorting_needed = target_cash - invested_cash  # Quantità di cash che si può usare per shortare
                        shares_to_short = int(shorting_needed / price)  # Numero di azioni da shortare

                        # Esegui lo short solo se è permesso e ci sono azioni da shortare
                        if self.allow_short_selling and shares_to_short > 0:
                            self.portfolio.short_sell(date, isin, price, shares_to_short)

                self.portfolio.update_portfolio_value(date)

            # Riporta il valore self.portfolio.cash_allocation[isin] al suo valore iniziale
            for isin, strategy in self.strategy_combination.items():
                self.portfolio.update_target_cash(isin=isin, pnl=None)

    def get_all_dates(self):
        """
        Ottiene un insieme combinato di tutte le date dai dati di ciascun ISIN.
        """
        date_sets = []
        for isin in self.strategy_combination.keys():
            data = self.portfolio.data[isin]['History']
            date_sets.append(data.index)
        all_dates = reduce(lambda x, y: x.union(y), date_sets)
        all_dates = all_dates.sort_values()
        return all_dates

    def generate_simulations(self, N=1000, T=252, arima_order=(1, 0, 1)):
        """
        Genera N simulazioni per ogni ISIN nel portafoglio utilizzando un modello ARIMA + GARCH.
        
        Parametri:
        - N: int
            Numero di simulazioni da eseguire.
        - T: int
            Numero di giorni da simulare.
        - arima_order: tuple
            Ordine del modello ARIMA (p, d, q).
        """
        for isin, data in self.portfolio.data.items():
            history = data['History']
            log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()

            # Fitta il modello ARIMA sui rendimenti logaritmici
            arima_model = ARIMA(log_returns, order=arima_order)
            arima_fit = arima_model.fit()

            # Estrai i residui dall'ARIMA per passare al GARCH
            residuals = arima_fit.resid

            # Fitta il modello GARCH sui residui dell'ARIMA
            garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp="off")

            # Estrai i parametri stimati
            omega = garch_fit.params['omega']
            alpha = garch_fit.params['alpha[1]']
            beta = garch_fit.params['beta[1]']

            # Inizializza la serie simulata con il prezzo finale della serie storica
            S0 = history['Close'].iloc[-1]
            simulated_prices = pd.DataFrame(index=range(T), columns=[f'Simulation_{i}' for i in range(N)])

            # Simulazione della volatilità condizionata e dei rendimenti
            for i in range(N):
                sigma2 = np.zeros(T)  # Varianza condizionata
                returns_sim = np.zeros(T)
                
                # Rumore bianco standard per simulare rendimenti
                shocks = np.random.randn(T)

                # Setta la varianza iniziale come la varianza dei residui
                sigma2[0] = residuals.var()

                # Genera la varianza condizionata per ogni passo temporale
                for t in range(1, T):
                    sigma2[t] = omega + alpha * (returns_sim[t-1]**2) + beta * sigma2[t-1]
                    returns_sim[t] = shocks[t] * np.sqrt(sigma2[t])

                # Somma la componente ARIMA e la componente GARCH
                arima_forecast = arima_fit.forecast(steps=T)

                # Se arima_forecast è un singolo valore scalare, convertiamolo in un array
                if isinstance(arima_forecast, (float, np.float64)):
                    arima_forecast = np.full(T, arima_forecast)
                else:
                    arima_forecast = arima_forecast.values

                combined_returns = arima_forecast + returns_sim

                # Converti i rendimenti simulati in prezzi
                simulated_prices[f'Simulation_{i}'] = S0 * np.exp(np.cumsum(combined_returns))
            
            # Salva il risultato in self.simulations[isin]
            self.simulations[isin] = simulated_prices

    def plot_performance(self, isins: str=None, benchmark: pd.DataFrame=None, normalize: bool=False, add_drawdown: bool=True):
        """
        Traccia il valore storico del portafoglio e dei valori delle posizioni individuali utilizzando Plotly.
        """
        self.portfolio.plot_performance(isins=isins, benchmark=benchmark, normalize=normalize, add_drawdown=add_drawdown)

    def plot_signals(self, isin):
        """
        Traccia i prezzi delle azioni, i segnali di trading e il drawdown per un singolo asset.

        Parametri:
        - isin: str
            Il codice ISIN dell'asset da plottare.
        """
        data = self.portfolio.data[isin]['History']
        data.index = self.get_all_dates()
        signals = self.signals_dict[isin]

        # Ottieni il valore della posizione del portafoglio per l'ISIN
        position_values = pd.Series(
            [record['positions'].get(isin, 0) for record in self.portfolio.portfolio_history],
            index=[record['date'] for record in self.portfolio.portfolio_history]
        ).replace(0, None).ffill().bfill()

        # Calcola il drawdown specifico per l'ISIN
        cumulative_max = position_values.cummax()
        drawdowns = (position_values - cumulative_max) / cumulative_max
        drawdown_df = pd.DataFrame({'Drawdown': drawdowns})

        # Crea la figura
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=(f'Prezzo e Segnali di Trading per {isin}', 'Drawdown')
        )

        # Plotta il prezzo dell'azione
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close']/data['Close'][0],
            mode='lines',
            name=f'Prezzo Azione {isin}'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=position_values/position_values[0],
            mode='lines',
            name=f'Strategia Azione {isin}'
        ))

        # Plotta i segnali di acquisto
        buy_signals = signals[signals['positions'] == 1.0]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=data.loc[buy_signals.index]['Close']/data['Close'][0],
            mode='markers',
            marker_symbol='triangle-up',
            marker_color='green',
            marker_size=10,
            name='Segnale di Acquisto'
        ), row=1, col=1)

        # Plotta i segnali di vendita
        sell_signals = signals[signals['positions'] == -1.0]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=data.loc[sell_signals.index]['Close']/data['Close'][0],
            mode='markers',
            marker_symbol='triangle-down',
            marker_color='red',
            marker_size=10,
            name='Segnale di Vendita'
        ), row=1, col=1)

        # Plotta il drawdown per l'ISIN
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Drawdown'],
            fill='tozeroy',
            mode='lines',
            name=f'Drawdown {isin}',
            line=dict(color='red')
        ), row=2, col=1)

        fig.update_layout(
            title=f'Segnali di Trading e Drawdown per {isin}',
            xaxis_title='Data',
            yaxis_title='Prezzo',
            hovermode='x unified',
            height=450*2
        )

        fig.show()

    def plot_simulated_isin_trajectory(self, isin):
        """
        Plotta i risultati delle simulazioni per un ISIN, evidenziando la traiettoria attesa in rosso.
        
        Parametri:
        - isin: str
            Il codice ISIN dell'asset da plottare.
        """
        fig = go.Figure()

        # Plotta tutte le simulazioni con lo stesso colore e trasparenza
        for col in self.simulations[isin].columns:
            fig.add_trace(go.Scatter(
                x=self.simulations[isin].index,
                y=self.simulations[isin][col],
                mode='lines',
                line=dict(color='blue', width=1, dash='solid'),
                opacity=0.3,
                name=f'{col}'
            ))

        # Evidenzia la traiettoria attesa (la media)
        expected_trajectory = self.simulations[isin].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=expected_trajectory.index,
            y=expected_trajectory,
            mode='lines',
            line=dict(color='red', width=2),
            name='Expected Trajectory'
        ))

        fig.update_layout(
            title=f'Performance Simulata per {isin}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        fig.show()

    def plot_simulated_portfolio_results(self):
        """
        Plotta i risultati dei diversi portafogli simulati.
        """
        fig = go.Figure()

        # Lista per raccogliere tutte le serie per il calcolo della media
        all_portfolio_values = []
        
        # Itera su tutti gli elementi del dizionario
        for idx, sim_data in self.simulation_results.items():
            portfolio_value = sim_data['portfolio_value']
            all_portfolio_values.append(portfolio_value)
            
            # Plotta ogni serie simulata in grigio semitrasparente
            fig.add_trace(go.Scatter(x=portfolio_value.index, 
                                    y=portfolio_value.values, 
                                    mode='lines', 
                                    line=dict(color='gray', width=1, dash='solid'),
                                    opacity=0.5,
                                    name=f'Simulation {idx}'))

        # Calcola la media in ogni punto temporale
        portfolio_df = pd.concat(all_portfolio_values, axis=1)
        mean_portfolio_value = portfolio_df.mean(axis=1)
        
        # Plotta la traiettoria attesa (media) in rosso
        fig.add_trace(go.Scatter(x=mean_portfolio_value.index, 
                                y=mean_portfolio_value.values, 
                                mode='lines', 
                                line=dict(color='red', width=2),
                                name='Expected Portfolio (Mean)'))

        # Aggiungi i titoli e label
        fig.update_layout(
            title='Portfolio Simulations with Expected Trajectory (Mean)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            showlegend=True
        )
        
        # Mostra il grafico
        fig.show()

    def plot_weights_portfolio_over_time(self, normalize: bool=False):
        df = pd.DataFrame(pd.concat([pd.DataFrame(self.portfolio.portfolio_history)['positions'].apply(pd.Series),
                                     pd.DataFrame(self.portfolio.portfolio_history)['cash'],
                                     pd.DataFrame(self.portfolio.portfolio_history)['date']], axis=1))
        df.set_index('date', inplace=True)

        # Calcoliamo la somma del portafoglio per ogni riga
        df['total_portfolio'] = df.sum(axis=1)

        # Creiamo il grafico con Plotly
        fig = go.Figure()

        # Aggiungi le barre per l'i-simo isin
        for id, el in enumerate(df.columns):
            if el != 'total_portfolio':
                fig.add_trace(go.Bar(x=df.index, y=df[el]/df['total_portfolio'] if normalize else df[el], name=el, offsetgroup=id))

        # Aggiungi la traccia per il valore totale del portafoglio
        if not normalize:
            fig.add_trace(go.Scatter(x=df.index, y=df['total_portfolio'], mode='lines', name='Total Portfolio')) #, line=dict(dash='dash')

        # Configuriamo il layout del grafico
        fig.update_layout(barmode='stack',  # Modalità di visualizzazione a barre non stacked
                          title='Valore storico del portafoglio (ISIN e Liquidità)',
                          xaxis_title='Index',
                          yaxis_title='Valore',
                          hovermode='x unified')

        # Mostriamo il grafico
        fig.show()
 