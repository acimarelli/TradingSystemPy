import pandas as pd
import itertools
import plotly.graph_objects as go
from trading_system_py.analysis.backtest.backtest import Backtest
from trading_system_py.analysis.portfolio.portfolio import Portfolio


class BacktestMultipleCombinationPortfolio:
    def __init__(self, portfolio, strategy_dict, verbose=False):
        """
        Inizializza il BacktestPortfolio.

        Parametri:
        - portfolio: Portfolio
            Il portafoglio contenente gli asset e il capitale iniziale.
        - strategy_dict: dict
            Un dizionario {ISIN: [lista di strategie]} contenente le strategie per ciascun asset.
        - verbose: bool
            Se True, stampa i dettagli delle transazioni.
        """
        self.portfolio = portfolio
        self.strategy_dict = strategy_dict
        self.verbose = verbose
        self.all_combinations = []  # Lista di tutte le combinazioni di strategie
        self.results = {}  # Risultati del backtest per ciascuna combinazione

    def generate_combinations(self):
        """
        Genera tutte le combinazioni possibili di strategie tra gli ISIN.
        """
        isins = list(self.strategy_dict.keys())
        strategies_list = [self.strategy_dict[isin] for isin in isins]
        combinations = list(itertools.product(*strategies_list))

        # Ogni combinazione è una tupla di strategie, una per ciascun ISIN
        for combo in combinations:
            combo_dict = {isin: strategy for isin, strategy in zip(isins, combo)}
            self.all_combinations.append(combo_dict)

    def run(self):
        """
        Esegue il backtest per tutte le combinazioni di strategie.
        """
        self.generate_combinations()

        for idx, combo in enumerate(self.all_combinations):
            if self.verbose:
                print(f"\nEsecuzione del backtest per la combinazione {idx + 1}/{len(self.all_combinations)}:")
                for isin, strategy in combo.items():
                    print(f" - {isin}: {strategy.__class__.__name__}")

            # Crea una copia del portafoglio per la combinazione corrente
            portfolio_copy = Portfolio(
                data=self.portfolio.data,
                weights=self.portfolio.weights,
                init_cash=self.portfolio.init_cash,
                fee_plus=self.portfolio.fee_plus,
                broker_fee=self.portfolio.broker_fee,
                verbose=self.verbose
            )

            # Inizializza un backtest con la combinazione corrente
            backtest = Backtest(
                portfolio=portfolio_copy,
                strategy_combination=combo,
                verbose=self.verbose
            )

            # Esegui il backtest
            backtest.run()

            # Salva i risultati del backtest
            self.results[idx] = {
                'combination': combo,
                'portfolio_history': backtest.portfolio.portfolio_history,
                'transaction_history': backtest.portfolio.transaction_history,
                'signals_dict': backtest.signals_dict
            }

    def plot_performance(self):
        """
        Traccia il valore storico del portafoglio per tutte le combinazioni.
        """
        fig = go.Figure()

        for idx, result in self.results.items():
            portfolio_history = result['portfolio_history']
            dates = [record['date'] for record in portfolio_history]
            total_values = [record['total_value'] for record in portfolio_history]

            combo_str = ', '.join(
                [f"{isin}: {strategy.__class__.__name__}" for isin, strategy in result['combination'].items()]
            )

            fig.add_trace(go.Scatter(
                x=dates,
                y=total_values,
                mode='lines',
                name=f'Combo {idx + 1}' #f'Combo {idx + 1}: {combo_str}'
            ))

        fig.update_layout(
            title='Andamento del Portafoglio per Tutte le Combinazioni',
            xaxis_title='Data',
            yaxis_title='Valore (€)',
            hovermode='x unified'
        )

        fig.show()

    def plot_signals(self, isin):
        """
        Traccia i segnali per un ISIN specifico, differenziando le strategie.

        Parametri:
        - isin: str
            Il codice ISIN dell'asset da plottare.
        """
        data = self.portfolio.data[isin]['History']

        fig = go.Figure()

        # Plotta il prezzo dell'azione
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close']/data['Close'][0],
            mode='lines',
            name=f'Prezzo Azione {isin}'
        ))

        for idx, result in self.results.items():
            strategy = result['combination'].get(isin)
            if not strategy:
                continue  # Se l'ISIN non è presente in questa combinazione, salta

            signals = result['signals_dict'][isin]
            combo_str = strategy.__class__.__name__

            # Ottieni il valore della posizione del portafoglio per l'ISIN
            position_values = pd.Series([record['positions'].get(isin, 0) for record in result['portfolio_history']],
                                        index=[record['date'] for record in result['portfolio_history']]
                                       ).replace(0, None).ffill().bfill()
            
            # Plotta il valore della posizione del portafoglio per l'ISIN
            fig.add_trace(go.Scatter(
                x=position_values.index,
                y=position_values/position_values[0],
                mode='lines',
                name=f'{combo_str} {isin}'
            ))

            # Plotta i segnali
            buy_signals = signals[signals['positions'] == 1.0]
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=data.loc[buy_signals.index]['Close']/data['Close'][0],
                mode='markers',
                marker_symbol='triangle-up',
                marker_size=10,
                name=f'{combo_str} - Buy'
            ))

            sell_signals = signals[signals['positions'] == -1.0]
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=data.loc[sell_signals.index]['Close']/data['Close'][0],
                mode='markers',
                marker_symbol='triangle-down',
                marker_size=10,
                name=f'{combo_str} - Sell'
            ))

        fig.update_layout(
            title=f'Segnali per {isin} con Diverse Strategie',
            xaxis_title='Data',
            yaxis_title='Prezzo',
            hovermode='x unified'
        )

        fig.show()
