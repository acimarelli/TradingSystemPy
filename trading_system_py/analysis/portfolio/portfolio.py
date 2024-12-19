import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Portfolio:
    def __init__(self, data, weights=None, init_cash=100000, fee_plus=0.26, broker_fee=0, stop_loss=None, take_profit=None, verbose=False):
        """
        Inizializza il Portafoglio.

        Parametri:
        - data: dict o pd.DataFrame
            Se dict: {ISIN: {'History': pd.DataFrame, ...}, ...}
            Se pd.DataFrame: DataFrame come scaricato da yfinance
        - weights: dict {ISIN: float, ...}, opzionale
            Pesi degli asset. Se non forniti e data è dict, applica equal weight.
        - init_cash: float
            Importo di cassa iniziale.
        - fee_plus: float
            Tasso di tassazione sulle plusvalenze realizzate (es. 0.26 per 26%).
        - broker_fee: float
            Commissione applicata a ogni transazione.
        - stop_loss: float
            Percentuale di stop loss.
        - take_profit: float
            Percentuale di take profit.
        - verbose: bool
            Se True, stampa informazioni dettagliate.
        """
        self._verbose = verbose
        self.init_cash = init_cash
        self.cash = init_cash
        self.fee_plus = fee_plus
        self.broker_fee = broker_fee
        self.stop_loss = stop_loss  # Percentuale di stop loss (es. 0.05 per 5%)
        self.take_profit = take_profit  # Percentuale di take profit (es. 0.1 per 10%)
        self.positions = {}  # {ISIN: {'shares': int, 'cost_basis': float}}
        self.portfolio_history = []  # Storico del valore del portafoglio
        self.transaction_history = []  # Storico delle transazioni

        # Processa data e weights
        if isinstance(data, dict):
            self.data = data
            self.assets = list(data.keys())
            if not weights:
                n_assets = len(self.assets)
                self.weights = {asset: 1 / n_assets for asset in self.assets}
            else:
                self.weights = weights
        elif isinstance(data, pd.DataFrame):
            # Assume che data sia per un singolo asset
            self.data = {'Asset': {'History': data}}
            self.assets = ['Asset']
            if not weights:
                self.weights = {'Asset': 1.0}
            else:
                self.weights = weights
        else:
            raise ValueError("Data deve essere un dict o un pandas DataFrame.")

        # Calcola il cash target per ogni asset in base ai pesi
        self.cash_allocation = {isin: self.init_cash * weight for isin, weight in self.weights.items()}

        if self._verbose:
            print(f"Cash allocation per asset: {self.cash_allocation}")
    
    @property
    def verbose(self):
        """
        Getter per l'attributo verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """
        Setter per l'attributo verbose.
        """
        # Aggiorna l'attributo privato senza ricorsione
        self._verbose = value
    
    def buy(self, date, isin, price, shares):
        """
        Esegue una transazione di acquisto e aggiorna il target_cash in base al profitto/perdita.
        """
        cost = price * shares + self.broker_fee
        if self.cash >= cost:
            self.cash -= cost
            if isin in self.positions:
                position = self.positions[isin]
                total_shares = position['shares'] + shares
                total_cost = position['cost_basis'] * position['shares'] + price * shares
                position['shares'] = total_shares
                position['cost_basis'] = total_cost / total_shares
            else:
                # Imposta stop loss e take profit solo se definiti
                stop_loss_price = price * (1 - self.stop_loss) if self.stop_loss is not None else None
                take_profit_price = price * (1 + self.take_profit) if self.take_profit is not None else None
                
                self.positions[isin] = {
                    'shares': shares, 
                    'cost_basis': price,
                    'stop_loss_price': stop_loss_price,  # Valore di stop loss
                    'take_profit_price': take_profit_price  # Valore di take profit
                }
            
            # Registra la transazione
            self.transaction_history.append({
                'date': date,
                'type': 'BUY',
                'isin': isin,
                'price': price,
                'shares': shares,
                'cash': self.cash
            })
            if self._verbose:
                print(f"{date}: BUY {shares} shares of {isin} at {price}, cash remaining: {self.cash}")
        else:
            if self._verbose:
                print(f"Fondi insufficienti per acquistare {shares} azioni di {isin} il {date}.")

    def sell(self, date, isin, price, shares):
        """
        Esegue una transazione di vendita e verifica le condizioni di stop loss e take profit.
        """
        if isin in self.positions and self.positions[isin]['shares'] >= shares:
            position = self.positions[isin]
            
            # Verifica stop loss solo se definito
            if self.stop_loss is not None and price <= position['stop_loss_price']:
                if self._verbose:
                    print(f"{date}: Stop loss triggered for {isin} at {price}. Selling {shares} shares.")
            
            # Verifica take profit solo se definito
            elif self.take_profit is not None and price >= position['take_profit_price']:
                if self._verbose:
                    print(f"{date}: Take profit triggered for {isin} at {price}. Selling {shares} shares.")

            # Prosegui con la vendita
            proceeds = price * shares 
            cost_basis = position['cost_basis']
            profit = (price - cost_basis) * shares
            tax = self.fee_plus * max(profit, 0)
            self.cash += proceeds - tax
            position['shares'] -= shares
            if position['shares'] == 0:
                del self.positions[isin]

            # Registra la transazione
            self.transaction_history.append({
                'date': date,
                'type': 'SELL',
                'isin': isin,
                'price': price,
                'shares': shares,
                'cash': self.cash,
                'profit': profit,
                'tax': tax
            })
            if self._verbose:
                print(f"{date}: SELL {shares} shares of {isin} at {price}, cash after sale: {self.cash}")
            
            # Aggiorna il target_cash con il valore attuale dell'asset
            self.update_target_cash(isin, profit - tax)
        else:
            if self._verbose:
                print(f"Azioni insufficienti per vendere {shares} azioni di {isin} il {date}.")
    
    def short_sell(self, date, isin, price, shares, margin_requirement=1.0):
        """
        Esegue una transazione di vendita allo scoperto e aggiorna il margine di sicurezza richiesto.
        """
        # Calcola il valore delle azioni shortate e il margine richiesto
        short_value = price * shares
        margin = short_value * margin_requirement + self.broker_fee
        
        # Verifica se c'è abbastanza liquidità per coprire il margine
        if self.cash < margin:
            if self._verbose:
                print(f"{date}: Fondi insufficienti per garantire il margine per shortare {shares} azioni di {isin}.")
            return  # Non eseguire l'operazione se non c'è abbastanza liquidità per il margine

        # Prosegui con la vendita allo scoperto
        if isin in self.positions:
            position = self.positions[isin]
            
            # Aggiorna la posizione short esistente
            total_shares = position['shares'] - shares
            total_cost = position['cost_basis'] * abs(position['shares']) + short_value
            position['shares'] = total_shares
            position['cost_basis'] = total_cost / abs(total_shares)  # il costo medio ponderato
        else:
            # Nuova posizione short
            stop_loss_price = price * (1 + self.stop_loss) if self.stop_loss is not None else None
            take_profit_price = price * (1 - self.take_profit) if self.take_profit is not None else None
            
            self.positions[isin] = {
                'shares': -shares,  # Numero di azioni shortate
                'cost_basis': price,  
                'stop_loss_price': stop_loss_price,  
                'take_profit_price': take_profit_price
            }

        # Riduci la liquidità per coprire il margine
        self.cash -= margin

        # Registra la transazione
        self.transaction_history.append({
            'date': date,
            'type': 'SHORT_SELL',
            'isin': isin,
            'price': price,
            'shares': -shares,
            'cash': self.cash
        })
        
        if self._verbose:
            print(f"{date}: SHORT SELL {shares} shares of {isin} at {price}, liquidità residua dopo il margine: {self.cash}")

    def cover_short(self, date, isin, price, shares, margin_requirement=1.0):
        """
        Copre una posizione di vendita allo scoperto e verifica le condizioni di stop loss e take profit.
        """
        if isin in self.positions and self.positions[isin]['shares'] < 0:
            position = self.positions[isin]
            
            # Verifica stop loss solo se definito
            if self.stop_loss is not None and price >= position['stop_loss_price']:
                if self._verbose:
                    print(f"{date}: Stop loss triggered for short {isin} at {price}. Covering {shares} shares.")
            
            # Verifica take profit solo se definito
            elif self.take_profit is not None and price <= position['take_profit_price']:
                if self._verbose:
                    print(f"{date}: Take profit triggered for short {isin} at {price}. Covering {shares} shares.")

            # Calcola il valore delle azioni da coprire e il margine associato
            cover_value = position['cost_basis'] * shares #price * shares
            margin = cover_value * margin_requirement
            
            # Prosegui con la copertura
            cost_basis = position['cost_basis']
            profit = (cost_basis - price) * shares  # Profitto: differenza tra prezzo iniziale e prezzo attuale
            tax = self.fee_plus * max(profit, 0)
            
            # Restituzione del margine e aggiornamento liquidità
            self.cash += margin+profit  # Restituisce il margine depositato inizialmente incrementato del profitto generato
            self.cash -= tax  # Applica le tasse sulle plusvalenze

            # Aggiorna la posizione
            position['shares'] += shares
            if position['shares'] == 0:
                del self.positions[isin]  # Elimina la posizione se completamente coperta

            # Registra la transazione
            self.transaction_history.append({
                'date': date,
                'type': 'COVER_SHORT',
                'isin': isin,
                'price': price,
                'shares': shares,
                'cash': self.cash,
                'profit': profit,
                'tax': tax
            })
            if self._verbose:
                print(f"{date}: COVER SHORT {shares} shares of {isin} at {price}, cash after cover: {self.cash}")
            
            # Aggiorna il target_cash con il valore attuale dell'asset
            self.update_target_cash(isin, profit - tax)
        else:
            if self._verbose:
                print(f"Nessuna posizione short da coprire per {isin} il {date}.")
    
    def update_target_cash(self, isin: str, pnl: float = None):
        """
        Aggiorna il target_cash per un asset in base al valore attuale e alle performance.
        """
        # Se ci sono azioni, aggiorna l'allocazione target in base al valore attuale del portafoglio
        if pnl is not None:
            self.cash_allocation[isin] += pnl
        else:
            # Se non ci sono posizioni, manteniamo il valore originario o lo resettiamo
            self.cash_allocation[isin] = self.init_cash * self.weights[isin]
        
        if self._verbose:
            print(f"Updated target cash allocation for {isin}: {self.cash_allocation[isin]}")

    # def update_portfolio_value(self, date):
    #     """
    #     Aggiorna il valore totale del portafoglio alla data specificata.
    #     """
    #     total_value = self.cash
    #     positions_value = {}
    #     for isin in self.assets:
    #         shares = abs(self.positions.get(isin, {}).get('shares', 0))
    #         price = self.get_price(isin, date)
    #         value = shares * price
    #         total_value += value
    #         positions_value[isin] = value
    #     self.portfolio_history.append({
    #         'date': date,
    #         'total_value': total_value,
    #         'cash': self.cash,
    #         'positions': positions_value.copy()
    #     })
    def update_portfolio_value(self, date):
        """
        Aggiorna il valore totale del portafoglio alla data specificata, tenendo conto delle posizioni long e short.
        """
        total_value = self.cash  # Inizializza con il valore della cassa
        positions_value = {}

        for isin in self.assets:
            shares = self.positions.get(isin, {}).get('shares', 0)
            price = self.get_price(isin, date)
            
            # Calcolo del valore della posizione
            if shares > 0:
                # Posizione long
                value = shares * price
            elif shares < 0:
                # Posizione short, quindi il valore aumenta se il prezzo scende
                value = -shares * (self.positions[isin]['cost_basis'] - price) + self.positions[isin]['cost_basis'] * abs(shares)
            else:
                value = 0

            # Aggiorna il valore totale e quello delle posizioni
            total_value += value
            positions_value[isin] = value

        # Aggiungi le informazioni aggiornate a portfolio_history
        self.portfolio_history.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions': positions_value.copy()
        })

        if self._verbose:
            print(f"{date}: Portfolio value updated to {total_value}")

    def get_price(self, isin, date):
        """
        Recupera il prezzo di un asset alla data specificata.
        """
        history = self.data[isin]['History']
        if date in history.index:
            return history.loc[date]['Close']
        else:
            previous_dates = history.index[history.index <= date]
            if len(previous_dates) > 0:
                prev_date = previous_dates[-1]
                return history.loc[prev_date]['Close']
            else:
                if self._verbose:
                    print(f"Nessun dato prezzo disponibile per {isin} al {date}.")
                return np.nan

    def get_portfolio_value(self):
        """
        Ottiene il valore totale corrente del portafoglio.
        """
        if self.portfolio_history:
            return self.portfolio_history[-1]['total_value']
        else:
            return self.init_cash

    def calculate_drawdowns(self):
        """
        Calcola il drawdown a ogni istante del portafoglio.
        """
        portfolio_values = pd.Series(
            [record['total_value'] for record in self.portfolio_history],
            index=[record['date'] for record in self.portfolio_history]
        )
        cumulative_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        drawdown_df = pd.DataFrame({'Drawdown': drawdowns})
        return drawdown_df

    def calculate_max_drawdown(self):
        """
        Calcola il massimo drawdown del portafoglio.
        """
        drawdown_df = self.calculate_drawdowns()
        max_drawdown = drawdown_df['Drawdown'].min()
        return abs(max_drawdown)

    def plot_drawdown(self):
        """
        Traccia il grafico del drawdown nel tempo utilizzando Plotly.
        """
        drawdown_df = self.calculate_drawdowns()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Drawdown'],
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='Portfolio Drawdown Over Time',
            xaxis_title='Data',
            yaxis_title='Drawdown',
            yaxis=dict(tickformat='.2%'),
            hovermode='x unified'
        )

        fig.show()

    def plot_performance(self, isins: str=None, benchmark: pd.DataFrame=None, normalize: bool=False, add_drawdown: bool=True):
        """
        Traccia il valore storico del portafoglio e dei valori delle posizioni individuali utilizzando Plotly.
        """
        # Estrae date e valori totali del portafoglio
        dates = [record['date'] for record in self.portfolio_history]
        total_values = [record['total_value'] for record in self.portfolio_history]

        # Inizializza un DataFrame per i valori delle posizioni
        positions_df = pd.DataFrame(index=dates)

        # Estrae i valori delle posizioni per ogni data
        for record in self.portfolio_history:
            date = record['date']
            positions = record['positions']  # Dizionario {ISIN: valore}
            for isin, value in positions.items():
                positions_df.loc[date, isin] = value

        # Riempie i valori mancanti
        positions_df = positions_df
        positions_df = positions_df.reindex(dates)

        # Seleziona gli ISIN da plottare
        if isins is None:
            isins = positions_df.columns.tolist()
        else:
            positions_df = positions_df[isins]


        num_plots = 1
        if add_drawdown:
            num_plots += 1
            drawdown_df = self.calculate_drawdowns()

        # Creiamo una figura con subplot
        fig = make_subplots(
            rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05
        )

        # Plotta il valore totale del portafoglio
        fig.add_trace(go.Scatter(
            x=dates,
            y=[x/self.init_cash if normalize else x for x in total_values],
            mode='lines',
            name='Valore Totale Portafoglio',
            line=dict(width=3)
        ), row=1, col=1)

        # Plotta i valori delle posizioni individuali
        # for isin in isins:
        #    first_position_price = positions_df[isin].replace(0,None).dropna()[0]
        #    fig.add_trace(go.Scatter(
        #        x=positions_df.index,
        #        y=[x/first_position_price if normalize and x is not None 
        #           else None if normalize and x is None 
        #           else x for x in positions_df[isin].replace(0,None).ffill().bfill()],
        #        mode='lines',
        #        name=f'Valore Posizione {isin}'
        #    ), row=1, col=1)
        
        # Plotta il max drawdown se selezioanto dall'utente
        if add_drawdown:
            fig.add_trace(go.Scatter(
                x=drawdown_df.index,
                y=drawdown_df['Drawdown'],
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ), row=2, col=1)

        # Opzionalmente, plotta il benchmark
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark['Close'],
                mode='lines',
                name='Benchmark',
                line=dict(dash='dash')
            ), row=1, col=1)

        fig.update_layout(
            title='Andamento del Portafoglio e Valori delle Posizioni',
            xaxis_title='Data',
            yaxis_title='Valore (€)',
            hovermode='x unified',
            height=450*num_plots
        )

        fig.show()


