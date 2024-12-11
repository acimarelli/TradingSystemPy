import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px

from src.utils.generic_calculation import GenericCalculation


class EfficientFrontierCalculation:

    # RANDOM GENERATED PORTFOLIOS
    @staticmethod
    def random_portfolios(num_portfolios: int, num_asset: int, mean_returns: np.array,
                          cov_matrix: np.matrix, risk_free_rate: float):
        """
        Genera portafogli casuali basati sui rendimenti medi e sulla matrice di covarianza.

        :param num_portfolios: Numero di portafogli da generare.
        :param num_asset: Numero di asset nel portafoglio.
        :param mean_returns: Un array dei rendimenti medi per ciascun asset.
        :param cov_matrix: Una matrice di covarianza per gli asset.
        :param risk_free_rate: Il tasso di interesse privo di rischio per il calcolo del rapporto di Sharpe.
        :return: Una tuple contenente i risultati (volatilità, rendimenti, Sharpe ratio) e i pesi del portafoglio.
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(num_asset)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = \
                GenericCalculation.portfolio_annualised_performance(weights=weights,
                                                                    mean_returns=mean_returns,
                                                                    cov_matrix=cov_matrix)
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return results, weights_record

    @staticmethod
    def generate_random_portfolios(data: pd.DataFrame, risk_free_rate: float = 0, num_portfolios: int = 50000):
        """
        Genera portafogli casuali e restituisce un DataFrame con i risultati.

        :param data: DataFrame contenente i prezzi storici degli asset.
        :param risk_free_rate: Tasso di interesse privo di rischio.
        :param num_portfolios: Numero di portafogli da generare.
        :return: DataFrame contenente i rendimenti, la volatilità e il rapporto di Sharpe per ciascun portafoglio.
        """
        returns = data.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_portfolios = num_portfolios
        risk_free_rate = risk_free_rate
        if type(returns.columns) == pd.core.indexes.multi.MultiIndex:
            ticker = [x[0] for x in returns.columns]
        else:
            ticker = returns.columns

        results, weights = EfficientFrontierCalculation.random_portfolios(num_portfolios=num_portfolios,
                                                                          num_asset=len(ticker),
                                                                          mean_returns=mean_returns,
                                                                          cov_matrix=cov_matrix,
                                                                          risk_free_rate=risk_free_rate)
        portfolio = {'Returns': results[1, :], 'Volatility': results[0, :], 'SharpeRatio': results[2, :]}

        # Estende il dizionario originale per includere ciascun asset e il suo peso nel portafoglio
        for counter, symbol in enumerate(ticker):
            portfolio[symbol + '_weight'] = [Weight[counter] for Weight in weights]

        return pd.DataFrame(portfolio)

    # OPTIMIZED EFFICIENT FRONTIER
    @staticmethod
    def neg_sharpe_ratio(weights: np.array, mean_returns: np.array, cov_matrix, risk_free_rate):
        """
        Calcola il rapporto di Sharpe negativo per l'ottimizzazione.

        :param weights: Un array contenente i pesi degli asset.
        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :param risk_free_rate: Il tasso privo di rischio.
        :return: Rapporto di Sharpe negativo.
        """
        p_var, p_ret = GenericCalculation.portfolio_annualised_performance(weights=weights,
                                                                           mean_returns=mean_returns,
                                                                           cov_matrix=cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    @staticmethod
    def max_sharpe_ratio(mean_returns: np.array, cov_matrix: np.matrix, risk_free_rate: float = 0):
        """
        Ottimizza il portafoglio per massimizzare il rapporto di Sharpe.

        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :param risk_free_rate: Il tasso privo di rischio.
        :return: Il risultato dell'ottimizzazione (portafoglio con massimo rapporto di Sharpe).
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(EfficientFrontierCalculation.neg_sharpe_ratio, num_assets * [1. / num_assets, ],
                              args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    @staticmethod
    def portfolio_volatility(weights: np.array, mean_returns: np.array, cov_matrix: np.matrix):
        """
        Calcola la volatilità annualizzata del portafoglio.

        :param weights: Pesi degli asset nel portafoglio.
        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :return: Volatilità annualizzata del portafoglio.
        """
        return GenericCalculation.portfolio_annualised_performance(weights=weights,
                                                                   mean_returns=mean_returns,
                                                                   cov_matrix=cov_matrix)[0]

    @staticmethod
    def min_variance(mean_returns: np.array, cov_matrix: np.matrix):
        """
        Ottimizza il portafoglio per minimizzare la volatilità.

        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :return: Il risultato dell'ottimizzazione (portafoglio con minima volatilità).
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))

        result = sco.minimize(EfficientFrontierCalculation.portfolio_volatility, num_assets * [1. / num_assets, ],
                              args=args, method='SLSQP', bounds=bounds, constraints=constraints)

        return result

    @staticmethod
    def efficient_return(mean_returns: np.array, cov_matrix: np.matrix, target: float):
        """
        Ottimizza il portafoglio per ottenere un rendimento target con la minima volatilità.

        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :param target: Il rendimento target desiderato.
        :return: Il risultato dell'ottimizzazione (portafoglio con il rendimento target).
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)

        def portfolio_return(weights):
            return GenericCalculation.portfolio_annualised_performance(weights=weights,
                                                                       mean_returns=mean_returns,
                                                                       cov_matrix=cov_matrix)[1]

        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = sco.minimize(EfficientFrontierCalculation.portfolio_volatility,
                              num_assets * [1. / num_assets, ],
                              args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    @staticmethod
    def efficient_frontier(mean_returns: np.array, cov_matrix: np.matrix, returns_range: np.array or list):
        """
        Calcola i portafogli efficienti per una gamma di rendimenti target.

        :param mean_returns: Un array dei rendimenti medi degli asset.
        :param cov_matrix: Una matrice di covarianza degli asset.
        :param returns_range: Una gamma di rendimenti target.
        :return: Una lista di portafogli ottimizzati per ciascun rendimento target.
        """
        efficients = []
        for ret in returns_range:
            efficients.append(EfficientFrontierCalculation.efficient_return(mean_returns, cov_matrix, ret))
        return efficients

    @staticmethod
    def generate_efficient_frontier(data: pd.DataFrame, risk_free_rate: float):
        """
        Genera l'intera frontiera efficiente basata sui dati storici forniti.

        :param data: DataFrame contenente i prezzi storici degli asset.
        :param risk_free_rate: Il tasso privo di rischio.
        :return: Un dizionario contenente i risultati dell'ottimizzazione, inclusi i portafogli con massimo Sharpe ratio, 
                 minima volatilità e la frontiera efficiente.
        """
        returns = data.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        risk_free_rate = risk_free_rate
        if type(returns.columns) == pd.core.indexes.multi.MultiIndex:
            ticker = [x[0] for x in returns.columns]
        else:
            ticker = returns.columns
        # MAX SHARPE RATIO OPTIMIZED PORTFOLIO
        max_sharpe = EfficientFrontierCalculation.max_sharpe_ratio(mean_returns=mean_returns,
                                                                   cov_matrix=cov_matrix,
                                                                   risk_free_rate=risk_free_rate)
        sdp, rp = GenericCalculation.portfolio_annualised_performance(weights=max_sharpe['x'],
                                                                      mean_returns=mean_returns,
                                                                      cov_matrix=cov_matrix)
        max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=ticker, columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i, 2) for i in max_sharpe_allocation.allocation]
        max_sharpe_allocation = max_sharpe_allocation.T
        # MIN VOLATILITY RATIO OPTIMIZED PORTFOLIO
        min_vol = EfficientFrontierCalculation.min_variance(mean_returns, cov_matrix)
        sdp_min, rp_min = GenericCalculation.portfolio_annualised_performance(weights=min_vol['x'],
                                                                              mean_returns=mean_returns,
                                                                              cov_matrix=cov_matrix)
        min_vol_allocation = pd.DataFrame(min_vol.x, index=ticker, columns=['allocation'])
        min_vol_allocation.allocation = [round(i, 2) for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T

        # ANNUAL RETURNS AND VOLATILITY
        an_vol = np.std(returns) * np.sqrt(252)
        an_rt = mean_returns * 252
        #
        target = np.linspace(rp_min, np.max(an_rt), 50)
        efficient_portfolios = EfficientFrontierCalculation.efficient_frontier(mean_returns=mean_returns,
                                                                               cov_matrix=cov_matrix,
                                                                               returns_range=target)

        return {
            "MaxOptimizedSharpeRatio": {
                "Returns": rp,
                "Volatility": sdp,
                "Allocation": {c: max_sharpe_allocation[c].iloc[0] for c in max_sharpe_allocation.columns}},
            "MinOptimizedVolatility": {
                "Returns": rp_min,
                "Volatility": sdp_min,
                "Allocation": {c: min_vol_allocation[c].iloc[0] for c in min_vol_allocation.columns}},
            "IndividualStocksReturnVolatility": {isin: {"Returns": round(an_rt[i], 2),
                                                        "Volatility": round(an_vol[i], 2)}
                                                 for i, isin in enumerate(ticker)},
            "OptimizedEfficientFrontier": {
                "Returns":  target,
                "Volatility": [p['fun'] for p in efficient_portfolios]}
            }

    # PLOTS
    @staticmethod
    def plot_generated_portfolios(volatility: list or pd.Series or np.array,
                                  returns: list or pd.Series or np.array,
                                  sharpe_ratio: list or pd.Series or np.array = None,
                                  title: str = 'Efficient Frontier',
                                  add_sharpe: bool = True):
        """
        Genera il grafico dei portafogli casuali generati, colorati in base al rapporto di Sharpe.

        :param volatility: La lista di volatilitá dei portafogli.
        :param returns: La lista dei rendimenti attesi dei portafogli.
        :param sharpe_ratio: Una lista opzionale di valori di Sharpe ratio.
        :param title: Titolo del grafico.
        :param add_sharpe: Se True, include il rapporto di Sharpe nel grafico come scala di colore.
        :return: Il grafico dei portafogli casuali.
        """
        if add_sharpe and sharpe_ratio is not None:
            fig = px.scatter(x=volatility, y=returns, color=sharpe_ratio,
                            color_continuous_scale='RdYlGn', labels={'color': 'Sharpe Ratio'})
        else:
            fig = go.Figure(data=go.Scatter(x=volatility, y=returns, mode='markers',
                                            marker=dict(color='blue', size=5, line=dict(width=1, color='black'))))

        fig.update_layout(title=title,
                        xaxis_title="Volatility (Std. Deviation)",
                        yaxis_title="Expected Returns",
                        coloraxis_colorbar=dict(title="Sharpe Ratio"))
        fig.show()
    
    @staticmethod
    def plot_efficient_frontier(efficient_frontier_points: list[tuple[float, float]],
                                min_volatility_point: tuple[float, float] = None,
                                max_sharpe_point: tuple[float, float] = None,
                                additional_points: dict[str, tuple[float, float]] = None):
        """
        Genera il grafico della frontiera efficiente e aggiunge punti chiave come il minimo di volatilità e il massimo Sharpe ratio.

        :param efficient_frontier_points: Lista di tuple contenenti volatilità e rendimento per la frontiera efficiente.
        :param min_volatility_point: Punto con la minima volatilità.
        :param max_sharpe_point: Punto con il massimo rapporto di Sharpe.
        :param additional_points: Dizionario con punti addizionali da plottare sul grafico.
        :return: Il grafico della frontiera efficiente.
        """
        # Plot della frontiera efficiente
        frontier_volatility = [point[0] for point in efficient_frontier_points]
        frontier_returns = [point[1] for point in efficient_frontier_points]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=frontier_volatility, y=frontier_returns, mode='lines',
                                 name='Efficient Frontier', line=dict(color='black', dash='dot')))

        # Aggiungi il punto di minima volatilità
        if min_volatility_point is not None:
            fig.add_trace(go.Scatter(x=[min_volatility_point[0]], y=[min_volatility_point[1]],
                                     mode='markers', marker=dict(color='red', size=10, symbol='star'),
                                     name='Minimum Volatility'))

        # Aggiungi il punto di massimo Sharpe Ratio
        if max_sharpe_point is not None:
            fig.add_trace(go.Scatter(x=[max_sharpe_point[0]], y=[max_sharpe_point[1]],
                                     mode='markers', marker=dict(color='green', size=10, symbol='star'),
                                     name='Maximum Sharpe Ratio'))

        # Aggiungi eventuali punti addizionali (azioni singole)
        if additional_points is not None:
            for isin, point in additional_points.items():
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]],
                                         mode='markers', marker=dict(size=8), name=isin))

        fig.update_layout(title="Efficient Frontier",
                         xaxis_title="Volatility (Std. Deviation)",
                         yaxis_title="Expected Returns",
                         legend_title="Legend")
        fig.show()

    @staticmethod
    def plot_generated_portfolios_matplotlib(volatility: list or pd.Series or np.array,
                                             returns: list or pd.Series or np.array,
                                             sharpe_ratio: list or pd.Series or np.array = None,
                                             title: str = 'Efficient Frontier',
                                             add_sharpe: bool = True):
        """
        Genera il grafico dei portafogli casuali generati, colorati in base al rapporto di Sharpe.

        :param volatility: La lista di volatilitá dei portafogli.
        :param returns: La lista dei rendimenti attesi dei portafogli.
        :param sharpe_ratio: Una lista opzionale di valori di Sharpe ratio.
        :param title: Titolo del grafico.
        :param add_sharpe: Se True, include il rapporto di Sharpe nel grafico come scala di colore.
        :return: Il grafico dei portafogli casuali.
        """
        if add_sharpe and sharpe_ratio is not None:
            _ = plt.scatter(x=volatility, y=returns, c=sharpe_ratio,
                            cmap='RdYlGn', edgecolors='black')
            plt.colorbar(_)
        else:
            _ = plt.scatter(x=volatility, y=returns)
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title(title)
        return _

    @staticmethod
    def plot_efficient_frontier_matplotlib(efficient_frontier_points: list[tuple[float, float]],
                                           min_volatility_point: tuple[float, float] = None,
                                           max_sharpe_point: tuple[float, float] = None,
                                           additional_points: dict[str, tuple[float, float]] = None):
        """
        Genera il grafico della frontiera efficiente e aggiunge punti chiave come il minimo di volatilità e il massimo Sharpe ratio.

        :param efficient_frontier_points: Lista di tuple contenenti volatilità e rendimento per la frontiera efficiente.
        :param min_volatility_point: Punto con la minima volatilità.
        :param max_sharpe_point: Punto con il massimo rapporto di Sharpe.
        :param additional_points: Dizionario con punti addizionali da plottare sul grafico.
        :return: Il grafico della frontiera efficiente.
        """
        _ = plt.plot([x[0] for x in efficient_frontier_points],
                     [x[1] for x in efficient_frontier_points], '-.', color='black', label='efficient frontier')
        # ADD POINTS TO PLOT
        if min_volatility_point is not None:
            plt.plot(min_volatility_point[0], min_volatility_point[1],
                     marker='*', color='r', label='Minimum volatility')
        if max_sharpe_point is not None:
            plt.plot(max_sharpe_point[0], max_sharpe_point[1],
                     marker='*', color='r', label='Maximum Sharpe ratio')
        if additional_points is not None:
            for isin, point in additional_points.items():
                plt.plot(point[0], point[1], marker='o', label=isin)
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        plt.legend()
        return _