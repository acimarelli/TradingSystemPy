import numpy as np
from src.retriever.data_object_manipulation import DataObjectManipulation
from src.analysis.asset_allocation.efficient_frontier_calculation import EfficientFrontierCalculation


class EfficientFrontier:
    """
    Questa classe estende EfficientFrontierCalculation per calcolare, generare e visualizzare portafogli di frontiera efficiente
    utilizzando dati storici e simulazioni Monte Carlo.
    """

    def __init__(self, data: dict[str, dict], risk_free_rate: float = 0.02,
                 attribute_for_returns: str = "Close", num_sim_portfolios: int = 50000):
        """
        Inizializza l'oggetto EfficientFrontier con i dati forniti e calcola sia portafogli simulati che ottimizzati.

        :param data: Dizionario dei dati storici per ciascun titolo (es. 'Open', 'Close', 'High', 'Low', ecc.)
        :param risk_free_rate: Tasso di rendimento privo di rischio, usato per calcolare il rapporto di Sharpe.
                               (Utilizzare il tasso giornaliero per analisi giornaliere e il tasso annualizzato per analisi 
                               su base annuale o a lungo termine, mantenendo la coerenza tra il tipo di rendimento del 
                               portafoglio e il tasso risk-free).
        :param attribute_for_returns: L'attributo dei dati (es. 'Close') da utilizzare per calcolare i rendimenti.
        :param num_sim_portfolios: Numero di portafogli simulati casualmente da generare.
        """
        self.data = data
        self.ticker = list(self.data.keys())
        self.risk_free_rate = risk_free_rate
        self.attribute_for_returns = attribute_for_returns
        self.num_sim_portfolios = num_sim_portfolios
        self.generated_portfolios = EfficientFrontierCalculation.generate_random_portfolios(
            data=DataObjectManipulation(data=self.data).get_all_ts_value(attribute_list=[self.attribute_for_returns]),
            risk_free_rate=self.risk_free_rate,
            num_portfolios=num_sim_portfolios)
        self.optimized_efficient_frontier = EfficientFrontierCalculation.generate_efficient_frontier(data=DataObjectManipulation(data=self.data).get_all_ts_value(attribute_list=[self.attribute_for_returns]),
                                                                             risk_free_rate=self.risk_free_rate)

    def get_max_sharpe_random_portfolios(self):
        """
        Restituisce il portafoglio simulato con il massimo rapporto di Sharpe tra quelli generati casualmente.

        :return: Dizionario con il rendimento, la volatilità e l'allocazione del portafoglio con massimo Sharpe.
        """
        raw_data = self.generated_portfolios.loc[self.generated_portfolios.get("SharpeRatio") ==
                                                 np.max(self.generated_portfolios.get("SharpeRatio"))]
        return {
            "Returns": raw_data.get('Returns').iloc[0],
            "Volatility": raw_data.get('Volatility').iloc[0],
            "Allocation": {c[:-7]: round(raw_data[c].iloc[0], 2) for c in raw_data.columns if '_weight' in c}
        }

    def get_min_volatility_random_portfolios(self):
        """
        Restituisce il portafoglio simulato con la minima volatilità tra quelli generati casualmente.

        :return: Dizionario con il rendimento, la volatilità e l'allocazione del portafoglio con minima volatilità.
        """
        raw_data = self.generated_portfolios.loc[self.generated_portfolios.get("Volatility") ==
                                                 np.min(self.generated_portfolios.get("Volatility"))]
        return {
            "Returns": raw_data.get('Returns').iloc[0],
            "Volatility": raw_data.get('Volatility').iloc[0],
            "Allocation": {c[:-7]: round(raw_data[c].iloc[0], 2) for c in raw_data.columns if '_weight' in c}
        }

    def get_optimized_max_sharpe(self):
        """
        Restituisce il portafoglio ottimizzato con il massimo rapporto di Sharpe.

        :return: Dizionario con il rendimento, la volatilità e l'allocazione del portafoglio ottimizzato per il massimo Sharpe.
        """
        return self.optimized_efficient_frontier.get("MaxOptimizedSharpeRatio")

    def get_optimized_min_volatility(self):
        """
        Restituisce il portafoglio ottimizzato con la minima volatilità.

        :return: Dizionario con il rendimento, la volatilità e l'allocazione del portafoglio ottimizzato per la minima volatilità.
        """
        return self.optimized_efficient_frontier.get("MinOptimizedVolatility")

    def get_optimized_efficient_frontier(self):
        """
        Restituisce i punti della frontiera efficiente ottimizzata.

        :return: Dizionario con i rendimenti e la volatilità della frontiera efficiente ottimizzata.
        """
        return self.optimized_efficient_frontier.get("OptimizedEfficientFrontier")

    def get_individual_stock_return_volatility(self):
        """
        Restituisce i rendimenti e la volatilità annualizzati per ciascun titolo individuale.

        :return: Dizionario contenente rendimenti e volatilità per ciascun titolo.
        """
        return self.optimized_efficient_frontier.get("IndividualStocksReturnVolatility")

    def plot_generated_portfolios(self, add_sharpe: bool = True):
        """
        Visualizza i portafogli generati casualmente, con la possibilità di aggiungere il rapporto di Sharpe come colore.

        :param add_sharpe: Se True, mostra il rapporto di Sharpe sui portafogli generati.
        :return: Un grafico scatter che mostra la frontiera efficiente basata sui portafogli casuali generati.
        """
        return EfficientFrontierCalculation.plot_generated_portfolios(volatility=self.generated_portfolios.get("Volatility"),
                                                                      returns=self.generated_portfolios.get("Returns"),
                                                                      sharpe_ratio=self.generated_portfolios.get("SharpeRatio"),
                                                                      title=f"Efficient Frontier from random generated portfolios of "
                                                                            f"stocks\n{self.ticker}",
                                                                      add_sharpe=add_sharpe)

    def plot_efficient_frontier(self):
        """
        Visualizza la frontiera efficiente ottimizzata con i punti di volatilità minima e massimo Sharpe.

        :return: Un grafico che mostra la frontiera efficiente con i portafogli ottimizzati e i singoli titoli.
        """
        efficient_frontier_points = [(x, y) for x, y in zip(self.get_optimized_efficient_frontier().get("Volatility"),
                                                            self.get_optimized_efficient_frontier().get("Returns"))]
        min_volatility_point = (self.get_optimized_min_volatility().get("Volatility"),
                                self.get_optimized_min_volatility().get("Returns"))
        max_sharpe_point = (self.get_optimized_max_sharpe().get("Volatility"),
                            self.get_optimized_max_sharpe().get("Returns"))
        additional_points = {isin: [self.get_individual_stock_return_volatility().get(isin).get("Volatility"),
                                    self.get_individual_stock_return_volatility().get(isin).get("Returns")] for isin in self.get_individual_stock_return_volatility()}

        return EfficientFrontierCalculation.plot_efficient_frontier(efficient_frontier_points=efficient_frontier_points,
                                                                    min_volatility_point=min_volatility_point,
                                                                    max_sharpe_point=max_sharpe_point,
                                                                    additional_points=additional_points)