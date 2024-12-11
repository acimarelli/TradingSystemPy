import talib
import inspect
import numpy as np
import pandas as pd

class TechnicalIndicators:

    @staticmethod
    def calculate_indicator(indicator_name: str, *args, **kwargs):
        """
        Calcola un indicatore tecnico dato il suo nome e i parametri.
        
        :param indicator_name: Nome dell'indicatore tecnico (es. 'SMA', 'RSI', ecc.)
        :param args: Argomenti posizionali da passare alla funzione dell'indicatore.
        :param kwargs: Argomenti opzionali da passare alla funzione dell'indicatore.
        :return: Valori dell'indicatore calcolato.
        """
        # Verifica se l'indicatore richiesto esiste in TA-Lib
        if hasattr(talib, indicator_name):
            # Ottieni la funzione di TA-Lib per l'indicatore
            func = getattr(talib, indicator_name)
            # Chiama la funzione con gli argomenti e ritorna il risultato
            return func(*args, **kwargs)
        else:
            raise ValueError(f"Indicatore '{indicator_name}' non trovato in TA-Lib.")
    
    @staticmethod
    def get_function_signature(func_name: str):
        """
        Stampa la firma della funzione specificata (es. 'ATR') per visualizzare gli argomenti richiesti.
        
        :param func_name: Nome della funzione in TA-Lib (es. 'ATR', 'RSI', 'SMA', ecc.)
        :return: La firma della funzione con i suoi argomenti.
        """
        if hasattr(talib, func_name):
            func = getattr(talib, func_name)
            # Ottiene la firma della funzione
            signature = inspect.signature(func)
            return signature
        else:
            raise ValueError(f"La funzione '{func_name}' non esiste in TA-Lib.")
    
    @staticmethod
    def get_all_available_indicators():
        """
        Restituisce una lista di tutti gli indicatori tecnici disponibili in TA-Lib.
        """
        return [func for func in dir(talib) if func.isupper() if '__TA_FUNCTION_NAMES__' not in func]
    
    @staticmethod
    def get_overlap_indicators():
        """
        BBANDS               Bollinger Bands
        DEMA                 Double Exponential Moving Average
        EMA                  Exponential Moving Average
        HT_TRENDLINE         Hilbert Transform - Instantaneous Trendline
        KAMA                 Kaufman Adaptive Moving Average
        MA                   Moving average
        MAMA                 MESA Adaptive Moving Average
        MAVP                 Moving average with variable period
        MIDPOINT             MidPoint over period
        MIDPRICE             Midpoint Price over period
        SAR                  Parabolic SAR
        SAREXT               Parabolic SAR - Extended
        SMA                  Simple Moving Average
        T3                   Triple Exponential Moving Average (T3)
        TEMA                 Triple Exponential Moving Average
        TRIMA                Triangular Moving Average
        WMA                  Weighted Moving Average
        """
        return ["BBANDS", "DEMA", "EMA", "HT_TRENDLINE", "KAMA", "MA", "MAMA", "MAVP", "MIDPOINT", 
                "MIDPRICE", "SAR", "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA"]
    
    @staticmethod
    def get_momentum_indicators():
        """
        ADX                  Average Directional Movement Index
        ADXR                 Average Directional Movement Index Rating
        APO                  Absolute Price Oscillator
        AROON                Aroon
        AROONOSC             Aroon Oscillator
        BOP                  Balance Of Power
        CCI                  Commodity Channel Index
        CMO                  Chande Momentum Oscillator
        DX                   Directional Movement Index
        MACD                 Moving Average Convergence/Divergence
        MACDEXT              MACD with controllable MA type
        MACDFIX              Moving Average Convergence/Divergence Fix 12/26
        MFI                  Money Flow Index
        MINUS_DI             Minus Directional Indicator
        MINUS_DM             Minus Directional Movement
        MOM                  Momentum
        PLUS_DI              Plus Directional Indicator
        PLUS_DM              Plus Directional Movement
        PPO                  Percentage Price Oscillator
        ROC                  Rate of change : ((price/prevPrice)-1)*100
        ROCP                 Rate of change Percentage: (price-prevPrice)/prevPrice
        ROCR                 Rate of change ratio: (price/prevPrice)
        ROCR100              Rate of change ratio 100 scale: (price/prevPrice)*100
        RSI                  Relative Strength Index
        STOCH                Stochastic
        STOCHF               Stochastic Fast
        STOCHRSI             Stochastic Relative Strength Index
        TRIX                 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        ULTOSC               Ultimate Oscillator
        WILLR                Williams' %R
        """
        return ["ADX",	"ADXR",	"APO",	"AROON",	"AROONOSC",	"BOP",	"CCI",	"CMO",	"DX",	"MACD",	"MACDEXT",	
                "MACDFIX",	"MFI",	"MINUS_DI",	"MINUS_DM",	"MOM",	"PLUS_DI",	"PLUS_DM",	"PPO",	"ROC",	"ROCP",	
                "ROCR",	"ROCR100",	"RSI",	"STOCH",	"STOCHF",	"STOCHRSI",	"TRIX",	"ULTOSC",	"WILLR"]
    
    @staticmethod
    def get_volume_indicators():
        """
        AD                   Chaikin A/D Line
        ADOSC                Chaikin A/D Oscillator
        OBV                  On Balance Volume
        """
        return ["AD", "ADOSC", "OBV"]
    
    @staticmethod
    def get_cycle_indicators():
        """
        HT_DCPERIOD          Hilbert Transform - Dominant Cycle Period
        HT_DCPHASE           Hilbert Transform - Dominant Cycle Phase
        HT_PHASOR            Hilbert Transform - Phasor Components
        HT_SINE              Hilbert Transform - SineWave
        HT_TRENDMODE         Hilbert Transform - Trend vs Cycle Mode
        """
        return ["HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE"]
    
    @staticmethod
    def price_transform():
        """
        AVGPRICE             Average Price
        MEDPRICE             Median Price
        TYPPRICE             Typical Price
        WCLPRICE             Weighted Close Price
        """
        return ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
    
    @staticmethod
    def get_volatility_indicators():
        """
        ATR                  Average True Range
        NATR                 Normalized Average True Range
        TRANGE               True Range
        """
        return ["ATR", "NATR", "TRANGE"]
    
    @staticmethod
    def get_pattern_recognition():
        """
        CDL2CROWS            Two Crows
        CDL3BLACKCROWS       Three Black Crows
        CDL3INSIDE           Three Inside Up/Down
        CDL3LINESTRIKE       Three-Line Strike
        CDL3OUTSIDE          Three Outside Up/Down
        CDL3STARSINSOUTH     Three Stars In The South
        CDL3WHITESOLDIERS    Three Advancing White Soldiers
        CDLABANDONEDBABY     Abandoned Baby
        CDLADVANCEBLOCK      Advance Block
        CDLBELTHOLD          Belt-hold
        CDLBREAKAWAY         Breakaway
        CDLCLOSINGMARUBOZU   Closing Marubozu
        CDLCONCEALBABYSWALL  Concealing Baby Swallow
        CDLCOUNTERATTACK     Counterattack
        CDLDARKCLOUDCOVER    Dark Cloud Cover
        CDLDOJI              Doji
        CDLDOJISTAR          Doji Star
        CDLDRAGONFLYDOJI     Dragonfly Doji
        CDLENGULFING         Engulfing Pattern
        CDLEVENINGDOJISTAR   Evening Doji Star
        CDLEVENINGSTAR       Evening Star
        CDLGAPSIDESIDEWHITE  Up/Down-gap side-by-side white lines
        CDLGRAVESTONEDOJI    Gravestone Doji
        CDLHAMMER            Hammer
        CDLHANGINGMAN        Hanging Man
        CDLHARAMI            Harami Pattern
        CDLHARAMICROSS       Harami Cross Pattern
        CDLHIGHWAVE          High-Wave Candle
        CDLHIKKAKE           Hikkake Pattern
        CDLHIKKAKEMOD        Modified Hikkake Pattern
        CDLHOMINGPIGEON      Homing Pigeon
        CDLIDENTICAL3CROWS   Identical Three Crows
        CDLINNECK            In-Neck Pattern
        CDLINVERTEDHAMMER    Inverted Hammer
        CDLKICKING           Kicking
        CDLKICKINGBYLENGTH   Kicking - bull/bear determined by the longer marubozu
        CDLLADDERBOTTOM      Ladder Bottom
        CDLLONGLEGGEDDOJI    Long Legged Doji
        CDLLONGLINE          Long Line Candle
        CDLMARUBOZU          Marubozu
        CDLMATCHINGLOW       Matching Low
        CDLMATHOLD           Mat Hold
        CDLMORNINGDOJISTAR   Morning Doji Star
        CDLMORNINGSTAR       Morning Star
        CDLONNECK            On-Neck Pattern
        CDLPIERCING          Piercing Pattern
        CDLRICKSHAWMAN       Rickshaw Man
        CDLRISEFALL3METHODS  Rising/Falling Three Methods
        CDLSEPARATINGLINES   Separating Lines
        CDLSHOOTINGSTAR      Shooting Star
        CDLSHORTLINE         Short Line Candle
        CDLSPINNINGTOP       Spinning Top
        CDLSTALLEDPATTERN    Stalled Pattern
        CDLSTICKSANDWICH     Stick Sandwich
        CDLTAKURI            Takuri (Dragonfly Doji with very long lower shadow)
        CDLTASUKIGAP         Tasuki Gap
        CDLTHRUSTING         Thrusting Pattern
        CDLTRISTAR           Tristar Pattern
        CDLUNIQUE3RIVER      Unique 3 River
        CDLUPSIDEGAP2CROWS   Upside Gap Two Crows
        CDLXSIDEGAP3METHODS  Upside/Downside Gap Three Methods
        """
        return ["CDL2CROWS",	"CDL3BLACKCROWS",	"CDL3INSIDE",	"CDL3LINESTRIKE",	"CDL3OUTSIDE",	
                "CDL3STARSINSOUTH",	"CDL3WHITESOLDIERS",	"CDLABANDONEDBABY",	"CDLADVANCEBLOCK",	"CDLBELTHOLD",	
                "CDLBREAKAWAY",	"CDLCLOSINGMARUBOZU",	"CDLCONCEALBABYSWALL",	"CDLCOUNTERATTACK",	"CDLDARKCLOUDCOVER",	
                "CDLDOJI",	"CDLDOJISTAR",	"CDLDRAGONFLYDOJI",	"CDLENGULFING",	"CDLEVENINGDOJISTAR",	"CDLEVENINGSTAR",	
                "CDLGAPSIDESIDEWHITE",	"CDLGRAVESTONEDOJI",	"CDLHAMMER",	"CDLHANGINGMAN",	"CDLHARAMI",	
                "CDLHARAMICROSS",	"CDLHIGHWAVE",	"CDLHIKKAKE",	"CDLHIKKAKEMOD",	"CDLHOMINGPIGEON",	
                "CDLIDENTICAL3CROWS",	"CDLINNECK",	"CDLINVERTEDHAMMER",	"CDLKICKING",	"CDLKICKINGBYLENGTH",	
                "CDLLADDERBOTTOM",	"CDLLONGLEGGEDDOJI",	"CDLLONGLINE",	"CDLMARUBOZU",	"CDLMATCHINGLOW",	
                "CDLMATHOLD",	"CDLMORNINGDOJISTAR",	"CDLMORNINGSTAR",	"CDLONNECK",	"CDLPIERCING",	
                "CDLRICKSHAWMAN",	"CDLRISEFALL3METHODS",	"CDLSEPARATINGLINES",	"CDLSHOOTINGSTAR",	"CDLSHORTLINE",	
                "CDLSPINNINGTOP",	"CDLSTALLEDPATTERN",	"CDLSTICKSANDWICH",	"CDLTAKURI",	"CDLTASUKIGAP",	
                "CDLTHRUSTING",	"CDLTRISTAR",	"CDLUNIQUE3RIVER",	"CDLUPSIDEGAP2CROWS",	"CDLXSIDEGAP3METHODS"]
    
    @staticmethod
    def get_statistic_function():
        """
        BETA                 Beta
        CORREL               Pearson's Correlation Coefficient (r)
        LINEARREG            Linear Regression
        LINEARREG_ANGLE      Linear Regression Angle
        LINEARREG_INTERCEPT  Linear Regression Intercept
        LINEARREG_SLOPE      Linear Regression Slope
        STDDEV               Standard Deviation
        TSF                  Time Series Forecast
        VAR                  Variance
        """
        return ["BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", 
                "LINEARREG_SLOPE", "STDDEV", "TSF", "VAR"]
    

class TechnicalIndicatorsDataObject(TechnicalIndicators):
    def __init__(self, open_prices, high_prices, low_prices, close_prices, volumes, real0 = None, real1 = None):
        """
        Inizializza l'oggetto con i dati storici (Open, High, Low, Close, Volume).
        """
        self.open = open_prices
        self.high = high_prices
        self.low = low_prices
        self.close = close_prices
        self.volume = volumes
        self.real = real0 if real0 is not None else close_prices
        self.real0 = real0 if real0 is not None else close_prices
        self.real1 = real1 if real1 is not None else close_prices

        # Dizionario per memorizzare i risultati degli indicatori calcolati
        self.indicator_results = {}

    def calculate_indicator(self, indicator_name: str, *args, **kwargs):
        """
        Calcola un indicatore tecnico dato il suo nome e i parametri.
        
        :param indicator_name: Nome dell'indicatore tecnico (es. 'SMA', 'RSI', ecc.)
        :param args: Argomenti posizionali da passare alla funzione dell'indicatore.
        :param kwargs: Argomenti opzionali da passare alla funzione dell'indicatore.
        :return: Valori dell'indicatore calcolato.
        """
        if hasattr(talib, indicator_name):
            func = getattr(talib, indicator_name)
            signature = inspect.signature(func)
            bound_args = self._bind_signature_to_data(signature, args, kwargs)
            return func(**bound_args)
        else:
            raise ValueError(f"Indicatore '{indicator_name}' non trovato in TA-Lib.")
    
    def _bind_signature_to_data(self, signature, args, kwargs):
        """
        Associa automaticamente i parametri della firma della funzione agli attributi di classe (open, high, low, close)
        e gestisce anche i parametri opzionali con valori predefiniti (come 'timeperiod').

        :param signature: La firma della funzione dell'indicatore tecnico.
        :param args: Argomenti posizionali passati dall'utente.
        :param kwargs: Argomenti opzionali passati dall'utente.
        :return: Un dizionario con gli argomenti corretti per la funzione.
        """
        bound_args = {}
        param_names = list(signature.parameters.keys())
        
        # Mappa i nomi dei parametri agli attributi di classe (self.open, self.close, ecc.)
        data_mapping = {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'real': self.real,
            'real0': self.real0,
            'real1': self.real1
        }
        
        for param in param_names:
            param_obj = signature.parameters[param]
            
            # Verifica se il parametro è passato esplicitamente in kwargs
            if param in kwargs:
                bound_args[param] = kwargs[param]
            # Se il parametro è uno degli attributi della classe (open, high, low, close), lo associa automaticamente
            elif param in data_mapping:
                bound_args[param] = data_mapping[param]
            # Se il parametro ha un valore predefinito (come timeperiod=14), usa il valore predefinito se non è passato dall'utente
            elif param_obj.default is not inspect.Parameter.empty:
                bound_args[param] = param_obj.default
            # Se ci sono args posizionali, usali
            elif args:
                bound_args[param] = args[0]
                args = args[1:]
            else:
                raise ValueError(f"Argomento '{param}' mancante.")
        
        return bound_args
    
    def calculate_all_indicators(self, specific_indicator_list: list[tuple[str, str, dict]] = None, as_pandas: bool = False):
        """
        Calcola automaticamente tutti gli indicatori disponibili in TA-Lib o solo quelli specificati e memorizza i risultati
        in self.indicator_results.

        :param specific_indicator_list: Una lista opzionale di tuple contenente il nome dell'output, 
                                        il nome dell'indicatore in TA-Lib e i suoi parametri. 
                                        Esempio: [('SMA_50', 'SMA', {'timeperiod': 50})]
        :param as_pandas: Se True, restituisce i risultati come DataFrame pandas, altrimenti restituisce un dizionario.
        :return: Dizionario o DataFrame contenente gli indicatori calcolati.
        """
        self.indicator_results = {}
        # Se specific_indicator_list è fornito, usa solo gli indicatori specificati
        if specific_indicator_list:
            indicator_list = specific_indicator_list
        else:
            # Altrimenti, usa tutti gli indicatori disponibili in TA-Lib
            indicator_list = [(indicator, indicator, {}) for indicator in self.get_all_available_indicators()]
        
        for output_name, indicator_name, params in indicator_list:
            try:
                # Calcoliamo l'indicatore con i parametri specificati, se presenti
                indicator_result = self.calculate_indicator(indicator_name, **params)

                # Verifichiamo se il risultato è una tupla (es. BBANDS)
                if isinstance(indicator_result, (tuple, list)):
                    for idx, result in enumerate(indicator_result):
                        self.indicator_results[f"{output_name}_{idx}"] = result
                else:
                    self.indicator_results[output_name] = indicator_result

            except Exception as e:
                print(f"Errore nel calcolo dell'indicatore '{indicator_name}': {str(e)}")

        return pd.DataFrame(self.indicator_results) if as_pandas else self.indicator_results
    
