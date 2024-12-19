import os
import pickle

from trading_system_py.retriever.data_object_manipulation import DataObjectManipulation
from trading_system_py.retriever.risk_free_object_manipulation import RiskFreeObjectManipulation


class ImportDataFromObject(DataObjectManipulation, RiskFreeObjectManipulation):

    def __init__(self, path: str, filename: str):
        # IMPORT PICKLE OBJECT
        with open(os.path.join(path, filename), 'rb') as pickle_obj:
            input_obj = pickle.load(pickle_obj)

        # SET __INIT__ FROM IMPORTED PICKLE OBJECT
        self.ticker = input_obj["tickers"]
        self.data = input_obj["data"]
        self.risk_free_ticker = input_obj["risk_free_ticker"]
        self.risk_free_rate_curve = input_obj["risk_free_rate_curve"]

        # CALL DATA MANIPULATION OBJECT TO USE ITS FUNCTIONS
        DataObjectManipulation.__init__(self, data=self.data)
        RiskFreeObjectManipulation.__init__(self, risk_free_rate_curve=self.risk_free_rate_curve)
