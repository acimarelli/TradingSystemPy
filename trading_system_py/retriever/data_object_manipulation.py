import datetime

import pandas as pd
from functools import reduce


class DataObjectManipulation:

    def __init__(self, data: dict[str, dict]):
        self.data = data
        self.ticker = list(self.data.keys())

    # FILTER DATA OBJECT BY KEY
    def filter_data_obj(self, isin_list: list[str]):
        return {ticker: self.data[ticker] for ticker in isin_list}

    # GET single attribute for SPECIFIC Ticker
    def get_description(self, isin: str):
        return self.data[isin]["Description"]

    def get_industry(self, isin: str):
        return self.data[isin]["Industry"]

    def get_sector(self, isin: str):
        return self.data[isin]["Sector"]

    def get_fundamental(self, isin: str):
        return self.data[isin]["Fundamentals"]

    def get_history(self, isin: str):
        return self.data[isin]["History"]
    
    def get_put_call_ratio(self, isin: str):
        return self.data[isin]["PutCallRatio"]

    # GET single/multiple attribute for SPECIFIC Ticker TIME SERIES
    def get_ts_value(self, isin: str, attribute_list: list):
        return self.get_history(isin=isin)[attribute_list]

    # GET single attribute for ALL Tickers
    def get_all_description(self):
        return {isin:self.get_description(isin=isin) for isin in self.ticker}

    def get_all_industry(self):
        return {isin:self.get_industry(isin=isin) for isin in self.ticker}

    def get_all_sector(self):
        return {isin:self.get_sector(isin=isin) for isin in self.ticker}

    def get_all_fundamental(self, as_pandas: bool = True):
        fundamentals = {isin:self.get_fundamental(isin=isin) for isin in self.ticker}
        if as_pandas:
            return pd.DataFrame(fundamentals)
        else:
            return fundamentals
    
    def get_all_put_call_ratio(self, as_pandas: bool = True):
        put_call_ratios = {isin:self.get_put_call_ratio(isin=isin) for isin in self.ticker}
        if as_pandas:
            dfs = []
            for key, df in put_call_ratios.items():
                df = df.rename(columns={'Put/Call Ratio': f'PutCallRatio_{key}'})  # Rinomina la colonna
                dfs.append(df)
            return reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
        else:
            return put_call_ratios

    def get_all_history(self, as_pandas: bool = True):
        list_all_ts = []
        if as_pandas:
            for isin in self.ticker:
                ts = pd.DataFrame(self.get_history(isin=isin))
                ts.columns = [(isin, c) for c in ts.columns]
                ts.columns = pd.MultiIndex.from_tuples(ts.columns, names=['Ticker','Attribute'])
                list_all_ts.append(ts)
            return pd.concat(list_all_ts, axis=1)
        else:
            return {isin: self.get_history(isin=isin) for isin in self.ticker}

    # GET single/multiple attribute for ALL Ticker TIME SERIES
    def get_all_ts_value(self, attribute_list: list, as_pandas: bool = True):
        if as_pandas:
            col_to_extract = []
            for isin in self.ticker:
                for attribute in attribute_list:
                    col_to_extract.append(tuple([isin,attribute]))
            return self.get_all_history(as_pandas=as_pandas)[col_to_extract]
        else:
            return {isin: self.get_ts_value(isin=isin, attribute_list=attribute_list) for isin in self.ticker}

    # GET  filtered data for SPECIFIC ticker LIST
    def get_filtered_history(self, isin_list: list[str], as_pandas: bool = True):
        list_all_ts = []
        if as_pandas:
            for isin in isin_list:
                ts = pd.DataFrame(self.get_history(isin=isin))
                ts.columns = [(isin, c) for c in ts.columns]
                ts.columns = pd.MultiIndex.from_tuples(ts.columns, names=['Ticker', 'Attribute'])
                list_all_ts.append(ts)
            return pd.concat(list_all_ts, axis=1)
        else:
            return {isin: self.get_history(isin=isin) for isin in isin_list}

    def get_filtered_ts_value(self, isin_list: list[str], attribute_list: list, as_pandas: bool = True):
        if as_pandas:
            col_to_extract = []
            for isin in isin_list:
                for attribute in attribute_list:
                    col_to_extract.append(tuple([isin, attribute]))
            return self.get_all_history(as_pandas=as_pandas)[col_to_extract]
        else:
            return {isin: self.get_ts_value(isin=isin, attribute_list=attribute_list) for isin in isin_list}

    # FILTER HISTORY IN DATA-OBJECT
    def filter_history(self, starting_date: datetime.datetime|str, end_date: datetime.datetime|str):
        for isin in self.ticker:
            self.data[isin].update({"History": self.data[isin]["History"].loc[(self.data[isin]["History"].index.tz_convert(None) >= starting_date) &
                                                                              (self.data[isin]["History"].index.tz_convert(None) <= end_date)]})
        return self.data
