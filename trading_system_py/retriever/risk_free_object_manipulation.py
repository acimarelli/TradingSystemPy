import datetime
import pandas as pd


class RiskFreeObjectManipulation:

    def __init__(self, risk_free_rate_curve: pd.DataFrame):
        self.risk_free_rate_curve = risk_free_rate_curve

    # GET ALL risk-free rate curve
    def get_all_risk_free_curve(self):
        return self.risk_free_rate_curve

    # GET SPECIFIC risk-free rate curve
    def get_first_risk_free_rate(self):
        return self.risk_free_rate_curve.loc[self.risk_free_rate_curve.index == min(self.risk_free_rate_curve.index)]

    def get_latest_risk_free_rate(self):
        return self.risk_free_rate_curve.loc[self.risk_free_rate_curve.index == max(self.risk_free_rate_curve.index)]

    def get_specific_risk_free_rate(self, evaluation_date: datetime.datetime):
        return self.risk_free_rate_curve.loc[self.risk_free_rate_curve.index ==
                                             self.nearest_date(items=self.risk_free_rate_curve.index,
                                                               evaluation_date=evaluation_date)]

    def get_mean_annualized_risk_free_rate_between_dates(self,
                                                         start_date: str or datetime.datetime,
                                                         end_date: str or datetime.datetime):
        return self.risk_free_rate_curve.loc[(self.risk_free_rate_curve.index >= start_date) &
                                             (self.risk_free_rate_curve.index <= end_date)]\
            .get('annualized').apply(lambda x: x/100).mean()

    @staticmethod
    def nearest_date(items, evaluation_date: datetime.datetime):
        return min(items, key=lambda x: abs(x - evaluation_date))
