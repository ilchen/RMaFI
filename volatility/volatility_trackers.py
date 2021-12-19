# coding: utf-8
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from math import sqrt

class VolatilityTracker:
    """
    Represents an estimator for volatility forecasting parameters
    """

    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    VARIANCE = 'Variance'
    TO_ANNUAL_MULTIPLIER = sqrt(252)

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        '''
        Calculates daily volatilitiesfrom either a panda series object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a desired asset class (i.e. the 'start' abd 'end'
        arguments must be provided)
        :param asset_prices_series: a pandas Series object indexed by dates
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param asset: the ticker simbol of the asset whose asset price changes are to be analyzed. It expects
                      a Yahoo Finance convention for ticker symbols
        '''
        if asset_prices_series is None:
            if start is None or end is None or asset is None:
                raise ValueError("Neither asset_price_series nor (start, end, asset) arguments are provided")
            data = web.get_data_yahoo(asset, start, end)
            asset_prices_series = data['Adj Close']
        # Dropping the first row as it doesn't contain a daily return value
        self.data = pd.DataFrame({self.CLOSE : asset_prices_series, self.DAILY_RETURN : asset_prices_series.pct_change()},
                                 index=asset_prices_series.index).iloc[1:]
        # Essentially only self.data[self.VARIANCE].iloc[1] needs to be set to self.data.ui.iloc[0]**2
        self.data[self.VARIANCE] = self.data.ui.iloc[0]**2

    def  get_daily_volatilities(self):
        return  np.sqrt(self.data[self.VARIANCE].iloc[1:])

    def  get_annual_volatilities(self):
        return  self.get_daily_volatilities() * self.TO_ANNUAL_MULTIPLIER

    def  get_dates(self):
        return  self.data.index[1:]

    def  get_adj_close_prices(self):
        return  self.data[self.CLOSE].values[1:]

class EWMAVolatilityTracker(VolatilityTracker):
    """
    Represents an maximum likelihood estimator for the λ parameter of the EWMA method of forecasting volatility
    """
    def __init__(self, lamda, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)
        self.lamda = lamda

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                       + β * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data[self.DAILY_RETURN])):
            self.data[self.VARIANCE].iloc[i] = (1 - lamda) * self.data[self.DAILY_RETURN].iloc[i - 1] ** 2 \
                                               + lamda * self.data[self.VARIANCE].iloc[i - 1]
