# coding: utf-8
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas.tseries.offsets import BDay
from math import sqrt, log, exp


class CorrelationTracker:
    """
    Represents a tracker for the correlation and for forecasting future correlations between given assets.
    You instantiate an object of this class with a consecutive range of daily closing prices of given assets,
    and then use it to inspect past correlations as well as to predict future correlations.
    """

    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    COVARIANCE = 'Covariance'
    TO_ANNUAL_MULTIPLIER = sqrt(252)

    def __init__(self, asset_prices=None, start=None, end=None, assets=['^GSPC', 'AAPL']):
        """
        Calculates daily volatilities from either a panda series object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a desired asset class (i.e. the 'start' abd 'end'
        arguments must be provided)
        :param asset_prices: a pandas DataFrame object indexed by dates containing two columns each representing closing
                             prices for a given asset
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param assets: a list containing two ticker symbols of the assets whose asset price changes are to be analyzed
                       to establish a correlation between them. It expects a Yahoo Finance convention for ticker symbols
        """
        if asset_prices is None:
            if start is None or end is None or assets is None or len(assets) != 2:
                raise ValueError("Neither asset_price_series nor (start, end, asset) arguments are provided")
            data = web.get_data_yahoo(assets, start, end)
            self.data = data['Adj Close']
        elif len(asset_prices.columns) != 2:
            raise ValueError("Wrong number of columns in submitted DataFrame")
        else:
            self.data = asset_prices.copy()

        # Daily percentage changes in the first asset class
        self.data.insert(loc=2, column=self.data.columns[0] + self.DAILY_RETURN,
                         value=self.data.iloc[:, 0].pct_change())
        # Daily percentage changes in the second asset class
        self.data.insert(loc=3, column=self.data.columns[1] + self.DAILY_RETURN,
                         value=self.data.iloc[:, 1].pct_change())
        self.data.insert(loc=4, column=self.COVARIANCE,
                         value=self.data.iloc[:, 2] * self.data.iloc[:, 3])
        # Dropping the first row as it doesn't contain a daily return value
        self.data = self.data.iloc[1:]

    def get_dates(self):
        return self.data.index[1:]

    def get_adj_close_prices_1st_asset(self):
        return self.data.iloc[1:, 0]

    def get_adj_close_prices_2nd_asset(self):
        return self.data.iloc[1:, 1]

    def get_next_business_day_correlation(self):
        """
        Returns the correlation on the next business day after the last day in the self.data.index
        """
        return pd.Series([None], index=[self.data.index[-1] + BDay()], dtype=self.data[self.COVARIANCE].dtype)

    def get_correlation_forecast(self, n):
        """
        :param n: an integer indicating for which business day in the future correlation should be forecast
        """
        return pd.Series([None], index=[self.data.index[-1] + n*BDay()], dtype=self.data[self.COVARIANCE].dtype)

    def get_volatility_forecast_for_next_n_days(self, n):
        """
        :param n: an integer indicating for how many business days in the future daily volatility should be forecast
        """
        return pd.concat([self.get_correlation_forecast(d) for d in range(1, n)])

    def get_annual_term_correlation_forecast(self, t):
        """
        :param t: a float indicating for which future term (expressed in years) average volatility needs to be forecast.

        This is a key method for pricing options.
        """
        raise NotImplementedError

class EWMACorrelationTracker(CorrelationTracker):
    """
    Represents an Exponentially-Weighted Moving Average volatility tracker with a given λ parameter
    """

    def __init__(self, lamda, asset_prices=None, start=None, end=None, assets=['^GSPC', 'AAPL']):
        super().__init__(asset_prices, start, end, assets)
        self.lamda = lamda

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = (1 - λ) * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                     + λ * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data)):
            self.data.iloc[i, 4] = (1 - lamda) * self.data.iloc[i-1, 2] * self.data.iloc[i-1, 3] \
                                               + lamda * self.data.iloc[i-1, 4]

    def get_next_business_day_correlation(self):
        s = super().get_next_business_day_correlation()
        last_idx = len(self.data) - 1
        s[0] = np.sqrt((1 - self.lamda) * self.data.iloc[last_idx, 2] * self.data.iloc[last_idx, 3]
                       + self.lamda * self.data.iloc[last_idx, 4])
        return s

    def get_correlation_forecast(self, n):
        """
        For EMWA the forecast for n business days in the future is the same as for the next business day
        """
        s = super().get_correlation_forecast(n)
        s[0] = self.get_next_business_day_correlation().values[0]
        return s

