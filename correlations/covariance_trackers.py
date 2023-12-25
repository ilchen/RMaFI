# coding: utf-8
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas.tseries.offsets import BDay


class CovarianceTracker:
    """
    Represents a tracker for covariance and for forecasting future covariance between given assets.
    You instantiate an object of this class with a consecutive range of daily closing prices of given assets,
    and then use it to inspect past covariances as well as to predict future ones.
    """
    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    COVARIANCE = 'Covariance'

    def __init__(self, asset_prices=None, start=None, end=None, assets=None):
        """
        Calculates daily covariances from either a panda series object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a desired asset class (i.e. the 'start' abd 'end'
        arguments must be provided)
        :param asset_prices: a pandas DataFrame object indexed by dates containing two columns each representing closing
                             prices for a given asset
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param assets: a list containing two ticker symbols of the assets whose asset price changes are to be analyzed
                       to establish a covariance between them. It expects a Yahoo Finance convention for ticker symbols
        """
        if assets is None:
            assets = ['^GSPC', 'AAPL']
        if asset_prices is None:
            if start is None or end is None or assets is None or len(assets) != 2:
                raise ValueError("Neither asset_prices nor (start, end, assets) arguments are provided")
            data = web.get_data_yahoo(assets, start, end)
            self.data = data['Adj Close']
        elif len(asset_prices.columns) != 2:
            raise ValueError("Wrong number of columns in submitted DataFrame")
        else:
            self.data = asset_prices.copy()

        self.data.dropna(inplace=True)

        # Daily percentage changes in the first asset
        self.data.insert(loc=2, column=self.data.columns[0] + self.DAILY_RETURN,
                         value=self.data.iloc[:, 0].pct_change())
        # Daily percentage changes in the second asset
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

    def get_covariances(self):
        """
        Calculates the past covariances for consecutive range of dates captured in self.data.index
        """
        return self.data.iloc[1:, 4]

    def get_next_business_day_covariance(self):
        """
        Returns the covariance on the next business day after the last day in the self.data.index
        """
        return pd.Series([None], index=[self.data.index[-1] + BDay()], dtype=self.data[self.COVARIANCE].dtype)

    def get_covariance_forecast(self, n):
        """
        :param n: an integer indicating for which business day in the future covariance should be forecast
        """
        return pd.Series([None], index=[self.data.index[-1] + n*BDay()], dtype=self.data[self.COVARIANCE].dtype)

    def get_covariance_forecast_for_next_n_days(self, n):
        """
        :param n: an integer indicating for how many business days in the future covariance should be forecast
        """
        return pd.concat([self.get_covariance_forecast(d) for d in range(1, n)])


class EWMACovarianceTracker(CovarianceTracker):
    """
    Represents an Exponentially-Weighted Moving Average covariance tracker with a given λ parameter
    """

    def __init__(self, lamda, asset_prices=None, start=None, end=None, assets=['^GSPC', 'AAPL']):
        super().__init__(asset_prices, start, end, assets)
        self.lamda = lamda

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = (1 - λ) * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                     + λ * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data)):
            if np.isinf(self.data.iloc[i-1, 2]) or np.isinf(self.data.iloc[i-1, 3]):
                self.data.iloc[i, 4] = self.data.iloc[i-1, 4]
            else:
                self.data.iloc[i, 4] = (1 - lamda) * self.data.iloc[i-1, 2] * self.data.iloc[i-1, 3] \
                                                   + lamda * self.data.iloc[i-1, 4]

    def get_next_business_day_covariance(self):
        s = super().get_next_business_day_covariance()
        last_idx = len(self.data) - 1
        s[0] = (1 - self.lamda) * self.data.iloc[last_idx, 2] * self.data.iloc[last_idx, 3] \
            + self.lamda * self.data.iloc[last_idx, 4]
        return s

    def get_covariance_forecast(self, n):
        """
        For EMWA the forecast for n business days in the future is the same as for the next business day
        """
        s = super().get_covariance_forecast(n)
        s[0] = self.get_covariance_forecast(n).values[0]
        return s


class GARCHCovarianceTracker(CovarianceTracker):
    """
    Represents a GARCH(1, 1) covariane tracker with given ω, α, and β parameters
    """

    def __init__(self, omega, alpha, beta, asset_prices=None, start=None, end=None, assets=['^GSPC', 'AAPL']):
        super().__init__(asset_prices, start, end, assets)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.vl = self.omega / (1 - self.alpha - self.beta)

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                       + β * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data)):
            self.data.iloc[i, 4] = omega + alpha * self.data.iloc[i-1, 2] * self.data.iloc[i-1, 3] \
                                               + beta * self.data.iloc[i-1, 4]

    def get_next_business_day_covariance(self):
        s = super().get_next_business_day_covariance()
        last_idx = len(self.data) - 1
        s[0] = self.omega + self.alpha * self.data.iloc[last_idx, 2] * self.data.iloc[last_idx, 3] \
            + self.beta * self.data.iloc[last_idx, 4]
        return s

    def get_covariance_forecast(self, n):
        s = super().get_covariance_forecast(n)
        next_bd_cov = self.get_next_business_day_covariance().values[0]
        s[0] = self.vl + (self.alpha + self.beta)**n * (next_bd_cov - self.vl)
        return s

    def get_long_term_covariance(self):
        return self.vl
