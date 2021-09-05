# coding: utf-8
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

class ParameterEstimator:
    """
    Represents an estimator for volatility forecasting parameters
    """

    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    VARIANCE = 'Variance'

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        '''
        Constructs a volatility estimator object from either a panda series object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a desired asset class (i.e. the 'start' abd 'end'
        arguments must be provided)
        :param asset_proce_series: a pandas Series object indexed by dates
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param asset: the ticker simbol of the asset whose asset price changes are to be analyzed. It expects
                      a Yahoo Finance convention for ticker simbolds
        '''
        if asset_prices_series is None:
            if start is None or end is None or asset is None:
                raise ValueError("Neither asset_price_series nor (start, end, asset) arguments are provided")
            data = web.get_data_yahoo(asset, start, end)
            asset_prices_series = data['Adj Close']
        # Dropping the first row as it doesn't contain a daily return value
        self.data = pd.DataFrame({self.CLOSE : asset_prices_series, self.DAILY_RETURN : asset_prices_series.pct_change()},
                                 index=asset_prices_series.index).iloc[1:]
        self.data[self.VARIANCE] = self.data.ui**2

class GARCHParameterEstimator(ParameterEstimator):
    """
    Represents an estimator for the GARCH(1, 1) method of forecasting volatility
    """
    # Ensuring that ω, α, and β values we will search for have roughly equal values in terms of magnitude
    GARCH_PARAM_MULTIPLIERS = np.array([1e5, 10, 1], dtype=np.float64)

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)

        # Initial values for ω, α, and β parameters for GARCH
        x0 = np.array([1e-8, 1e-3, 1e-1], dtype=np.float64)

        def objective_func(x):
            ''' This function searches for optimal values of the ω, α, and β parameters of the GARCH(1, 1)
            model given the sample of asset price changes stored in the 'self.data' DataFrame. Since SciPy only has
            optimization routines that minimize an objective function, this function returns negates the value of the
            log likelihood objective function for GARCH.
            :param x: a tuple of the ω, α, and β parameters where ω, α, and β
                      should be appropriately scaled to be in approximately the same range. This greatly aids the
                      speed of optimization
            '''
            ω, α, β = x / GARCHParameterEstimator.GARCH_PARAM_MULTIPLIERS

            # Unfortunately not vectorizable as the next value depends on the previous
            # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
            #                                       + β * self.data[self.VARIANCE].iloc[:-1]
            for i in range(1, len(self.data[self.DAILY_RETURN])):
                self.data[self.VARIANCE].iloc[i] = ω + α * self.data[self.DAILY_RETURN].iloc[i-1]**2 \
                                                     + β * self.data[self.VARIANCE].iloc[i-1]
            return -(-np.log(self.data[self.VARIANCE]) - self.data[self.DAILY_RETURN] ** 2 / self.data[self.VARIANCE]).sum()

        # print('Starting with objective function value of:', -objective_func(x0 * self.GARCH_PARAM_MULTIPLIERS))

        # omega [0; np.inf], alpha[0;1], beta[0;1]
        bounds = Bounds([0., 0., 0.], np.array([np.inf, 1., 1.]) * self.GARCH_PARAM_MULTIPLIERS)

        # 0*omega + alpha + beta <= 1
        constr = LinearConstraint([[0, 1 / self.GARCH_PARAM_MULTIPLIERS[1], 1 / self.GARCH_PARAM_MULTIPLIERS[2]]], [0], [1])

        constr2 = [{'type':'ineq', 'fun': lambda x:
            1-x[1]/self.GARCH_PARAM_MULTIPLIERS[1]-x[2]/self.GARCH_PARAM_MULTIPLIERS[2] }]

        res = minimize(objective_func, x0 * self.GARCH_PARAM_MULTIPLIERS, method='trust-constr',
                       bounds=bounds, constraints=constr) #, options={'maxiter': 150, 'verbose': 2})
        if res.success:
            ω, α, β = res.x / self.GARCH_PARAM_MULTIPLIERS
            # print('Optimal values for GARCH parameters:\n\tω=%.12f, α=%.5f, β=%.5f' % (ω, α, β))
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nit))
            self.omega = ω
            self.alpha = α
            self.beta = β
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")

class EWMAParameterEstimator(ParameterEstimator):
    """
    Represents an estimator for the EWMA method of forecasting volatility
    """
    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)

        # Initial value of λ for EWMA
        λ = .9

        def objective_func(λ):
            ''' This function searches for optimal values of the λ parameter of the EWMA
            model given the sample of asset price changes stored in the 'self.data' DataFrame. Since SciPy only has
            optimization routines that minimize an objective function, this function returns negates the value of the
            log likelihood objective function for EWMA.
            :param λ: the λ parameter in EWMA method of estimating volatility
            '''

            # Unfortunately not vectorizable as the next value depends on the previous
            # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
            #                                       + β * self.data[self.VARIANCE].iloc[:-1]
            for i in range(1, len(self.data[self.DAILY_RETURN])):
                self.data[self.VARIANCE].iloc[i] = (1 - λ) * self.data[self.DAILY_RETURN].iloc[i-1]**2 \
                                                   + λ * self.data[self.VARIANCE].iloc[i-1]
            return -(-np.log(self.data[self.VARIANCE]) - self.data[self.DAILY_RETURN] ** 2 / self.data[self.VARIANCE]).sum()

        # print('Starting with objective function value of:', -objective_func(λ))
        res = minimize_scalar(objective_func, bounds=(0,1), method='bounded')

        if res.success:
            # print('Optimal value for λ: %.5f' % res.x)
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nfev))
            self.lamda = res.x
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")

if __name__ == "__main__":
    import sys
    import os

    try:
        start = datetime.datetime(2005, 7, 27)
        end = datetime.datetime(2010, 7, 27)
        data = web.get_data_yahoo('EURUSD=X', start, end)
        asset_prices_series = data['Adj Close']
        ch10_ewma = EWMAParameterEstimator(asset_prices_series)
        ch10_garch = GARCHParameterEstimator(asset_prices_series)

    except (IndexError, ValueError) as ex:
        print(
        '''Invalid number of arguments or incorrect values. Usage:
    {0:s} <path-to-historical-simulation-spreadsheet> 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())