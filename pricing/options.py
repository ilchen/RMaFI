from enum import Enum, unique
from math import sqrt, log, exp
from datetime import date

import pandas as pd
from scipy.stats import norm

from volatility import volatility_trackers
from pricing import curves


@unique
class OptionType(Enum):
    EUROPEAN = 0
    AMERICAN = 1


class OptionsPricer:
    """
    Pricer for options base class
    """

    UNSPECIFIED_TICKER = 'Unspecified'

    def __init__(self, maturity, volatility_tracker, strike, curve, cur_price, is_call=True,
                 opt_type=OptionType.EUROPEAN, ticker=UNSPECIFIED_TICKER):
        """
        Constructs a pricer for based on specified maturities and rates.

        :param maturity: a datetime.date specifying the day the options matures on
        :param volatility_tracker: a VolatilityTracker object
        :param strike: a numpy.float64 object representing the strike price of the option
        :param curve: a yield curve object representing a riskless discount curve to use
        :param cur_price: the current price of the asset
        :param is_call: indicates if we are pricing a call or a put option
        :param opt_type: the type of option -- European or American
        :param ticker: the ticker symbol of the underlying asset (optional)
        """
        self.maturity_date = maturity
        self.vol_tracker = volatility_tracker
        self.strike = strike
        self.riskless_yield_curve = curve
        self.s0 = cur_price
        self.is_call = is_call
        self.opt_type = opt_type
        self.ticker = ticker
        self.T = curve.to_years(self.maturity_date)
        if self.T == 0.:
            self.T = 8 / (24 * (366 if curves.YieldCurve.is_leap_year(self.maturity_date.year) else 365))
        self.r = self.riskless_yield_curve.to_continuous_compounding(
            self.riskless_yield_curve.get_yield_for_maturity_date(self.maturity_date))
        self.annual_volatility = self.vol_tracker.get_annual_term_volatility_forecast(self.T)

    def __str__(self):
        return '%s %s option with strike %s and maturity %s, price: %.2f, \u03c3: %.2f, '\
               '\u0394: %.3f, \u0393: %.3f, \u03Bd: %.3f'\
              % (self.ticker, 'call' if self.is_call else 'put', locale.currency(self.strike, grouping=True),
                 self.maturity_date.strftime('%Y-%m-%d'), self.get_price(), self.annual_volatility,
                 self.get_delta(), self.get_gamma(), self.get_vega())

    def get_price(self):
        raise NotImplementedError

    def get_delta(self):
        raise NotImplementedError

    def get_theta(self):
        raise NotImplementedError

    def get_gamma(self):
        raise NotImplementedError

    def get_vega(self):
        raise NotImplementedError

    def get_rho(self):
        raise NotImplementedError


class BlackScholesMertonPricer(OptionsPricer):

    def __init__(self, maturity, volatility_tracker, strike, curve, cur_price, is_call=True,
                 opt_type=OptionType.EUROPEAN, ticker=OptionsPricer.UNSPECIFIED_TICKER, dividends=None):
        """
        Constructs a pricer for based on specified maturities and rates.

        :param maturity: a datetime.date specifying the day the options matures on
        :param volatility_tracker: a VolatilityTracker object
        :param strike: a numpy.float64 object representing the strike price of the option
        :param curve: a yield curve object representing a riskless discount curve to use
        :param cur_price: the current price of the asset
        :param is_call: indicates if we are pricing a call or a put option
        :param opt_type: the type of option -- European or American
        :param ticker: the ticker symbol of the underlying asset (optional)
        :param dividends: if the asset pays dividends during the specified maturity period, specifies a pandas.Series
                          object indexed by DatetimeIndex specifying expected dividend payments
        """

        super().__init__(maturity, volatility_tracker, strike, curve, cur_price, is_call, opt_type, ticker)

        if self.opt_type == OptionType.AMERICAN and not self.is_call:
            raise NotImplementedError('Cannot price an American Put option using the Black-Scholes-Merton model')

        if dividends is None:
            self.divs = None
        else:
            assert isinstance(dividends.index, (pd.core.indexes.datetimes.DatetimeIndex, pd.core.indexes.base.Index))\
                and dividends.index.is_monotonic_increasing
            # Get rid of dividends not relevant for valuing this option
            self.divs = dividends.truncate(after=self.maturity_date)
            if self.divs.empty:
                self.divs = None

        # Initializing the rest of the fields
        self._init_fields()

    def _init_fields(self):
        """
        Constructor helper method to initialize additional fields
        """
        npv_divs = 0.  # net present value of all dividends
        if self.divs is not None:
            for d in self.divs.index:
                if not isinstance(d, date):
                    d = d.date()
                if d > self.maturity_date:  continue
                dcf = self.riskless_yield_curve.get_discount_factor_for_maturity_date(d)
                npv_divs += self.divs[d] * dcf

        adjusted_s0 = self.s0 - npv_divs
        vol_to_maturity = self.annual_volatility * sqrt(self.T)
        d1 = (log(adjusted_s0 / self.strike) + (self.r + self.annual_volatility ** 2 / 2.) * self.T) / vol_to_maturity
        d2 = d1 - vol_to_maturity

        self.early_exercise_pricer = None
        if self.is_call:
            call_price = adjusted_s0 * norm.cdf(d1) - self.strike * exp(-self.r * self.T) * norm.cdf(d2)
            if self.opt_type == OptionType.AMERICAN and self.divs is not None:
                # We need to check if it's optimal to exercise immediately before the ex-dividend date.
                # It's best to do that recursively
                pricer = BlackScholesMertonPricer((self.divs.index[-1] - BDay(1)).date(), self.vol_tracker, self.strike,
                        self.riskless_yield_curve, self.s0, opt_type=OptionType.AMERICAN,
                        dividends=self.divs.truncate(after=self.divs.index[-2]) if len(self.divs.index) > 1 else None)
                if pricer.get_price() > call_price:
                    self.early_exercise_pricer = pricer
            self.price = call_price
        else:
            self.price = self.strike * exp(-self.r * self.T) * norm.cdf(-d2) - adjusted_s0 * norm.cdf(-d1)
        self.adjusted_s0 = adjusted_s0
        self.d1 = d1
        self.d2 = d2

    def get_price(self):
        """
        Calculates the present value of the option
        """
        # In case we are dealing with an American call option where early exercise is advantageous
        return self.price if self.early_exercise_pricer is None else self.early_exercise_pricer.get_price()
    
    def get_delta(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_delta()
        return norm.cdf(self.d1) if self.is_call else norm.cdf(self.d1) - 1

    def get_theta(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_theta()
        summand = self.r * self.strike * exp(-self.r * self.T)
        if self.is_call:
            summand *= -norm.cdf(self.d2)
        else:
            summand *= norm.cdf(-self.d2)
        return -self.adjusted_s0 * norm.pdf(self.d1) * self.annual_volatility / (2 * sqrt(self.T)) + summand

    def get_gamma(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_gamma()
        return norm.pdf(self.d1) / (self.adjusted_s0 * self.annual_volatility * sqrt(self.T))

    def get_vega(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_vega()
        return self.adjusted_s0 * sqrt(self.T) * norm.pdf(self.d1)

    def get_rho(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_rho()
        multiplicand = self.strike * self.T * exp(-self.r * self.T)
        return multiplicand * (norm.cdf(self.d2) if self.is_call else -norm.cdf(-self.d2))


if __name__ == "__main__":
    import sys
    import locale
    from datetime import date, datetime

    from dateutil.relativedelta import relativedelta
    import pandas_datareader.data as web
    from pandas.tseries.offsets import BDay
    import numpy as np

    from volatility import parameter_estimators
    from pricing import curves

    TICKER = 'AAPL'

    try:
        locale.setlocale(locale.LC_ALL, '')
        start = date(2018, 1, 1)
        end = date.today()
        data = web.get_data_yahoo(TICKER, start, end)
        asset_prices = data['Adj Close']
        cur_date = asset_prices.index[-1].date()
        cur_price = asset_prices[-1]
        print('S0 of %s on %s:\t%.5f' % (TICKER, date.strftime(cur_date, '%Y-%m-%d'), cur_price))

        maturity_date = date(2022, month=10, day=21)

        # Constructing the riskless yield curve based on the current fed funds rate and treasury yields
        data = web.get_data_fred(
            ['DFF', 'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30'],
            end - BDay(2), end)
        data.dropna(inplace=True)

        cur_date_curve = data.index[-1].date()

        # Convert to percentage points
        data /= 100.

        # Some adjustments are required:
        # 1. https://www.federalreserve.gov/releases/h15/default.htm -> day count convention for Fed Funds Rate needs
        # to be changed to actual
        # 2. Conversion to APY: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions
        data.DFF *= (366 if curves.YieldCurve.is_leap_year(cur_date_curve.year) else 365) / 360 # to x/actual
        data.DFF = 2 * (np.sqrt(data.DFF + 1) - 1)

        offsets = [relativedelta(), relativedelta(months=+1), relativedelta(months=+3), relativedelta(months=+6),
                   relativedelta(years=+1), relativedelta(years=+2), relativedelta(years=+3), relativedelta(years=+5),
                   relativedelta(years=+7), relativedelta(years=+10), relativedelta(years=+20),
                   relativedelta(years=+30)]

        # Define yield curves
        curve = curves.YieldCurve(cur_date, offsets, data[cur_date_curve:cur_date_curve + BDay()].to_numpy()[0, :],
                                  compounding_freq=2)

        # Obtaining a volatility estimate for maturity
        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        #
        # print('Optimal values for GARCH(1, 1) parameters:\n\tω=%.12f, α=%.5f, β=%.5f'
        #       % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        #
        # vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                          vol_estimator.beta, asset_prices)
        vol_tracker = volatility_trackers.GARCHVolatilityTracker(0.000021656157, 0.12512, 0.82521, asset_prices)

        # vol_estimator = parameter_estimators.EWMAParameterEstimator(asset_prices)
        # vol_tracker = volatility_trackers.EWMAVolatilityTracker(vol_estimator.lamda, asset_prices)
        # Volatility of AAPL for term 0.2137: 0.32047

        vol = vol_tracker.get_annual_term_volatility_forecast(curve.to_years(maturity_date))
        # vol = 0.3134162504522202 (1 Aug 2022)
        # vol = 0.29448 # (2 Aug 2022)
        print('Volatility of %s for term %.4f: %.5f' % (TICKER, curve.to_years(maturity_date), vol))

        strike = 180.

        # An approximate rule for Apple's dividends
        idx = (pd.date_range(date(2022, 8, 8), freq='WOM-2MON', periods=20)[::3] - BDay(1))
        divs = pd.Series([.23] * len(idx), index=idx, name=TICKER + ' Dividends')

        pricer = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                          ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)

        pricer_put = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price, is_call=False,
                                              ticker=TICKER, dividends=divs)
        print(pricer_put)

    # except (IndexError, ValueError) as ex:
    #     print(
    #         '''Invalid number of arguments or incorrect values. Usage:
    # {0:s}
    #             '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())

