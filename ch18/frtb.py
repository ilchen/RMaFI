# coding: utf-8
from pandas import Series, DataFrame
import pandas as pd
import locale
from math import sqrt


class FRTB:
    """
    Represents the dataset for Chapter Fundamental Review of the Trading Book.
    """
    scale_factor = 1e3
    columns = ['DJIA', 'FTSE', 'CAC', 'Nikkei']
    columns_20_days_time_horizon = ['CAC', 'Nikkei']
    columns_40_days_time_horizon = ['CAC']

    def __init__(self, path, today_values, today_indices):
        data = pd.read_excel(path)
        data.index = data['Date']
        data = data.drop(['Date'], axis=1)
        data = data.drop(['FTSE-100', 'USD/GBP', 'CAC-40', 'EUR/USD', 'Nikkei', 'YEN/USD'], axis=1)
        data = data.drop(data.columns[0], axis=1)
        data.columns = FRTB.columns
        self.data = data
        self.data_pct = data.pct_change(periods=10).dropna()
        self.today_values = today_values
        self.today_indices = today_indices #data.iloc[-1]
        idx = self.data_pct.index.get_loc('2008-9-9')
        # Exactly 250 scenarios
        self.scenarios = self.data_pct.iloc[idx-249:idx+1] * self.today_indices + self.today_indices
        self.scenarios_20 = self.data_pct[FRTB.columns_20_days_time_horizon].iloc[idx-249:idx+1] * self.today_indices + self.today_indices
        self.scenarios_40 = self.data_pct[FRTB.columns_40_days_time_horizon].iloc[idx-249:idx+1] * self.today_indices + self.today_indices

    def exercise_18_7(self, VaR_conf):
        portfolio_values_in_scenarios = (self.scenarios * self.today_values / self.today_indices).sum(axis=1)
        portfolio_values_in_scenarios_20 = (self.scenarios_20 * self.today_values / self.today_indices).sum(axis=1)
        portfolio_values_in_scenarios_40 = (self.scenarios_40 * self.today_values / self.today_indices).sum(axis=1)

        loss = self.today_values.sum() - portfolio_values_in_scenarios
        quantile = loss.quantile(VaR_conf)
        ES1 = loss[loss > quantile].mean()
        # Alternatively, use the mean of the highest 6 losses to get the same results as in John C. Hull's answers
        # when VaR_conf == .975
        # loss = loss.sort_values(ascending=False)
        # ES1 = loss.iloc[:6].mean()

        loss_20 = self.today_values[FRTB.columns_20_days_time_horizon].sum() - portfolio_values_in_scenarios_20
        quantile_20 = loss_20.quantile(VaR_conf)
        ES2 = loss_20[loss_20 > quantile_20].mean()
        # loss_20 = loss_20.sort_values(ascending=False)
        # ES2 = loss_20.iloc[:6].mean()

        loss_40 = self.today_values[FRTB.columns_40_days_time_horizon].sum() - portfolio_values_in_scenarios_40
        quantile_40 = loss_40.quantile(VaR_conf)
        ES3 = loss_40[loss_40 > quantile_40].mean()
        # loss_40 = loss_40.sort_values(ascending=False)
        # ES3 = loss_40.iloc[:6].mean()

        ES = sqrt(ES1**2 + ES2**2 + 2 * ES3**2)

        print("Exercise 18.8. %2.1f%% FRTB expected shortfall: %s"
              % (VaR_conf * 100, locale.currency(ES * self.scale_factor, grouping=True)))
        print("\tES1: %s\n\tES2: %s\n\tES3: %s"
              % (locale.currency(ES1 * self.scale_factor, grouping=True),
                 locale.currency(ES2 * self.scale_factor, grouping=True),
                 locale.currency(ES3 * self.scale_factor, grouping=True)))

if __name__ == "__main__":
    import sys
    import os
    from datetime import date

    try:
        # http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI3eHistoricalSimulation.xls
        # The first row is removed
        hist_simulation_spreadsheet = sys.argv[1]
        today = date.fromisoformat('2014-09-30')
        today_indices = Series([17042.90, 6622.7, 4416.24, 16173.52], FRTB.columns)
        fx_rates = Series([1., 1.6211, 1 / .7917, 1 / 109.64], FRTB.columns)
        today_indices *= fx_rates
        today_values = Series([4000, 3000, 1000, 2000], FRTB.columns)
        ch18 = FRTB(hist_simulation_spreadsheet, today_values, today_indices)
        locale.setlocale(locale.LC_ALL, '')
        ch18.exercise_18_7(.975)

    except (IndexError, ValueError) as ex:
        print(
        '''Invalid number of arguments or incorrect values. Usage:
    {0:s} <path-to-historical-simulation-spreadsheet> 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())


