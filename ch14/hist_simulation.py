# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import locale
import matplotlib.pyplot as plt
import datetime
from math import sqrt


class HistSimulation:
    """
    Represents the dataset for Chapter Market Risk VaR: The Historical Simulation Approach.
    """
    scale_factor = 1e3

    def __init__(self, path, today_value):
        data = pd.read_excel(path)
        data.index = data['Date']
        data = data.drop(['Date'], axis=1)
        data = data.drop(['FTSE-100', 'USD/GBP', 'CAC-40', 'EUR/USD', 'Nikkei', 'YEN/USD'], axis=1)
        data = data.drop(data.columns[0], axis=1)
        data.columns = ['DJIA', 'FTSE', 'CAC', 'Nikkei']
        self.data = data
        self.data_pct = data.pct_change().dropna()
        self.today_value = today_value
        self.today = data.iloc[-1]
        self.scenarios = self.data_pct * self.today + self.today
        self.weights = Series([4000, 3000, 1000, 2000], data.columns)
        self.weights_equal = self.weights.copy()
        self.weights_equal[:] = [2500, 2500, 2500, 2500]

    def exercise_14_4(self, VaR_conf):
        loss = (self.scenarios * self.weights / self.today).sum(axis=1)
        #portfolio_value=loss
        _scenarios = self.scenarios.copy(deep=False)
        _scenarios['Loss'] = self.today_value - loss
        # _scenarios.sort_values(by='Loss', ascending=False).iloc[24]
        # _scenarios.sort_values(by='Loss', ascending=False).iloc[14]
        for i in range(len(VaR_conf)):
            print("Exercise 14.4. One day VaR with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(_scenarios.Loss.quantile(VaR_conf[i]) * self.scale_factor,
                                                        grouping=True)))

    def exercise_14_5(self, VaR_conf):
        loss = (self.scenarios * self.weights_equal / self.today).sum(axis=1)
        _scenarios = self.scenarios.copy(deep=False)
        _scenarios['Loss'] = self.today_value - loss
        for i in range(len(VaR_conf)):
            print("Exercise 14.5. One day VaR with a confidence level of %2.1f%%: %s"
                % (VaR_conf[i] * 100, locale.currency(_scenarios.Loss.quantile(VaR_conf[i]) * self.scale_factor,
                                                      grouping=True)))
        # alternatively we could've just taken the fifth worst loss
        # _scenarios.Loss.sort_values(ascending=False)[4]

    def exercise_14_6(self, lbd):
        lbd_to_500 = 1 - pow(lbd, 500)
        _scenarios = self.scenarios.copy(deep=False)
        loss = (self.scenarios * self.weights / self.today).sum(axis=1)
        _scenarios['Loss'] = self.today_value - loss
        _scenarios['Weight'] = range(1, 501)
        _scenarios.Weight = _scenarios.Weight.map(lambda i: pow(lbd, 500 - i) * (1 - lbd) / lbd_to_500)
        _scenarios = _scenarios.sort_values(by='Loss', ascending=False)
        _scenarios['Cum_Sum'] = _scenarios.Weight.cumsum()
        print("Exercise 14.6. One day VaR with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(_scenarios[_scenarios.Cum_Sum > .01].Loss[0] * self.scale_factor, grouping=True)))

    def exercise_14_7(self, lbd):
        djia_var = self.data_pct.DJIA.var()
        ftse_var = self.data_pct.FTSE.var()
        cac_var  = self.data_pct.CAC.var()
        nikkei_var = self.data_pct.Nikkei.var()

        _data_pct = self.data_pct.copy(deep=False)
        _data_pct['DJIA_Var'] = djia_var
        _data_pct['FTSE_Var'] = ftse_var
        _data_pct['CAC_Var']  = cac_var
        _data_pct['Nikkei_Var'] = nikkei_var
        _data_pct = _data_pct[['DJIA', 'DJIA_Var', 'FTSE', 'FTSE_Var', 'CAC', 'CAC_Var', 'Nikkei', 'Nikkei_Var']]
        extra_row = DataFrame(np.zeros((1,8)), index=[_data_pct.index[len(_data_pct) - 1]+datetime.timedelta(days=1)],
                              columns=_data_pct.columns)
        _data_pct  = pd.concat([_data_pct, extra_row], axis=0)
        for i in range(1, len(self.data)):
            _data_pct.DJIA_Var[i] = lbd * _data_pct.DJIA_Var[i - 1] + (1 - lbd) * _data_pct.DJIA[i - 1] * _data_pct.DJIA[i - 1]
            _data_pct.FTSE_Var[i] = lbd * _data_pct.FTSE_Var[i - 1] + (1 - lbd) * _data_pct.FTSE[i - 1] * _data_pct.FTSE[i - 1]
            _data_pct.CAC_Var[i] = lbd * _data_pct.CAC_Var[i - 1] + (1 - lbd) * _data_pct.CAC[i - 1] * _data_pct.CAC[i - 1]
            _data_pct.Nikkei_Var[i] = lbd * _data_pct.Nikkei_Var[i - 1] + (1 - lbd) * _data_pct.Nikkei[i - 1] * _data_pct.Nikkei[i - 1]

        djia_var = _data_pct.DJIA_Var[len(self.data) - 1]
        ftse_var = _data_pct.FTSE_Var[len(self.data) - 1]
        cac_var = _data_pct.CAC_Var[len(self.data) - 1]
        nikkei_var = _data_pct.Nikkei_Var[len(self.data) - 1]

        _data_pct2 = pd.DataFrame(index=_data_pct.index, columns=self.data.columns)
        _data_pct2.drop(_data_pct2.index[-1])
        for i in range(1, len(self.data)):
            _data_pct2.DJIA[i - 1] = (self.data.DJIA[i - 1] + (self.data.DJIA[i] - self.data.DJIA[i - 1]) * sqrt(djia_var) / sqrt(
                _data_pct.DJIA_Var[i - 1])) / self.data.DJIA[i - 1]
            _data_pct2.FTSE[i - 1] = (self.data.FTSE[i - 1] + (self.data.FTSE[i] - self.data.FTSE[i - 1]) * sqrt(ftse_var) / sqrt(
                _data_pct.FTSE_Var[i - 1])) / self.data.FTSE[i - 1]
            _data_pct2.CAC[i - 1] = (self.data.CAC[i - 1] + (self.data.CAC[i] - self.data.CAC[i - 1]) * sqrt(cac_var) / sqrt(
                _data_pct.CAC_Var[i - 1])) / self.data.CAC[i - 1]
            _data_pct2.Nikkei[i - 1] = (self.data.Nikkei[i - 1] + (self.data.Nikkei[i] - self.data.Nikkei[i - 1]) * sqrt(nikkei_var) / sqrt(
                _data_pct.Nikkei_Var[i - 1])) / self.data.Nikkei[i - 1]
        _data_pct2 -= 1.

        _scenarios = _data_pct2 * self.today + self.today
        loss = (_scenarios * self.weights / self.today).sum(axis=1)
        _scenarios['Loss'] = self.today_value - loss
        print("Exercise 14.7. One day VaR with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(_scenarios.Loss.quantile(.99) * self.scale_factor, grouping=True)))

if __name__ == "__main__":
    import sys
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow kernel diagnostics

    try:
        # http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI3eHistoricalSimulation.xls
        # The first row is removed
        hist_simulation_spreadsheet = sys.argv[1]
        ch14 = HistSimulation(hist_simulation_spreadsheet, 1e4)
        locale.setlocale(locale.LC_ALL, '')
        ch14.exercise_14_4([.95, .97])
        ch14.exercise_14_5([.99])
        ch14.exercise_14_6(lbd=.99)
        ch14.exercise_14_7(lbd=.96)
        from evt import Evt
        evt = Evt(hist_simulation_spreadsheet, 1e4, num_iter=30000)
        evt.exercise_14_8_9(VaR_conf=.97)

        evt.exercise_14_10(VaR_conf=[.99, .999])
        evt.exercise_14_11(VaR_conf=[.99, .999])
    except (IndexError, ValueError) as ex:
        print(
        '''Invalid number of arguments or incorrect values. Usage:
    {0:s} <path-to-historical-simulation-spreadsheet> 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())


