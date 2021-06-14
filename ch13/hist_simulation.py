# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import locale
import datetime
from math import sqrt
import matplotlib.pyplot as plt

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

    def exercise_13_4(self, VaR_conf):
        loss = self.today_value - (self.scenarios * self.weights / self.today).sum(axis=1)
        # _scenarios.sort_values(by='Loss', ascending=False).iloc[24]
        # _scenarios.sort_values(by='Loss', ascending=False).iloc[14]

        for i in range(len(VaR_conf)):
            var = loss.quantile(VaR_conf[i])
            es = loss[loss > var].mean()
            print("Exercise 13.4. One day VaR with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(var * self.scale_factor, grouping=True)))
            print("               One day ES with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(es * self.scale_factor, grouping=True)))

    def exercise_13_5(self, VaR_conf):
        loss = self.today_value - (self.scenarios * self.weights_equal / self.today).sum(axis=1)
        for i in range(len(VaR_conf)):
            var = loss.quantile(VaR_conf[i])
            es = loss[loss > var].mean()
            print("Exercise 13.5. One day VaR with a confidence level of %2.1f%%: %s"
                % (VaR_conf[i] * 100, locale.currency(var * self.scale_factor, grouping=True)))
            print("Exercise 13.5. One day ES with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(es * self.scale_factor, grouping=True)))
        # alternatively we could've just taken the fifth worst loss
        # loss.sort_values(ascending=False)[4]

    def exercise_13_6(self, lbd):
        VaR_conf = .99
        conf = 1 - VaR_conf
        lbd_to_500 = 1 - pow(lbd, 500)
        _scenarios = self.scenarios.copy(deep=False)
        _scenarios['Loss'] = self.today_value - (self.scenarios * self.weights / self.today).sum(axis=1)
        _scenarios['Weight'] = range(1, 501)
        _scenarios.Weight = _scenarios.Weight.map(lambda i: pow(lbd, 500 - i) * (1 - lbd) / lbd_to_500)
        _scenarios = _scenarios.sort_values(by='Loss', ascending=False)
        _scenarios['Cum_Sum'] = _scenarios.Weight.cumsum()

        # Calculating VaR
        var = _scenarios[_scenarios.Cum_Sum > conf].Loss[0]

        # Only interested in the range of losses whose probability is less than 1%
        _scenarios = _scenarios[_scenarios.Cum_Sum <= conf]
        weight = conf - _scenarios.Weight.sum()
        es = ((_scenarios.Loss * _scenarios.Weight).sum() + var * weight) / conf
        print("Exercise 13.6. One day VaR with a confidence level of %2.1f%% and \u03BB=%1.2f: %s"
              % (VaR_conf * 100, lbd, locale.currency(var * self.scale_factor, grouping=True)))
        print("Exercise 13.6. One day ES with a confidence level of %2.1f%% and \u03BB=%1.2f: %s"
              % (VaR_conf * 100, lbd, locale.currency(es * self.scale_factor, grouping=True)))

    def exercise_13_7_old(self, lbd):
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

        # Adding an extra row to calculate volatilities on the day after the last one in the sample -- 26th Sept 2008
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

        # The last row for 26th September is not needed anymore
        _data_pct2.drop(_data_pct2.index[-1], inplace=True)

        # Applying formula 13.2
        for i in range(1, len(self.data)):
            _data_pct2.DJIA[i - 1] = (self.data.DJIA[i - 1] + (self.data.DJIA[i] - self.data.DJIA[i - 1]) * sqrt(djia_var) / sqrt(
                _data_pct.DJIA_Var[i - 1])) / self.data.DJIA[i - 1]
            _data_pct2.FTSE[i - 1] = (self.data.FTSE[i - 1] + (self.data.FTSE[i] - self.data.FTSE[i - 1]) * sqrt(ftse_var) / sqrt(
                _data_pct.FTSE_Var[i - 1])) / self.data.FTSE[i - 1]
            _data_pct2.CAC[i - 1] = (self.data.CAC[i - 1] + (self.data.CAC[i] - self.data.CAC[i - 1]) * sqrt(cac_var) / sqrt(
                _data_pct.CAC_Var[i - 1])) / self.data.CAC[i - 1]
            _data_pct2.Nikkei[i - 1] = (self.data.Nikkei[i - 1] + (self.data.Nikkei[i] - self.data.Nikkei[i - 1]) * sqrt(nikkei_var) / sqrt(
                _data_pct.Nikkei_Var[i - 1])) / self.data.Nikkei[i - 1]
        _scenarios = _data_pct2 * self.today # Scenarios with volatility scaling for market variables

        portfolio_value_under_scenarios = (_scenarios * self.weights / self.today).sum(axis=1)
        loss = self.today_value - portfolio_value_under_scenarios

        loss.sort_values(ascending=False, inplace=True)
        var = loss[4]
        es = loss[:4].mean()

        # Better approach that will work when the confidence level is parameterized
        # var = loss.quantile(.99)
        # es = loss[loss > var].mean()

        print("Exercise 13.7 old. One day VaR with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(var * self.scale_factor, grouping=True)))
        print("Exercise 13.7 old. One day ES with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(es * self.scale_factor, grouping=True)))

        # Visualizing losses
        x = np.arange(len(loss))
        plt.plot(x, loss, label="Losses", linewidth=2, color='g')
        plt.legend(loc='best')
        plt.title('Sorted portfolio losses (using volatility scaling for market variables)')
        plt.ylabel('Portfolio losses in $M')
        plt.xlabel('Sorted scenarios')
        plt.show()

    def exercise_13_7(self, lbd):
        loss = self.today_value - (self.scenarios * self.weights / self.today).sum(axis=1)
        loss_variance = Series(loss.var(), loss.index)

        # Unfortunately not vectorizable as the next value depends on the previous
        # loss_variance.iloc[1:] = lbd * loss_variance.iloc[:-1] + (1 - lbd) * loss.iloc[:-1]**2
        for i in range(1, len(loss_variance)):
            loss_variance.iloc[i] = lbd * loss_variance[i - 1] + (1 - lbd) * loss[i - 1]**2

        loss_std = loss_variance.apply(np.sqrt)
        sd_ratio = loss_std[-1] / loss_std

        # Adjusting the loss for the volatility of the portfolio
        adjusted_loss = loss * sd_ratio

        adjusted_loss.sort_values(ascending=False, inplace=True)
        var = adjusted_loss[4]
        es = adjusted_loss[:4].mean()

        # Better approach that will work when the confidence level is parameterized, however produced answers slightly
        # disagree from those in the textbook
        # var = loss.quantile(.99)
        # es = loss[loss > var].mean()

        print("Exercise 13.7. One day VaR with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(var * self.scale_factor, grouping=True)))
        print("Exercise 13.7. One day ES with a confidence level of 99.0%% and \u03BB=%1.2f: %s"
              % (lbd, locale.currency(es * self.scale_factor, grouping=True)))

        # Visualizing losses
        x = np.arange(len(loss))
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, loss.sort_values(ascending=False), label="Losses", linewidth=1, color='b')
        ax.plot(x, adjusted_loss, label="Adjusted losses", linewidth=2, color='g')
        ax.legend(loc='best')
        ax.grid(True)
        ax.set_xticks(np.arange(0, len(loss_variance)+1, 20))
        ax.set_title('Sorted portfolio losses and adjusted losses with volatility scaling for the portfolio')
        ax.set_ylabel('Portfolio losses in $M')
        ax.set_xlabel('Sorted scenarios')


if __name__ == "__main__":
    import sys
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow kernel diagnostics

    try:
        # http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI3eHistoricalSimulation.xls
        # The first row is removed
        hist_simulation_spreadsheet = sys.argv[1]
        ch13 = HistSimulation(hist_simulation_spreadsheet, 1e4)
        locale.setlocale(locale.LC_ALL, '')
        ch13.exercise_13_4([.95, .97])
        ch13.exercise_13_5([.99])
        ch13.exercise_13_6(lbd=.99)
        ch13.exercise_13_7_old(lbd=.96)
        ch13.exercise_13_7(lbd=.96)
        from evt import Evt
        evt = Evt(hist_simulation_spreadsheet, 1e4, num_iter=30000)
        evt.exercise_13_8_9(VaR_conf=.97)

        evt.exercise_13_10(VaR_conf=[.99, .999])
        evt.exercise_13_11(VaR_conf=[.99, .999])
    except (IndexError, ValueError) as ex:
        print(
        '''Invalid number of arguments or incorrect values. Usage:
    {0:s} <path-to-historical-simulation-spreadsheet> 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())


