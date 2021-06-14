# coding: utf-8
from pandas import Series
import pandas as pd
from scipy.stats import norm
import locale
import math

class ModelBuilding:
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
        self.weights = Series([4000, 3000, 1000, 2000], data.columns)
        self.weights_equal = self.weights.copy()
        self.weights_equal[:] = [2500, 2500, 2500, 2500]

    def exercise_14_14a(self, VaR_conf):
        n = self.data_pct.shape[0]
        # cov_matrix = DataFrame(np.cov(data_pct, bias=True, rowvar=False),
        #                        columns=data_pct.columns, index=data_pct.columns)
        cov_matrix = self.data_pct.cov() * (n - 1) / n
        p_sd = math.sqrt(self.weights_equal.dot(cov_matrix).dot(self.weights_equal))
        for i in range(len(VaR_conf)):
            print("Exercise 14.14a. One day VaR with a confidence level of %2.1f%%: %s"
                % (VaR_conf[i] * 100, locale.currency(norm.ppf(.99) * p_sd * self.scale_factor, grouping=True)))
            print("Exercise 14.14a. One day ES with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(ModelBuilding.es(p_sd, VaR_conf[i]) * self.scale_factor, grouping=True)))

    def exercise_14_14b(self, VaR_conf, weights, lbd=.94):
        cov_matrix = self.data_pct.ewm(alpha = 1. - lbd).cov(bias=True)
        last_idx = cov_matrix.index.levels[0][-1]
        cov_matrix = cov_matrix.loc[last_idx]
        p_sd = math.sqrt(weights.dot(cov_matrix).dot(weights))
        for i in range(len(VaR_conf)):
            print("Exercise 14.14b. One day VaR with a confidence level of %2.1f%% and \u03BB=%1.2f: %s"
                % (VaR_conf[i] * 100, lbd, locale.currency(norm.ppf(.99) * p_sd * self.scale_factor, grouping=True)))
            print("Exercise 14.14b. One day ES with a confidence level of %2.1f%% and \u03BB=%1.2f: %s"
                  % (VaR_conf[i] * 100, lbd, locale.currency(ModelBuilding.es(p_sd, VaR_conf[i]) * self.scale_factor, grouping=True)))

    @staticmethod
    def es(sd, confidence=.99):
        return math.pow(math.e, -(norm.ppf(confidence)**2/2)) * sd / (math.sqrt(2*math.pi) * (1 - confidence))

if __name__ == "__main__":
    import sys
    import os

    try:
        # http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI3eHistoricalSimulation.xls
        # The first row is removed.
        # The following code replicates all the calculations that are carried out in
        # http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI3eModelBuilding.xls
        hist_simulation_spreadsheet = sys.argv[1]
        ch14 = ModelBuilding(hist_simulation_spreadsheet, 1e4)
        locale.setlocale(locale.LC_ALL, '')
        ch14.exercise_14_14a([.99])
        ch14.exercise_14_14b([.99], ch14.weights_equal)
        ch14.exercise_14_14b([.99], ch14.weights, lbd=.97)
    except (IndexError, ValueError) as ex:
        print(
        '''Invalid number of arguments or incorrect values. Usage:
    {0:s} <path-to-historical-simulation-spreadsheet> 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())
