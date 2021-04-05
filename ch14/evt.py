# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import locale
from math import sqrt
#import pandas.io.data as web


class Evt:
    """
    Represents the dataset for Chapter Market Risk VaR: The Historical Simulation Approach.
    
    This class contains solutions for exercises relying on *Extreme Value Theory*. Since these exercises involve
    solving optimization problems, the class depends on TensorFlow -- the leading machine learning framework.
    """
    scale_factor = 1e3

    def __init__(self, path, today_value, num_iter):
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
        loss = (self.scenarios * self.weights / self.today).sum(axis=1)
        self.scenarios['Loss'] = today_value - loss
        self.num_iter = num_iter

    def exercise_14_8_9(self, threashold = 400, VaR_conf=.97):
        sorted_scenarios = self.scenarios.sort_values(by='Loss', ascending=False)

        beta = tf.Variable(40, dtype=tf.float64)
        gamma = tf.Variable(.3, dtype=tf.float64)
        u = 160.
        next_index = sorted_scenarios[sorted_scenarios.Loss <= u].index[0]
        n_u = sorted_scenarios.index.get_loc(next_index)
        n = len(sorted_scenarios)

        X   = tf.constant(sorted_scenarios.Loss.iloc[:n_u])
        # 'None' because I will change the shape in Exercise 14.10
        # X = tf.placeholder(dtype=tf.float64, shape=[None, ])
        #X = tf.keras.Input(dtype=tf.float64, name='X', shape=[None, ])

        # Changing the sign due to tensorflow's optimizers not having a 'maximize' method
        probability_ln = lambda: tf.reduce_sum(-tf.math.log((1. / beta) * tf.pow(1. + gamma * (X - u) / beta, -1. / gamma - 1.)))

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.05,
                                                                       decay_steps=self.num_iter*1000, decay_rate=.96)
        print("Cost initially: %.7f" % probability_ln())
        print("Starting values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        opt = tf.keras.optimizers.Adam(learning_rate)

        for _ in range(self.num_iter):  opt.minimize(probability_ln, [beta, gamma])
        print("Cost after running optimizer for %d iterations: %.7f"
               % (self.num_iter, probability_ln()))
        print("New values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        ex_14_8 = n_u / n * (1. + gamma * (threashold - u) / beta) ** (-1. / gamma)
        print("Exercise 14.8. The probability the loss will exceed $%d thousand: %.7f" % (threashold, ex_14_8))

        # Exercise 14.9
        ex_14_9 = u + beta / gamma * ((n / n_u * (1. - VaR_conf)) ** -gamma - 1)
        print("Exercise 14.9. One day VaR with a confidence level of %2.f%%: %.7f\n"
              % (VaR_conf * self.scale_factor, ex_14_9))

    def exercise_14_10(self, VaR_conf):
        sorted_scenarios = self.scenarios.sort_values(by='Loss', ascending=False)

        beta = tf.Variable(40, dtype=tf.float64)
        gamma = tf.Variable(.3, dtype=tf.float64)
        u = 150.
        next_index = sorted_scenarios[sorted_scenarios.Loss <= u].index[0]
        n_u = sorted_scenarios.index.get_loc(next_index)
        n = len(sorted_scenarios)

        X = tf.constant(sorted_scenarios.Loss.iloc[:n_u])

        # Changing the sign due to Tensorflow's optimizers not having a 'maximize' method
        probability_ln = lambda: tf.reduce_sum(-tf.math.log((1. / beta) * tf.pow(1. + gamma * (X - u) / beta, -1. / gamma - 1.)))
        print("Cost initially: %.7f" % probability_ln())
        print("Starting values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        opt = tf.keras.optimizers.Adam()

        for _ in range(self.num_iter):  opt.minimize(probability_ln, [beta, gamma])

        print("Cost after running optimizer for %d iterations: %.7f" % (self.num_iter, probability_ln()))
        print("New values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        # Exercise 14.10
        ex_14_10 = u + beta / gamma * ((n / n_u * (1. - np.array(VaR_conf))) ** -gamma - 1)
        for i in range(len(VaR_conf)):
            print("Exercise 14.10. One day VaR with a confidence level of %2.1f%%: %.7f"
                  % (VaR_conf[i] * 100, ex_14_10[i]))
        print()

    def exercise_14_11(self, VaR_conf, threashold = 600, lbd = .94):
        djia_var = self.data_pct.DJIA.var()
        ftse_var = self.data_pct.FTSE.var()
        cac_var = self.data_pct.CAC.var()
        nikkei_var = self.data_pct.Nikkei.var()
        _data_pct = self.data_pct.copy(deep=False)
        _data_pct['DJIA_Var'] = djia_var
        _data_pct['FTSE_Var'] = ftse_var
        _data_pct['CAC_Var'] = cac_var
        _data_pct['Nikkei_Var'] = nikkei_var
        _data_pct = _data_pct[['DJIA', 'DJIA_Var', 'FTSE', 'FTSE_Var', 'CAC', 'CAC_Var', 'Nikkei', 'Nikkei_Var']]
        extra_row = DataFrame(np.zeros((1, 8)), index=[_data_pct.index[len(_data_pct) - 1] + datetime.timedelta(days=1)],
                              columns=_data_pct.columns)
        _data_pct = pd.concat([_data_pct, extra_row], axis=0)
        for i in range(1, len(self.data)):
            _data_pct.DJIA_Var[i] = lbd * _data_pct.DJIA_Var[i - 1] + (1 - lbd) * _data_pct.DJIA[i - 1] * _data_pct.DJIA[
                i - 1]
            _data_pct.FTSE_Var[i] = lbd * _data_pct.FTSE_Var[i - 1] + (1 - lbd) * _data_pct.FTSE[i - 1] * _data_pct.FTSE[
                i - 1]
            _data_pct.CAC_Var[i] = lbd * _data_pct.CAC_Var[i - 1] + (1 - lbd) * _data_pct.CAC[i - 1] * _data_pct.CAC[i - 1]
            _data_pct.Nikkei_Var[i] = lbd * _data_pct.Nikkei_Var[i - 1] + (1 - lbd) * _data_pct.Nikkei[i - 1] * \
                                                                        _data_pct.Nikkei[i - 1]

        djia_var = _data_pct.DJIA_Var[len(self.data) - 1]
        ftse_var = _data_pct.FTSE_Var[len(self.data) - 1]
        cac_var = _data_pct.CAC_Var[len(self.data) - 1]
        nikkei_var = _data_pct.Nikkei_Var[len(self.data) - 1]

        data_pct = pd.DataFrame(index=_data_pct.index, columns=self.data.columns)
        data_pct.drop(data_pct.index[-1])
        for i in range(1, len(self.data)):
            data_pct.DJIA[i - 1] = (self.data.DJIA[i - 1] + (self.data.DJIA[i] - self.data.DJIA[i - 1])
                                     * sqrt(djia_var) / sqrt(_data_pct.DJIA_Var[i - 1])) / self.data.DJIA[i - 1]
            data_pct.FTSE[i - 1] = (self.data.FTSE[i - 1] + (self.data.FTSE[i] - self.data.FTSE[i - 1])
                                     * sqrt(ftse_var) / sqrt(_data_pct.FTSE_Var[i - 1])) / self.data.FTSE[i - 1]
            data_pct.CAC[i - 1] = (self.data.CAC[i - 1] + (self.data.CAC[i] - self.data.CAC[i - 1])
                                    * sqrt(cac_var) / sqrt(_data_pct.CAC_Var[i - 1])) / self.data.CAC[i - 1]
            data_pct.Nikkei[i - 1] = (self.data.Nikkei[i - 1] + (self.data.Nikkei[i] - self.data.Nikkei[i - 1])
                                       * sqrt(nikkei_var) / sqrt(_data_pct.Nikkei_Var[i - 1])) / self.data.Nikkei[i - 1]
        data_pct -= 1.

        scenarios = data_pct * self.today + self.today
        loss = (scenarios * self.weights / self.today).sum(axis=1)
        scenarios['Loss'] = self.today_value - loss

        # Let's get rid of the last row as it is a byproduct of calculating volatilities and is not needed anymore
        scenarios = scenarios.drop(scenarios.index[-1], axis=0)
        sorted_scenarios = scenarios.sort_values(by='Loss', ascending=False)

        beta = tf.Variable(40, dtype=tf.float64)
        gamma = tf.Variable(.3, dtype=tf.float64)
        u = 400.
        next_index = sorted_scenarios[sorted_scenarios.Loss <= u].index[0]
        n_u = sorted_scenarios.index.get_loc(next_index)
        n = len(sorted_scenarios)

        X = tf.constant(sorted_scenarios.Loss.iloc[:n_u])

        # Changing the sign due to Tensorflow's optimizers not having a 'maximize' method
        probability_ln = lambda: tf.reduce_sum(-tf.math.log((1. / beta) * tf.pow(1. + gamma * (X - u) / beta, -1. / gamma - 1.)))

        print("Cost initially: %.7f" % probability_ln())
        print("Starting values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        opt = tf.keras.optimizers.Adam()

        num_iter = 2 * self.num_iter
        step = 100
        x = np.arange(0, num_iter // step)
        y = np.zeros((3, num_iter // step))

        for i in range(num_iter):
            opt.minimize(probability_ln, [beta, gamma])
            if i % step == 0:
                y[0, i // step] = probability_ln()
                y[1, i // step] = beta
                y[2, i // step] = gamma

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(x, y[0], 'g')
        ax[0, 0].set_ylabel("Cost")
        ax[0, 1].plot(x, y[1], 'b--')
        ax[0, 1].set_ylabel("\u03B2")
        ax[1, 0].plot(x, y[2], 'k.')
        ax[1, 0].set_ylabel("\u03B3")

        print("Cost after running optimizer for %d iterations: %.7f" % (num_iter, probability_ln()))
        print("Values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta, gamma))

        ex_14_11 = u + beta / gamma * ((n / n_u * (1. - np.array(VaR_conf))) ** -gamma - 1)
        for i in range(len(VaR_conf)):
            print("Exercise 14.11a. One day VaR with a confidence level of %2.1f%%: %s"
                  % (VaR_conf[i] * 100, locale.currency(ex_14_11[i] * self.scale_factor, grouping=True)))

        ex_14_11b = n_u / n * (1. + gamma * (threashold - u) / beta) ** (-1. / gamma)
        print("Exercise 14.11b. The probability the loss will exceed $%d thousand: %.7f\n" % (threashold, ex_14_11b))

if __name__ == "__main__":
    # The first row is removed
    evt = Evt('~/Downloads/VaRExampleRMFI3eHistoricalSimulation.xls', 1e4, num_iter=20000)
    locale.setlocale(locale.LC_ALL, '')
    evt.exercise_14_8_9(VaR_conf=.97)
    evt.exercise_14_10(VaR_conf=[.99, .999])
    evt.exercise_14_11(VaR_conf=[.99, .999])

