# coding: utf-8
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from math import sqrt
import locale


data = pd.read_excel('/Users/ilchen/Downloads/VaRExampleRMFI3eHistoricalSimulation.xls')
data.index = data['Date']
data = data.drop(['Date'], axis=1)
data = data.drop(['FTSE-100', 'USD/GBP', 'CAC-40', 'EUR/USD', 'Nikkei', 'YEN/USD'], axis=1)
data.columns = ['DJIA', 'FTSE', 'CAC', 'Nikkei']
today = data.iloc[-1]
data_pct = data.pct_change().dropna()

weights = Series([4000, 3000, 1000, 2000], data.columns)
today_value = 10000.

lbd = .94
djia_var = data_pct.DJIA.var()
ftse_var = data_pct.FTSE.var()
cac_var  = data_pct.CAC.var()
nikkei_var = data_pct.Nikkei.var()
data_pct['DJIA_Var'] = djia_var
data_pct['FTSE_Var'] = ftse_var
data_pct['CAC_Var']  = cac_var
data_pct['Nikkei_Var'] = nikkei_var
data_pct = data_pct[['DJIA', 'DJIA_Var', 'FTSE', 'FTSE_Var', 'CAC', 'CAC_Var', 'Nikkei', 'Nikkei_Var']]
extra_row = DataFrame(np.zeros((1,8)), index=[data_pct.index[len(data_pct) - 1]+datetime.timedelta(days=1)],
                      columns=data_pct.columns)
data_pct  = pd.concat([data_pct, extra_row], axis=0)
for i in range(1, len(data)):
    data_pct.DJIA_Var[i] = lbd * data_pct.DJIA_Var[i - 1] + (1 - lbd) * data_pct.DJIA[i - 1] * data_pct.DJIA[i - 1]
    data_pct.FTSE_Var[i] = lbd * data_pct.FTSE_Var[i - 1] + (1 - lbd) * data_pct.FTSE[i - 1] * data_pct.FTSE[i - 1]
    data_pct.CAC_Var[i] = lbd * data_pct.CAC_Var[i - 1] + (1 - lbd) * data_pct.CAC[i - 1] * data_pct.CAC[i - 1]
    data_pct.Nikkei_Var[i] = lbd * data_pct.Nikkei_Var[i - 1] + (1 - lbd) * data_pct.Nikkei[i - 1] * data_pct.Nikkei[i - 1]

djia_var = data_pct.DJIA_Var[len(data) - 1]
ftse_var = data_pct.FTSE_Var[len(data) - 1]
cac_var = data_pct.CAC_Var[len(data) - 1]
nikkei_var = data_pct.Nikkei_Var[len(data) - 1]

data_pct3 = pd.DataFrame(index=data_pct.index, columns=data.columns)
data_pct3.drop(data_pct3.index[-1])
for i in range(1, len(data)):
    data_pct3.DJIA[i-1] = (data.DJIA[i-1] + (data.DJIA[i] - data.DJIA[i-1])
                           * sqrt(djia_var) / sqrt(data_pct.DJIA_Var[i-1])) / data.DJIA[i-1]
    data_pct3.FTSE[i-1] = (data.FTSE[i-1] + (data.FTSE[i] - data.FTSE[i-1])
                           * sqrt(ftse_var) / sqrt(data_pct.FTSE_Var[i-1])) / data.FTSE[i-1]
    data_pct3.CAC[i-1] = (data.CAC[i-1] + (data.CAC[i]  - data.CAC[i-1])
                          * sqrt(cac_var) / sqrt(data_pct.CAC_Var[i-1])) / data.CAC[i-1]
    data_pct3.Nikkei[i-1] = (data.Nikkei[i-1] + (data.Nikkei[i] - data.Nikkei[i-1])
                             * sqrt(nikkei_var) / sqrt(data_pct.Nikkei_Var[i-1])) / data.Nikkei[i-1]
data_pct3 -= 1.

scenarios = data_pct3 * today + today
loss = (scenarios*weights/today).sum(axis=1)
scenarios['Loss'] = today_value - loss

# Let's get rid of the last row as it is a byproduct of calculating volatilities and is not needed anymore
scenarios = scenarios.drop(scenarios.index[-1], axis=0)
sorted_scenarios = scenarios.sort_values(by='Loss', ascending=False)


beta =  tf.Variable(40, dtype=tf.float64)
gamma = tf.Variable(.3, dtype=tf.float64)
u = 400.
next_index = sorted_scenarios[sorted_scenarios.Loss <= u].index[0]
n_u = sorted_scenarios.index.get_loc(next_index)
n = len(sorted_scenarios)

X = tf.constant(sorted_scenarios.Loss.iloc[:n_u])

# Changing the sign due to Tensorflow's optimizers not having a 'maximize' method
out = -tf.log((1. / beta) * tf.pow(1. + gamma * (X - u) / beta, -1. / gamma - 1.) )
probability_ln = tf.reduce_sum(out)

opt =  tf.train.AdamOptimizer().minimize(probability_ln)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(probability_ln)

num_iter = 60000
step = 100
x = np.arange(0, num_iter // step)
y = np.zeros((3, num_iter // step))

for i in range(num_iter):
    sess.run(opt)
    if i % step == 0:
        y[0, i // step] = sess.run(probability_ln)
        y[1, i // step] = sess.run(beta)
        y[2, i // step] = sess.run(gamma)

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(x, y[0], 'g')
ax[0, 0].set_ylabel("Cost")
ax[0, 1].plot(x, y[1], 'b--')
ax[0, 1].set_ylabel("\u03B2")
ax[1, 0].plot(x, y[2], 'k.')
ax[1, 0].set_ylabel("\u03B3")

print("Cost after running optimizer for %d iterations: %.7f" % (num_iter, -sess.run(probability_ln)))

beta_val  = sess.run(beta)
gamma_val = sess.run(gamma)

print("Values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta_val, gamma_val))

scale_factor = 1e3
VaR_conf = np.array([.99, .999])
locale.setlocale(locale.LC_ALL, '')
ex_14_11 = u + beta_val / gamma_val * ( (n / n_u * (1. - VaR_conf)) ** -gamma_val - 1)
for i in range(len(VaR_conf)):
    print("One day VaR with a confidence level of %2.1f%%: %s"
          % (VaR_conf[i] * 100, locale.currency(ex_14_11[i] * scale_factor, grouping=True)) )

threashold = 600

ex_14_11b = n_u / n * (1. + gamma_val * (threashold - u) / beta_val) ** (-1. / gamma_val)
print("The probability the loss will exceed $%d thousand: %.7f\n" % (threashold, ex_14_11b))

sess.close()
