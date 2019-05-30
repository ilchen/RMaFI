# coding: utf-8
import numpy as np
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_excel('/Users/ilchen/Downloads/VaRExampleRMFI3eHistoricalSimulation.xls')
data.index = data['Date']
data = data.drop(['Date'], axis=1)
data = data.drop(['FTSE-100', 'USD/GBP', 'CAC-40', 'EUR/USD', 'Nikkei', 'YEN/USD'], axis=1)
data.columns = ['DJIA', 'FTSE', 'CAC', 'Nikkei']
today = data.iloc[-1]
data_pct = data.pct_change().dropna()
scenarios = data_pct * today + today

weights = Series([4000, 3000, 1000, 2000], scenarios.columns)
loss = (scenarios*weights/today).sum(axis=1)
today_value = 10000.

scenarios['Loss'] = today_value-loss
sorted_scenarios = scenarios.sort_values(by='Loss', ascending=False)

beta =  tf.Variable(40, dtype=tf.float64)
gamma = tf.Variable(.3, dtype=tf.float64)
u = 150.
next_index = sorted_scenarios[sorted_scenarios.Loss <= u].index[0]
n_u = sorted_scenarios.index.get_loc(next_index)
n   = len(sorted_scenarios)

X   = tf.constant(sorted_scenarios.Loss.iloc[:n_u])

# Changing the sign due to Tensorflow's optimizers not having a 'maximize' method
out = -tf.log((1. / beta) * tf.pow(1. + gamma * (X - u) / beta, -1. / gamma - 1.) )
probability_ln = tf.reduce_sum(out)

opt =  tf.train.AdamOptimizer().minimize(probability_ln)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(probability_ln)

num_iter = 30000
for i in range(num_iter):  sess.run(opt)

print("Cost after running optimizer for %d iterations: %.7f" % (num_iter, -sess.run(probability_ln)))
beta_val  = sess.run(beta)
gamma_val = sess.run(gamma)

print("New values: \u03B2 = %.7f, \u03B3 = %.7f" % (beta_val, gamma_val))

# Exercise 14.10
VaR_conf = np.array([.99, .999])
ex_14_10 = u + beta_val / gamma_val * ( (n / n_u * (1. - VaR_conf)) ** -gamma_val - 1)
for i in range(len(VaR_conf)):
    print("One day VaR with a confidence level of %2.1f%%: %.7f" % (VaR_conf[i] * 100, ex_14_10[i]))

sess.close()

