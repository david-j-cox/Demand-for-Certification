#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    cox.david.j@gmail.com
    https://www.researchgate.net/profile/David_Cox26
    twitter: @davidjcox_
    LinkedIn: https://www.linkedin.com/in/coxdavidj/
    Website: https://davidjcox.xyz
"""
#Set current working directory to the folder that contains your data.
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set path to local scripts
sys.path.append('/Users/davidjcox/Dropbox/Coding/Local Python Modules/')

# Set path to data
os.chdir('/Users/davidjcox/Dropbox/Projects/CurrentProjectManuscripts/Empirical/PersonalFun/Demand for Certification/Data/')

# Change settings to view all columns of data
pd.set_option('display.max_columns', None)

#%% Read in the data
raw_data =pd.read_csv('all_raw_data.csv')
data = raw_data.copy()
data.head()

#%% Subset cost cols for analysis
cost_df = data[['cost_0', 'cost_25', 'cost_50', 'cost_100', 'cost_200', 'cost_300', 'cost_400', 'cost_500', 'cost_600', \
              'cost_700', 'cost_800', 'cost_900', 'cost_1000', 'cost_1100', 'cost_1200', 'cost_1300', 'cost_1400', \
              'cost_1500', 'cost_1600', 'cost_1700', 'cost_1800', 'cost_1900', 'cost_2000', 'cost_2100', 'cost_2200', \
              'cost_2300', 'cost_2400', 'cost_2500', 'cost_2600', 'cost_2700', 'cost_2800', 'cost_2900', 'cost_3000']]
cost_df.head()

#%% Plot distributions of data
for i in list(cost_df):
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.hist(x=cost_df[i], color='k', alpha=0.8)
    plt.xlabel('Likeilihood of Paying', fontsize=26)
    plt.ylabel('Number of Participants', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(i, fontsize=30)
    plot_name = plt.savefig('%s_hist.png' %i, bbox_inches='tight')
    plt.show()     

#%% Reshape data to long form for swarmplots
cost_df_long = cost_df.stack()
cost_df_long = cost_df_long.reset_index()
cost_df_long.columns = ['participant', 'cost', 'likelihood']
cost_df_long.to_csv('all_data_long.csv')

cost = [0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, \
        1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

#%% Plot swarmplots
fig, ax = plt.subplots(figsize=(30, 20))
sns.swarmplot(x=cost_df_long['cost'], y=cost_df_long['likelihood'], color='k')
plt.xlabel('Cost in Canadian Dollars', fontsize=60, labelpad=(20))
plt.ylabel('Likelihood of Paying', fontsize=60, labelpad=(20))
plt.xticks(ticks=list(range(len(cost))), labels=cost, fontsize=24, rotation=90)
plt.yticks(fontsize=30)
plot_name = plt.savefig('swarmplots_cost.png', bbox_inches='tight')
plt.show()  

#%% Plot jitterplot
fig, ax = plt.subplots(figsize=(30, 20))
sns.stripplot(x=cost_df_long['cost'], y=cost_df_long['likelihood'], color='k')
plt.xlabel('Cost in Canadian Dollars', fontsize=60, labelpad=(20))
plt.ylabel('Likelihood of Paying', fontsize=60, labelpad=(20))
plt.xticks(ticks=list(range(len(cost))), labels=cost, fontsize=24, rotation=90)
plt.yticks(fontsize=30)
plot_name = plt.savefig('jitterplots_cost.png', bbox_inches='tight')
plt.show()

#%% Plot violinplots
fig, ax = plt.subplots(figsize=(30, 20))
sns.stripplot(x=cost_df_long['cost'], y=cost_df_long['likelihood'], color='k')
plt.xlabel('Cost in Canadian Dollars', fontsize=60, labelpad=(20))
plt.ylabel('Likelihood of Paying', fontsize=60, labelpad=(20))
plt.xticks(ticks=list(range(len(cost))), labels=cost, fontsize=24, rotation=90)
plt.yticks(fontsize=30)
plot_name = plt.savefig('violinplots_cost.png', bbox_inches='tight')
plt.show()  

#%% Define the demand model to the median values
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def demand(x, alpha):
    return 100*10**(3*np.exp(-alpha*100*x))

#%% Fit demand equation to all data


#%%


#Subset data 
data_fit = data[cols_check]
cost = [1, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, \
        1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
x = pd.array(cost, int)

q_s = []
a_s = []
r2_vals = []

for i in list(range(len(data_fit))):
    likelihood = data_fit.loc[i]
    y = pd.array(likelihood, int)
    c, cov = curve_fit(demand, x, y)
    preds = demand(x, *c)
    r_2 = r2_score(y, preds)
    q_s.append(c[0])
    a_s.append(c[1])
    r2_vals.append(round(r_2, 4))

#%% Combine params into single df
estimated_params = pd.DataFrame(q_s)
esrtimated_params = pd.concat([k_s, a_s, r2_vals], axis=1)
estimated_params

#%% 
med_vals = []

for i in list(data_fit):
    med = round(data_fit[i].mean(), 4)
    med_vals.append(med)

#%% 
med_vals_logged = np.log(med_vals)
costs_logged = np.log(cost)

#%% Predicted values
from pandas import Series
popt, pcov = curve_fit(demand, cost, med_vals)
predPlot = Series(preds)

#%% Regular plot
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(cost,demand(cost*popt), color='k--') # Predicted curve
plt.scatter(x=cost, y=med_vals, marker='o', color='k', s=75) # Observed curve
plt.xlim(.8, 3500)
plt.xscale('log')
plt.ylim(.8, 100)
plt.yscale('log')
plt.ylabel('Likelihood Purchase', fontsize=30)
plt.xlabel('Cost of Certification ($)', fontsize=30)
plt.xticks([1, 25, 50, 100, 200, 400, 800, 1600, 3000], fontsize=14, labels=['0', '25', '50', '100', '200', '400', '800', '1600', '3000'])
plt.yticks([1, 10, 25, 50, 100], fontsize=14, labels=['1', '10', '25', '50', '100'])
right_side = ax.spines["right"]
right_side.set_visible(False)
top = ax.spines["top"]
top.set_visible(False)
plt.show()

#%% Linear x axis
x_vals = list(range(1, 34))
fig, ax = plt.subplots(figsize=(10, 7))
plt.scatter(x=x_vals, y=med_vals, marker='o', color='k', s=75)
plt.xlim(0, 35)
plt.ylim(.8, 100)
plt.yscale('log')
plt.ylabel('Likelihood Purchase', fontsize=30)
plt.xlabel('Cost of Certification ($)', fontsize=30)
plt.xticks([1, 4, 8, 13, 18, 23, 28, 33], fontsize=14, labels=['0', '100', '500', '1000', '1500', '2000', '2500', '3000'])
plt.yticks([1, 10, 25, 50, 100], fontsize=14, labels=['1', '10', '25', '50', '100'])
right_side = ax.spines["right"]
right_side.set_visible(False)
top = ax.spines["top"]
top.set_visible(False)
plt.show()