# -*- coding: utf-8 -*-
"""
This file estimates a production function using an anonomyzed industry-firm 
data set. I will estimate the production function using various techniques 
described in the documentation pdf.
"""

# added a comment at the top

import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error


#read the data file
output_data_file='data-olley-pakes.xlsx'
output_data=pd.read_excel(output_data_file)
    
#Although, there are multiple estimation techniques,
# through mathematical manipulation, that can
# all be reduced to doing linear regressions.
# More details are in the documentation pdf.
regr=linear_model.LinearRegression()

#ESTIMATION TECHNIQUE: UNBALANCED OLS

#Remove all entries with no output since the firm is not active
unbalanced_data=output_data.loc[output_data.output>0]

#Take logarithms of all the relevant variables
unbalanced_data['log_output']=np.log(unbalanced_data.output)
unbalanced_data['log_capital']=np.log(unbalanced_data.capital)
unbalanced_data['log_labor']=np.log(unbalanced_data.labor)

y_unbalanced=unbalanced_data['log_output']

output_features=['log_labor','log_capital']
X_unbalanced=unbalanced_data[output_features]

#Run OLS and get results
regr.fit(X_unbalanced,y_unbalanced)
beta_unbalanced=regr.coef_
intrcpt_unbalanced=regr.intercept_

#mean squared error
y_pred=regr.predict(X_unbalanced)
MSE_unbalanced=mean_squared_error(unbalanced_data['log_output'], y_pred)


#ESTIMATION TECHNIQUE: BALANCED OLS
balanced_data=output_data


# The following while loop gets rid of all firms that do not have positive output
# data for every single year in the panel. This leaves a balanced panel. An output of 0
# means the firm is not active.
i=0
num_data_pts=len(output_data)

while (i<num_data_pts):
    if (balanced_data['output'][i]>0 and balanced_data['output'][i+1]>0 and balanced_data['output'][i+2]>0 and balanced_data['output'][i+3]>0 and balanced_data['output'][i+4]>0 and balanced_data['output'][i+5]>0):
        i+=6
    else:
        balanced_data=balanced_data.drop([i,i+1,i+2,i+3,i+4,i+5])
        i+=6
       
#Take logarithms of all the relevant variables        
balanced_data['log_output']=np.log(balanced_data.output)
balanced_data['log_capital']=np.log(balanced_data.capital)
balanced_data['log_labor']=np.log(balanced_data.labor)

y_balanced=balanced_data['log_output']

output_features=['log_labor','log_capital']
X_balanced=balanced_data[output_features]

#run OLS and get results
regr.fit(X_balanced,y_balanced)
beta_balanced=regr.coef_
intrcpt_balanced=regr.intercept_

#mean squared error
y_pred=regr.predict(X_balanced)
MSE_balanced=mean_squared_error(balanced_data['log_output'], y_pred)


#FIXED EFFECT ESTIMATOR

FE_data=unbalanced_data


#The following while loop will find the average of
# each firm's capital, labor and output and then find the difference
# between the actual output in each period and the mean.
# If a firm has entry for only a single period it is removed.



def calc_diff(group):
    avg_o=np.nanmean(group['log_output'])
    avg_k=np.nanmean(group['log_capital'])
    avg_l=np.nanmean(group['log_labor'])
    t=len(group)
    group['diff_output']=group['log_output']-avg_o
    group['diff_labor']=group['log_labor']-avg_l
    group['diff_capital']=group['log_capital']-avg_k
    group['num_periods']=t
    return group

FE_data=FE_data.groupby('firm').apply(calc_diff)
        
# Firms with a single entry will have diff_output=0. These are removed
FE_data=FE_data[FE_data['num_periods']>1]
     
y_FE=FE_data['diff_output']

output_features=['diff_labor','diff_capital']
X_FE=FE_data[output_features]

# run OLS on the fixed effect variables and get results
regr.fit(X_FE,y_FE)
beta_FE=regr.coef_
intrcpt_FE=regr.intercept_

# Get mean squared error.
y_pred=regr.predict(X_FE)
MSE_FE=mean_squared_error(FE_data['log_output'], y_pred)

OP_data=unbalanced_data


OP_data['log_capital2']=np.power(OP_data.log_capital,2)
OP_data['log_capital3']=np.power(OP_data.log_capital,3)
OP_data['log_inv']=np.log(OP_data.inv)
OP_data['log_inv2']=np.power(OP_data.log_inv,2)
OP_data['log_inv3']=np.power(OP_data.log_inv,3)

OP_data['log_inv2capital']=OP_data.log_inv2*OP_data.log_capital
OP_data['log_capital2inv']=OP_data.log_inv*OP_data.log_capital2
OP_data['log_capitalinv']=OP_data.log_inv*OP_data.log_capital

y_OP=OP_data['log_output']


output_features=['log_labor','log_capital','log_capital2','log_capital3','log_inv','log_inv2','log_inv3','log_inv2capital','log_capital2inv','log_capitalinv']
X_OP=OP_data[output_features]

regr=linear_model.LinearRegression(fit_intercept=False)

regr.fit(X_OP,y_OP)
beta_OP=regr.coef_
intrcpt_OP=regr.intercept_

beta_l=beta_OP[0]

output_features=['log_capital','log_capital2','log_capital3','log_inv','log_inv2','log_inv3','log_inv2capital','log_capital2inv','log_capitalinv']
X_phi=OP_data[output_features]
X_phi_array=X_phi.to_numpy()

phi_coef=regr.coef_
phi_coef=phi_coef[1:]
phi=np.dot(X_phi_array,phi_coef)

OP_data['phi']=phi
OP_data['lag_phi']=OP_data['phi'].shift(1)
OP_data['lag_capital']=OP_data['log_capital'].shift(1)
OP_data=OP_data.loc[OP_data.year>1]

opt_beta_k=0
cur_min=1000
for j in range(1001):
    beta_k=j/1000
    OP_data['w']=OP_data['phi']-beta_k*OP_data['log_capital']
    OP_data['lag_w']=OP_data['lag_phi']-beta_k*OP_data['lag_capital']
    OP_data['lag_w2']=np.power(OP_data['lag_w'],2)
    OP_data['lag_w3']=np.power(OP_data['lag_w'],3)
    output_features=['lag_w','lag_w2','lag_w3']
    X_w=OP_data[output_features]
    y_w=OP_data['w']
    regr.fit(X_w,y_w)
    w_pred=regr.predict(X_w)
    temp=mean_squared_error(OP_data['w'], w_pred)
    if (temp<cur_min):
        opt_beta_k=j
        cur_min=temp

beta_k=opt_beta_k/1000
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    