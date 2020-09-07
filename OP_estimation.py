# -*- coding: utf-8 -*-
"""
This file estimates a production function using an anonomyzed industry-firm 
data set. I will estimate the production function using various techniques 
described in the documentation file.
"""

import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error


#Enter the path to the data file that can be downloaded 
#from the github repository.
output_data_file='data-olley-pakes.xlsx'


#read the data file
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


# The following while loop gets rid of all firms that do not have
# data for every single year in the panel. This leaves a balanced panel.
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


#ESTIMATION TECHNIQUE: FIXED EFFECT ESTIMATOR

FE_data=unbalanced_data


#The following while loop will find the average of
# each firm's capital, labor and output and then find the difference
# between the actual output in each period and the mean.
# If a firm has entry for only a single period it is removed.

k=0
j=1
FirmID=1
sum_o=0
sum_k=0
sum_l=0
rr=0

#create new variables and temporarily fill them in.

FE_data['diff_output']=FE_data['output']
FE_data['diff_capital']=FE_data['capital']
FE_data['diff_labor']=FE_data['labor']

num_data_pts=len(FE_data)
i=0
while (i<num_data_pts-1):
    sum_o=sum_o+FE_data['log_output'][FE_data.index[i-rr]]
    sum_k=sum_k+FE_data['log_capital'][FE_data.index[i-rr]]
    sum_l=sum_l+FE_data['log_labor'][FE_data.index[i-rr]]
    if (FE_data['firm'][FE_data.index[i-rr+1]]!=FirmID and j>1):
        avg_o=sum_o/j
        avg_k=sum_k/j
        avg_l=sum_l/j
        for m in range(j):
            FE_data['diff_output'][FE_data.index[k+m-rr]]=FE_data['log_output'][FE_data.index[k+m-rr]]-avg_o
            FE_data['diff_capital'][FE_data.index[k+m-rr]]=FE_data['log_capital'][FE_data.index[k+m-rr]]-avg_k
            FE_data['diff_labor'][FE_data.index[k+m-rr]]=FE_data['log_labor'][FE_data.index[k+m-rr]]-avg_l
        k=k+j
        j=1;
        FirmID+=1
        sum_o=0
        sum_k=0
        sum_l=0
    elif (FE_data['firm'][FE_data.index[i-rr+1]]!=FirmID and j==1):
        FE_data=FE_data.drop(FE_data.index[i-rr])
        FirmID+=1
        rr+=1
        k+=1
        sum_o=0
        sum_k=0
        sum_l=0
    else:
        j+=1
    i+=1
        
sum_o=sum_o+FE_data['log_output'][FE_data.index[i-rr]]
sum_k=sum_k+FE_data['log_capital'][FE_data.index[i-rr]]
sum_l=sum_l+FE_data['log_labor'][FE_data.index[i-rr]]
if (j>1):
    avg_o=sum_o/j
    avg_k=sum_k/j
    avg_l=sum_l/j
    for m in range(j):
         FE_data['diff_output'][FE_data.index[k+m-rr]]=FE_data['log_output'][FE_data.index[k+m-rr]]-avg_o
         FE_data['diff_capital'][FE_data.index[k+m-rr]]=FE_data['log_capital'][FE_data.index[k+m-rr]]-avg_k
         FE_data['diff_labor'][FE_data.index[k+m-rr]]=FE_data['log_labor'][FE_data.index[k+m-rr]]-avg_l
        
else:
    FE_data=FE_data.drop(FE_data.index[i-rr])
        
#Obtain the variables needed for a fixed effects estimator
y_FE=FE_data['diff_output']

output_features=['diff_labor','diff_capital']
X_FE=FE_data[output_features]

#run OLS with fixed effect variables
regr.fit(X_FE,y_FE)
beta_FE=regr.coef_
intrcpt_FE=regr.intercept_

#Get mean squared error
y_pred=regr.predict(X_FE)
MSE_FE=mean_squared_error(FE_data['log_output'], y_pred)


#ESTIMATION TECHNIQUE: Olley-Pakes estimation approach


# first stage:
    
OP_data=unbalanced_data

# create variables needed for the semi-parametric first stage
# phi as defined in documentation will be a 3rd order polynomial
# of investment and capital
OP_data['log_capital2']=np.power(OP_data.log_capital,2)
OP_data['log_capital3']=np.power(OP_data.log_capital,3)
OP_data['log_inv']=np.log(OP_data.inv)
OP_data['log_inv2']=np.power(OP_data.log_inv,2)
OP_data['log_inv3']=np.power(OP_data.log_inv,3)

OP_data['log_inv2capital']=OP_data.log_inv2*OP_data.log_capital
OP_data['log_capital2inv']=OP_data.log_inv*OP_data.log_capital2
OP_data['log_capitalinv']=OP_data.log_inv*OP_data.log_capital

# choose the required features to do the estimation
y_OP=OP_data['log_output']

output_features=['log_labor','log_capital','log_capital2','log_capital3','log_inv','log_inv2','log_inv3','log_inv2capital','log_capital2inv','log_capitalinv']
X_OP=OP_data[output_features]


regr=linear_model.LinearRegression(fit_intercept=False)

#run ols to estimate beta_l and phi
regr.fit(X_OP,y_OP)
beta_OP=regr.coef_
intrcpt_OP=regr.intercept_

#get beta_l
beta_l=beta_OP[0]


#Second stage

#get the values of phi for each firm and period
output_features=['log_capital','log_capital2','log_capital3','log_inv','log_inv2','log_inv3','log_inv2capital','log_capital2inv','log_capitalinv']
X_phi=OP_data[output_features]
X_phi_array=X_phi.to_numpy()

phi_coef=regr.coef_
phi_coef=phi_coef[1:]
phi=np.dot(X_phi_array,phi_coef)

#put phi into the data frame and also add a lagged phi
OP_data['phi']=phi
OP_data['lag_phi']=OP_data['phi'].shift(1)
OP_data['lag_capital']=OP_data['log_capital'].shift(1)
OP_data=OP_data.loc[OP_data.year>1]

#To complete the estimation, search through a set of possible beta_k
#For each beta_k, estimate non-parametric function h as defined in
#the documentation. Select the beta_k that produces the lowest mean 
#squared error when estimating h. h is chosen to be a third order
#polynomial.

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
    
    
 #Get MSE for OP estimates   
 
OP_data['w']=OP_data['phi']-beta_k*OP_data['log_capital']
 
y_pred=beta_l*OP_data['log_labor']+beta_k*OP_data['log_capital']+OP_data['w']
MSE_OP=mean_squared_error(OP_data['log_output'], y_pred)

 
 
    
    
    
    
    
    
    
    
    
    
    
    
    