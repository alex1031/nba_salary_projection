# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:51:13 2022

@author: Alex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mod_1 = pd.read_csv('model_1_eda_data.csv')
mod_2 = pd.read_csv('model_2_eda_data.csv')

# Choose data
mod_1_y = mod_1['Salary Adjusted'].values
mod_1_x = mod_1[['Pos', 'age_bin', 'games_bin', 'games_started_bin',
                 'OBPM', 'DBPM', 'ORtg', 'DRtg']]

mod_2_y = mod_2['Salary Adjusted'].values
mod_2_x = mod_2[['Pos', 'age_bin','games_bin', 'games_started_bin',
                 'eFG%', 'FTR', 'TRB%', 'TOV%']]

# Dummy Variable for position and intervals
mod_1_dum = pd.get_dummies(mod_1_x)
mod_2_dum = pd.get_dummies(mod_2_x)


# Train test split
from sklearn.model_selection import train_test_split

X_train_mod_1, X_test_mod_1, y_train_mod_1, y_test_mod_1 = train_test_split(mod_1_dum, mod_1_y, test_size = 0.2, random_state=4)
X_train_mod_2, X_test_mod_2, y_train_mod_2, y_test_mod_2 = train_test_split(mod_2_dum, mod_2_y, test_size = 0.2, random_state=4)

# Standardise non-normal adjusted salary - Cube Root Transformation
y_train_mod_1_norm = np.cbrt(y_train_mod_1)
y_test_mod_1_norm = np.cbrt(y_test_mod_1)
y_train_mod_2_norm = np.cbrt(y_train_mod_2)
y_test_mod_2_norm = np.cbrt(y_test_mod_2)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lm_mod_1 = LinearRegression()
lm_mod_1.fit(X_train_mod_1, y_train_mod_1_norm)

lm_mod_2 = LinearRegression()
lm_mod_2.fit(X_train_mod_2, y_train_mod_2_norm)

print(np.mean(cross_val_score(lm_mod_1, X_train_mod_1, y_train_mod_1_norm, scoring = 'neg_mean_absolute_error')))
print(np.mean(cross_val_score(lm_mod_2, X_train_mod_2, y_train_mod_2_norm, scoring = 'neg_mean_absolute_error')))

# Ridge Regression - multicollinearity of some parameters
from sklearn.linear_model import Ridge
rd_mod_1 = Ridge(alpha=2.6)
rd_mod_1.fit(X_train_mod_1, y_train_mod_1_norm)
rd_mod_2 = Ridge(alpha=1.94)
rd_mod_2.fit(X_train_mod_2, y_train_mod_2_norm)

# Optimisation
alpha = []
error = []

for i in range(1,300):
    alpha.append(i/100)
    rd = Ridge(alpha=(i/100))
    error.append(np.mean(cross_val_score(rd,X_train_mod_1,y_train_mod_1_norm, scoring = 'neg_mean_absolute_error')))
    
plt.plot(alpha, error)
plt.show()

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
print(df_err[df_err.error == max(df_err.error)])

alpha = []
error = []

for i in range(1,250):
    alpha.append(i/100)
    rd = Ridge(alpha=(i/100))
    error.append(np.mean(cross_val_score(rd,X_train_mod_2,y_train_mod_2_norm, scoring = 'neg_mean_absolute_error')))
    
plt.plot(alpha, error)
plt.show()

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
print(df_err[df_err.error == max(df_err.error)])

# Random Forest - Don't necessarily need to normalise output but we use normalised for consistency
from sklearn.ensemble import RandomForestRegressor
rf_mod_1 = RandomForestRegressor()
rf_mod_2 = RandomForestRegressor()
print(np.mean(cross_val_score(rf_mod_1, X_train_mod_1, y_train_mod_1_norm, scoring='neg_mean_absolute_error')))
print(np.mean(cross_val_score(rf_mod_2, X_train_mod_2, y_train_mod_2_norm, scoring='neg_mean_absolute_error')))

# GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 150, 10), 'criterion':('mse', 'mae'), 'max_features':('auto','sqrt','log2')}
gs_mod_1 = GridSearchCV(rf_mod_1, parameters, n_jobs=-1, scoring='neg_mean_absolute_error')
gs_mod_1.fit(X_train_mod_1, y_train_mod_1_norm)
print(gs_mod_1.best_score_)
print(gs_mod_1.best_estimator_)

gs_mod_2 = GridSearchCV(rf_mod_2, parameters, n_jobs=-1, scoring='neg_mean_absolute_error')
gs_mod_2.fit(X_train_mod_2, y_train_mod_2_norm)
print(gs_mod_2.best_score_)
print(gs_mod_2.best_estimator_)

lm_pred_mod_1 = lm_mod_1.predict(X_test_mod_1)
lm_pred_mod_2 = lm_mod_2.predict(X_test_mod_2)
rd_pred_mod_1 = rd_mod_1.predict(X_test_mod_1)
rd_pred_mod_2 = rd_mod_2.predict(X_test_mod_2)
gs_pred_mod_1 = gs_mod_1.best_estimator_.predict(X_test_mod_1)
gs_pred_mod_2 = gs_mod_2.best_estimator_.predict(X_test_mod_2)

# Support Vector Regression
# Standardise Numerical Values for model 1 - ORtg and DRtg are very large compared to OBPM and DBPM
from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()
X_train_mod_1_catvar = X_train_mod_1.iloc[:, 4:len(X_train_mod_1)].reset_index().drop(columns='index')
X_train_mod_1_numvar = pd.DataFrame(sc_X_train.fit_transform(X_train_mod_1.iloc[:, 0:4]))
X_svm_train_mod_1 = pd.concat([X_train_mod_1_numvar, X_train_mod_1_catvar], axis=1)

sc_X_test = StandardScaler()
X_test_mod_1_catvar = X_test_mod_1.iloc[:, 4:len(X_test_mod_1)].reset_index().drop(columns='index')
X_test_mod_1_numvar = pd.DataFrame(sc_X_test.fit_transform(X_test_mod_1.iloc[:, 0:4]))
X_svm_test_mod_1 = pd.concat([X_test_mod_1_numvar, X_test_mod_1_catvar], axis=1)

sc_X_train_mod_2 = StandardScaler()
X_train_mod_2_catvar = X_train_mod_2.iloc[:, 4:len(X_train_mod_2)].reset_index().drop(columns='index')
X_train_mod_2_numvar = pd.DataFrame(sc_X_train_mod_2.fit_transform(X_train_mod_2.iloc[:, 0:4]))
X_svm_train_mod_2 = pd.concat([X_train_mod_2_numvar, X_train_mod_2_catvar], axis=1)

sc_X_test_mod_2 = StandardScaler()
X_test_mod_2_catvar = X_test_mod_2.iloc[:, 4:len(X_test_mod_2)].reset_index().drop(columns='index')
X_test_mod_2_numvar = pd.DataFrame(sc_X_test_mod_2.fit_transform(X_test_mod_2.iloc[:, 0:4]))
X_svm_test_mod_2 = pd.concat([X_test_mod_2_numvar, X_test_mod_2_catvar], axis=1)

from sklearn.svm import LinearSVR

sc_y_train_1 = StandardScaler()
sv_y_train_mod_1 = sc_y_train_1.fit_transform(y_train_mod_1_norm.reshape(-1, 1))
sc_y_test_1 = StandardScaler()
sv_y_test_mod_1 = sc_y_test_1.fit_transform(y_test_mod_1_norm.reshape(-1, 1))
lsvr_mod_1 = LinearSVR()

sc_y_train_2 = StandardScaler()
sv_y_train_mod_2 = sc_y_train_2.fit_transform(y_train_mod_2_norm.reshape(-1, 1))
sc_y_test_2 = StandardScaler()
sv_y_test_mod_2 = sc_y_test_2.fit_transform(y_test_mod_2_norm.reshape(-1, 1))
lsvr_mod_2 = LinearSVR()

# Optimisation
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
param = {'epsilon': [0, 0.01, 0.1, 0.5, 1, 2, 4], 'C': [0.1, 1, 10, 100, 1000], 'max_iter':range(10000, 20000, 500)}
gsvr_mod_1 = GridSearchCV(lsvr_mod_1, param, n_jobs=-1, scoring='neg_mean_absolute_error')
gsvr_mod_1.fit(X_svm_train_mod_1, sv_y_train_mod_1.ravel())
print(gsvr_mod_1.best_score_)
print(gsvr_mod_1.best_estimator_)

gsvr_mod_2 = GridSearchCV(lsvr_mod_2, param, n_jobs=-1, scoring='neg_mean_absolute_error')
gsvr_mod_2.fit(X_svm_train_mod_2, sv_y_train_mod_2.ravel())
print(gsvr_mod_2.best_score_)
print(gsvr_mod_2.best_estimator_)

# test and evaluate
gsvr_pred_mod_1 = gsvr_mod_1.best_estimator_.predict(X_svm_test_mod_1)
gsvr_pred_mod_2 = gsvr_mod_2.best_estimator_.predict(X_svm_test_mod_2)

# evaluate models
from sklearn.metrics import mean_absolute_error
print('Model 1 Results')
print('Linear Regression:', mean_absolute_error(y_test_mod_1_norm, lm_pred_mod_1))
print('Ridge Regression:',mean_absolute_error(y_test_mod_1_norm, rd_pred_mod_1))
print('Random Forest:', mean_absolute_error(y_test_mod_1_norm, gs_pred_mod_1))
print('Support Vector Regression:', mean_absolute_error(y_test_mod_1_norm, sc_y_test_1.inverse_transform(gsvr_pred_mod_1)))

print('Model 2 Results')
print('Linear Regression:', mean_absolute_error(y_test_mod_2_norm, lm_pred_mod_2))
print('Ridge Regression:', mean_absolute_error(y_test_mod_2_norm, rd_pred_mod_2))
print('Random Forest:', mean_absolute_error(y_test_mod_2_norm, gs_pred_mod_2))
print('Support Vector Regression:', mean_absolute_error(y_test_mod_2_norm, sc_y_test_2.inverse_transform(gsvr_pred_mod_2)))
# ----- Support Vector Regression for Model 1 and Random Forest for Model 2 ----- #

# Creating Display Dataframe for FlaskAPI
disp_cols = ['Player', 'Pos', 'Age', 'G', 'GS', 'MP', 'ORtg', 'DRtg', 'OBPM', 'DBPM']
disp_cols_2 = ['eFG%', 'FTR', 'TRB%', 'TOV%', 'Salary', 'Salary Adjusted']
disp_df_1 = mod_1.iloc[429:, :][disp_cols].reset_index().drop(columns='index')
disp_temp = mod_2.iloc[429:, :][disp_cols_2].reset_index().drop(columns='index')
disp_df_1 = pd.concat([disp_df_1, disp_temp], axis=1)
sc_disp = StandardScaler()
disp_catvar = mod_1.iloc[429:, :][['Pos', 'age_bin', 'games_bin', 'games_started_bin']].reset_index().drop(columns='index')
disp_numvar = pd.DataFrame(sc_disp.fit_transform(mod_1.iloc[429:, :][['OBPM', 'DBPM', 'ORtg', 'DRtg']]))
disp_x = pd.concat([disp_numvar, disp_catvar], axis=1)
disp_x = pd.get_dummies(disp_x)
sc_disp_y = StandardScaler()
y_1 = sc_disp_y.fit_transform(np.cbrt(mod_1.iloc[420:, :]['Salary Adjusted'].values).reshape(-1, 1))

disp_df_1['Rating Model Salary'] = sc_disp_y.inverse_transform(gsvr_mod_1.best_estimator_.predict(disp_x))**3

disp_x_2 = mod_2_dum.iloc[429:, :].reset_index().drop(columns='index')
disp_df_1['Four Factors Salary'] = gs_mod_2.best_estimator_.predict(disp_x_2)**3
disp_df_1['Average Predicted Salary'] = (disp_df_1['Rating Model Salary'] + disp_df_1['Four Factors Salary'])/2
disp_df_1.to_csv('display_df.csv')

gsvr_mod_1.best_estimator_.coef_



