import os,sys
import numpy as np
import pandas as pd
import pytest
from isoweek import Week
from datetime import datetime, timedelta
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, WEEKLY
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_absolute_error


def data_preprocess(data_csv_path):
    '''
    input: raw csv containing two rows, the first row contains the yearweek, second row contains the values.
    output:
        ds: last day (Sunday) of the corresponding week.
        y: transportation volume
    '''
    print('\n\nStarting data preprocessing ...')
    raw_df = pd.read_csv(data_csv_path,header=None).T
    raw_df.columns = ['yearweek','y']
    raw_df['ds'] = raw_df.apply(lambda x: Week(int(str(x['yearweek'])[:4]), \
                                               int(str(x['yearweek'])[-2:])).sunday(), axis=1)
    final_df = raw_df[['ds','y']]
    print('\nInput data (tail):')
    print(final_df.tail(10))
    print('\nInput data info:')
    print("Shape:",final_df.shape)
    print(final_df.describe())
    print('Data preprocessing done.\n\n')
    return final_df


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_seasonality_trend_residue(dataframe, n=None):
    '''
    Input specifications:
    A positional input which takes in the output from the previous step.
    An optional input which is an integer indication how many data points to predict, defaults to 6.
    
    Output specifications:
    This function returns two DataFrame objects of the same format as the input. The first DataFrame object indicates the seasonality and trend detected. The second DataFrame object contains the residue of subtracting the seasonality and trend from the input.
    Both outputs would have n number of extra rows in the end where n is the same as the integer indicated by the optional input.
    For the first output, the n extra rows contains predicted seasonality and trend of the next n weeks .
    For the second output, the ds column should be filled for the n extra rows but the y column should be left as numpy.nan. They will be filled during the next step.
    
    Requirements:
    The process of hyper-parameter tuning has to be automated.
    Set the second parameter to 6.
    '''
    
    print('\n\nStarting timeseries component extraction...')
    
    # initialized kernels with default parameter values to be tuned during model fit
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
    k1 = ConstantKernel(constant_value=2) * ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
    k2 = ConstantKernel(constant_value=10, constant_value_bounds=(1e-2, 1e3)) * RBF(length_scale=1e2, length_scale_bounds=(1e-2, 1e3)) 
    
    kernel  = k0+k1+k2
    gpr = GaussianProcessRegressor( kernel=kernel,
                                    normalize_y=True,
                                    alpha=1e-10,
                                    random_state=0
                                    )
    
    # generate time variable 
    l = dataframe.shape[0]
    dataframe['t'] = np.arange(l)
    X = dataframe['t'].values.reshape(l, 1)
    y = dataframe['y'].values.reshape(l, 1)
    
    
    # Model fittings, kernel hyperparameter tuning takes place during the fit itself
    gpr.fit(X, y)
    print("GP kernel: %s" % gpr.kernel_)
    print("n=",n)
    
    # generate predictions
    y_pred, y_std = gpr.predict(X, return_std=True)
    seasonal_df = dataframe.copy()
    residue_df = dataframe.copy()
    seasonal_df['y'] = y_pred
    residue_df['y'] = dataframe['y'] - seasonal_df['y']
    seasonal_df.drop(['t'],axis=1,inplace=True)
    residue_df.drop(['t'],axis=1,inplace=True)
    
    print(f'GPR R2 Score = {gpr.score(X=X, y=y): 0.3f}')
    print(f'GPR MAPE = {mean_absolute_percentage_error(y_true=y, y_pred=gpr.predict(X)): 0.3f}%')

    if n!=None:
        dt_dict = {
                    'ds': [str(dt.date()) for dt in list(rrule(freq=WEEKLY, interval=1, dtstart=seasonal_df['ds'][l-1]+timedelta(days=7), count=n))],
                    'y': np.full([n], np.nan)
                  }
        newdt_df = pd.DataFrame(dt_dict)
        seasonal_df = pd.concat([seasonal_df,newdt_df],axis=0)
        seasonal_df.reset_index(drop=True,inplace=True)
        residue_df = pd.concat([residue_df,newdt_df],axis=0)
        residue_df.reset_index(drop=True,inplace=True)
        
        l = seasonal_df.shape[0]
        seasonal_df['t'] = np.arange(l)
        X = seasonal_df['t'].values.reshape(l, 1)
        seasonal_df['y'] = gpr.predict(X)
        seasonal_df.drop(['t'],axis=1,inplace=True)
    print("\n")
    print('Seasonality+Trend (tail):')
    print(seasonal_df.tail(10))
    print("\n")
    print('Residue (tail):')
    print(residue_df.tail(10))
    print('Timeseries component extraction done.\n\n')
    return seasonal_df, residue_df


def get_model_CV_report(residue_df, regressor_num=1, forecast_k=1, past_lags=10, k_fold=5):
    '''
    Input:
    residue_df: residue dataframe
    regressor_num: 1-random forest regression, 2-gradient boosting regression, 3-support vector regression (default: 1)
    forecast_k: Future forecasts weeks on residue_df  (default: 1)
    past_lags: Number of lags to consider for regression (default: 10)
    k_fold: cross validation k-fold
    '''
    lags = pd.DataFrame()
    residue_df = residue_df.dropna()
    for i in range(past_lags+forecast_k,forecast_k-1,-1):
        lags['t-'+str(i)] = residue_df["y"].shift(i)
    lags['t'] = residue_df['y'].values
    lags = lags[past_lags+forecast_k+1:]
    lags.reset_index(drop=True,inplace=True)

    X = lags.loc[:,[col for col in set(lags.columns)-set('t')]]
    y = lags.loc[:,'t']

    # define k-fold object
    kf = KFold(n_splits=k_fold, random_state=None)

    # select regressor object
    if regressor_num==1:
        print("RandomForestRegressor selected.")
        model = RandomForestRegressor(n_estimators=500, random_state=0)
    elif regressor_num==2:
        print("GradientBoostingRegressor selected.")
        model = GradientBoostingRegressor(random_state=0)
    elif regressor_num==3:
        print("SupportVectorRegressor selected.")
        model = SVR(kernel='rbf')
    else:
        print("Check model selection again.")

    # run cross validation
    mae_score = []
    for train_index, test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        model = RandomForestRegressor(n_estimators=500, random_state=1)
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        
        mae = mean_absolute_error(pred_values , y_test)
        mae_score.append(mae)
    avg_mae_score = sum(mae_score)/k_fold
    print(f'Number of weeks forecasted:{forecast_k}    Avg MAE:{avg_mae_score}')
    return model, past_lags, avg_mae_score


def forecast_residue(residue_df, model, past_lags):
    '''
    Input:
    A residue DataFrame object with the y column of the last n rows left as numpy.nan (n=6 in this case)

    Output:
    The residue DataFrame object with the y column of the last n rows filled with appropriate predicted values.
    '''
    
    feature_names = ['t-'+str(i+1) for i in range(past_lags+1)]
    feature_names.reverse()

    idx = np.where(np.isnan(residue_df['y']))[0]
    y_test = residue_df.dropna()['y'].tolist()
    y_pred = []

    for i in idx:    
        features = y_test[i-past_lags-1:i]
        feature_df = pd.DataFrame(features).T
        feature_df.columns = feature_names
        y_pred = model.predict(feature_df)[0]
        y_test.append(y_pred)
    residue_df['y']=pd.Series(y_test)

    print(residue_df.tail(10))
    print("Residue forecasts done.")
    return residue_df

    
if __name__ == '__main__':

    ## Input arguements
    csv_path = str(sys.argv[1])
    n = int(sys.argv[2])

    ## 1. Data Engineering
    try:
        ts_df = data_preprocess(csv_path)
    except:
        print("Invalid input file format or file doesn't exist.")
    
    ## 2. Seasonality and Trend Detection
    seasonal_df,residue_df = get_seasonality_trend_residue(ts_df,n)

    ## 3. Model Comparisons
    model, past_lags, acc = get_model_CV_report(residue_df,regressor_num=1,forecast_k=1)
    model, past_lags, acc = get_model_CV_report(residue_df,regressor_num=1,forecast_k=6)

    ## 4. Residue Prediction    
    residue_df = forecast_residue(residue_df, model, past_lags)
    print("--- END OF SCRIPT ---")