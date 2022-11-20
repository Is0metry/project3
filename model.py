'''model contains helper funmctions to assist in Modeling portion of final_report.ipynb'''
import typing as t

import evaluate as e
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from numpy.typing import ArrayLike
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error

DataType = t.Union[ArrayLike, pd.Series,pd.DataFrame]
PandasDataType = t.Union[pd.Series,pd.DataFrame]
ModelType = t.Union[LinearRegression,LassoLars,TweedieRegressor]
def select_baseline(ytrain:pd.Series)->md:
    '''tests mean and median of training data as a baseline metric.
    ## Parameters
    ytrain: `pandas.Series` containing the target variable
    ## Returns
    Formatted `Markdown` with information on best-performing baseline.
    '''
    med_base = ytrain.median()
    mean_base = ytrain.mean()
    mean_eval = e.regression_errors(ytrain,mean_base,'Mean Baseline')
    med_eval = e.regression_errors(ytrain,med_base,'Median Baseline')
    ret_md = pd.concat([mean_eval,med_eval]).to_markdown()
    ret_md += '\n### Because mean outperformed median on all metrics, \
        we will use mean as our baseline'
    return md(ret_md)
def linear_regression(x:pd.DataFrame,y:pd.DataFrame,\
    linreg:t.Union[LinearRegression,None]=None)->None:
    '''runs linear regression on x and y
    ## Parameters
    x: DataFrame of features

    y: DataFrame of target

    linreg: Optional `LinearRegression` object,
    used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    linreg: linear regression model trained on data.
    '''
    if linreg is None:
        linreg = LinearRegression(normalize=True)
        linreg.fit(x,y)
    ypred = linreg.predict(x)
    return ypred,linreg
def lasso_lars(x:pd.DataFrame,y:pd.DataFrame,llars:t.Union[None,LassoLars] = None)\
    ->t.Tuple[np.array,LassoLars]:
    '''runs LASSO+LARS on x and y
    ## Parameters
    x: Dataframe of features

    y: DataFrame of target

    llars: Optional LASSO + LARS object, used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    linreg: `LassoLars` model trained on data.
    '''
    if llars is None:
        llars = LassoLars(alpha=3.0)
        llars.fit(x,y)
    ypred = llars.predict(x)
    return ypred,llars
def lgm(x:pd.DataFrame,y:pd.DataFrame, tweedie:t.Union[TweedieRegressor,None] = None)\
    ->t.Tuple[np.array,TweedieRegressor]:
    '''runs Generalized Linear Model (GLM) on x and y
    ## Parameters
    x: `DataFrame` of features

    y: 'DataFrame' of target

    tweedie: `TweedieRegressor` object, used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    tweedie: `TweedieRegressor` model trained on data.
    '''
    if tweedie is None:
        tweedie = TweedieRegressor(power=0,alpha=3.0)
        tweedie.fit(x,y)
    ypred = tweedie.predict(x)
    return ypred, tweedie
def rmse_eval(ytrue:t.Dict[str,np.array],**kwargs)->pd.DataFrame:
    '''
    performs Root Mean Squared evaluation on parameters
    ## Parameters
    ytrue: a dictionary of `numpy.array` containing the true Y values on which to evaluate
    kwargs: named dictionary of `numpy.array` objects
        with predicted y values where for each key:value
    pain in ytrue there is a corresponding key:value pair in `kwargs[REGRESSION FUNCTION NAME]`
    ## Returns
    a `pandas.DataFrame` of Root Mean Squared Evaluation for each dataset in kwargs.
    '''
    ret_df = pd.DataFrame()
    for key,value in kwargs.items():
        for k_key,v_value in value.items():
            ret_df.loc[key,k_key] = np.round(np.sqrt(mean_squared_error(ytrue[k_key],v_value)),2)
    return ret_df
def evaluate_models(xtrain:pd.DataFrame,ytrain:pd.DataFrame,\
    xvalid:pd.DataFrame,yvalid:pd.DataFrame)->pd.DataFrame:
    '''
    Performs LASSO+LARS, Linear Regression, and Generalized Linear Model regression
        on train and validate data sets provided
    ## Parameters
    xtrain: `pandas.DataFrame` containing x values from training dataset
        on which to perform regression fit and prediction
    ytrain: `pandas.DataFrame` containing y values on which to perform regression prediction
    xvalid: `pandas.DataFrame` containing x values from validate data set
        on which to perform regression predictions
    yvalid: `pandas.DataFrame` containing y values from
        validate data set on which to perform regression predictions
    ## Returns
    `IPython.display.Markdown` object containing table of Reverse Mean Squared Error
        evaluation of train and validate.
    data sets with each of the models.

    `ModelType` which performed the best against both data sets (LASSO+LARS)
    '''
    ptrain_linreg, linreg = linear_regression(xtrain,ytrain)
    pvalid_linreg, _ = linear_regression(xvalid,yvalid,linreg)
    ptrain_llars,llars = lasso_lars(xtrain,ytrain)
    pvalid_llars,_ = lasso_lars(xvalid,yvalid,llars)
    ptrain_lgm, tweedie = lgm(xtrain,ytrain)
    pvalid_lgm,_ = lgm(xvalid,yvalid,tweedie)
    ytrue = {'train':ytrain,'validate':yvalid}
    btrain = np.full_like(np.arange(xtrain.shape[0],dtype=int),ytrain.tax_value.mean())
    bvalid = np.full_like(np.arange(xvalid.shape[0],dtype=int),ytrain.tax_value.mean())
    linreg_pred = {'train':ptrain_linreg,'validate':pvalid_linreg}
    llars_pred = {'train':ptrain_llars,'validate':pvalid_llars}
    lgm_pred = {'train':ptrain_lgm,'validate':pvalid_lgm}
    baseline_pred = {'train':btrain,'validate':bvalid}
    evaluation_matrix = rmse_eval(ytrue,baseline=baseline_pred,linear_regression=linreg_pred,\
        lasso_lars=llars_pred,lgm=lgm_pred)
    evaluation_matrix.index= ['Baseline','Linear Regression','LASSO LARS','General Linear Model']
    evaluation_matrix.columns = ['Train RMSE','Validate RMSE']
    return md('| Methodology' + evaluation_matrix.to_markdown()[1:]),llars
def run_test(model:ModelType,xtest:pd.DataFrame,ytest:pd.DataFrame)->md:
    '''
    Runs best performing regression model on test data set.
    ## Parameters
    model: `ModelType` of best performing model on train and validate data sets
    xtest: Features of test data set

    ytest: Target of test data set
    ## Returns
    displays a chart of residual plot of test results

    `IPython.display.Markdown` object containing the Root Mean Squared Error
        of predictions on test data set.
    '''
    ypred = model.predict(xtest)
    plot_residuals(ytest.tax_value,ypred)
    return md('### Mean Squared Error: ' + \
        str(np.round(np.sqrt(mean_squared_error(ytest,ypred)),2)))

def get_residuals(y_true:pd.Series,y_pred:t.Union[pd.Series,float]):
    '''
    gets residuals and residual squared values for ytrue and ypred
    ## Parameters
    y_true: `pandas.DataFrame
    ## Returns
    `pandas.DataFrame` of actual value, residual, and residual squared.
    '''
    ret_frame = pd.DataFrame()
    ret_frame['actual'] = y_true
    ret_frame['residual'] = y_true - y_pred
    ret_frame['residual_squared'] = ret_frame.residual ** 2
    return ret_frame

def plot_residuals(y_true:DataType,y_pred:DataType)->None:
    '''
    Plots the residuals of y_pred vs y_true
    ## Parameters
    y_true: `DataType` of true y values.
    y_pred: `DataType` of predicted y values.
    ## Returns
    None
    '''
    res = get_residuals(y_true,y_pred)
    sns.scatterplot(data=res,x='actual',y='residual')
    plt.axhline(0,0,1)
    plt.show()
