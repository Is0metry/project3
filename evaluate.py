import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import typing

def plot_residuals(y_true:pd.Series,y_pred:pd.Series)->None:
    res = get_residuals(y_true,y_pred)
    sns.scatterplot(data=res,x='actual',y='residual')
    plt.axhline(0,0,1)
    plt.show()
def get_residuals(y_true:pd.Series,y_pred:typing.Union[pd.Series,float]):
    ret_frame = pd.DataFrame()
    ret_frame['actual'] = y_true
    ret_frame['residual'] = y_true - y_pred
    ret_frame['residual_squared'] = ret_frame.residual ** 2
    return ret_frame
def sum_of_squared_errors(y_true:pd.Series,y_pred:typing.Union[pd.Series,float]):
    ret_frame = get_residuals(y_true,y_pred)
    sse = sum(ret_frame.residual)
    return sse
def explained_sum_of_sqrd(y_true:pd.Series,y_pred:pd.Series):
    return sum((y_pred - y_true.mean())**2)
def total_sum_of_squares(y_true:pd.Series,y_pred:typing.Union[pd.Series,float]):
    return explained_sum_of_sqrd(y_true,y_pred) + sum_of_squared_errors(y_true,y_pred)
def regression_errors(y_true,y_pred,title):
    ret_dict = {}
    ret_dict['SSE'] = sum_of_squared_errors(y_true,y_pred)
    if not isinstance(y_pred,float):
        ret_dict['ESS'] = explained_sum_of_sqrd(y_true,y_pred)
        ret_dict['TSS'] = ret_dict['SSE'] + ret_dict['ESS']
        ret_dict['MSE'] = mean_squared_error(y_true,y_pred)
    else:
        ret_dict['MSE'] = mean_squared_error(y_true,[y_pred for i in range(y_true.count())])
    ret_dict['RMSE']= np.sqrt(ret_dict['MSE'])
    ret_frame =  pd.DataFrame(ret_dict,index=[title])
    return ret_frame
def baseline_mean_errors(y_true:pd.Series, baseline:typing.Union[float,int])->pd.DataFrame:
    return regression_errors(y_true,baseline,'Baseline')