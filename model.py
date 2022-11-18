import numpy as np
import pandas as pd
import evaluate as e
import typing as t
from IPython.display import Markdown as md
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.feature_selection import RFE
def select_baseline(ytrain:pd.Series)->pd.DataFrame:
    med_base = ytrain.median()
    mean_base = ytrain.mean()
    mean_eval = e.regression_errors(ytrain,mean_base,'Mean Baseline')
    med_eval = e.regression_errors(ytrain,med_base,'Median Baseline')
    ret_md = pd.concat([mean_eval,med_eval]).to_markdown()
    ret_md += '\n### Because mean outperformed median on all metrics, we will use mean as our baseline'
    return md(ret_md)
def linear_regression(x:pd.DataFrame,y:pd.DataFrame,linreg:t.Union[LinearRegression,None]=None)->None:
    if linreg is None:
        linreg = LinearRegression(normalize=True)
        linreg.fit(x,y)
    ypred = linreg.predict(x)
    return ypred
        
