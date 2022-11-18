import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy.stats import f_oneway, spearmanr


def p_to_md(p:float,alpha:float,**kwargs):
    '''takes a p-value, alpha, and any T-test arguments and creates a Markdown object with the information
    '''
    ret_str = ''
    reject_flag = p < alpha
    for k in kwargs:
        ret_str += f'## {k} = {kwargs[k]}\n\n'
    ret_str += f'## p = {p}\n\n'
    ret_str += f'## Because $\\alpha$ {">" if reject_flag else "<"} p,' + \
            f'we {"failed to " if ~reject_flag else ""} reject $H_0$'
    return md(ret_str)
def tax_sqft_test(train:pd.DataFrame,alpha:float=0.05)->md:
    '''
    performs a spearman correlation test between tax_value and calc_sqft
    ## Parameters
    train: training DataFrame with information from Zillow
    
    alpha: ⍺ value for test, defaults to ɑ =.05
    ## Returns
    IPython.display.Markdown object with p and r values, and a statement regarding whether \
    H0 is rejected.
    '''
    x = train.calc_sqft
    y = train.tax_value
    r,p = spearmanr(x,y)
    reject_flag = p < alpha
    return p_to_md(p,alpha,r=r)
def fips_v_tax(train:pd.DataFrame, alpha:float =.05)->md:
    mu1 = train[train.fips == 6037].tax_value
    mu2 = train[train.fips == 6059].tax_value
    mu3 = train[train.fips == 6111].tax_value
    t,p = f_oneway(mu1,mu2,mu3)
    return p_to_md(p,alpha,t=t)
def plot_1(train:pd.DataFrame)->None:
    sns.lmplot(data=train,x='calc_sqft',y='tax_value',scatter_kws={'color':'#37659E'},line_kws=\
    {'color':'#2E1F3B'})
    plt.show()



def p_to_md(p:float,alpha:float,**kwargs):
    ret_str = ''
    reject_flag = p < alpha
    for k in kwargs:
        ret_str += f'## {k} = {kwargs[k]}\n\n'
    ret_str += f'## p = {p}\n\n'
    ret_str += f'## Because $\\alpha$ {">" if reject_flag else "<"} p,' + \
            f'we {"failed to " if ~reject_flag else ""} reject $H_0$'
    return md(ret_str)
def tax_sqft_test(train:pd.DataFrame,alpha:float=0.05)->md:
    '''
    performs a spearman correlation test between tax_value and calc_sqft
    ## Parameters
    train: training DataFrame with information from Zillow
    
    alpha: ⍺ value for test, defaults to ɑ =.05
    ## Returns
    IPython.display.Markdown object with p and r values, and a statement regarding whether \
    H0 is rejected.
    '''
    x = train.calc_sqft
    y = train.tax_value
    r,p = spearmanr(x,y)
    reject_flag = p < alpha
    return p_to_md(p,alpha,r=r)
def obligatory_pie_chart(train:pd.DataFrame)-> None:
    fips = train.fips.value_counts().sort_values()
    plt.pie(fips,autopct='%1.0f',colors=['#2E1F3B','#348EA7','#8CDAB2'])
    plt.legend(train.fips.unique())
    plt.show()
def lat_long(train:pd.DataFrame):
    sns.scatterplot(data=train,x='latitude',y='longitude',hue='fips')
    plt.show()