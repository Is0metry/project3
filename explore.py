'''Explore contains helper functions to assist in exploration portion of final_report.ipynb'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy.stats import kruskal, levene, spearmanr


def p_to_md(p:float,alpha:float=.05,**kwargs)->md:
    '''
    returns the result of a p test as a `IPython.display.Markdown`
    ## Parameters
    p: `float` of the p value from performed Hypothesis test
    alpha: `float` of alpha value for test, defaults to 0.05
    kwargs: any additional return values of statistical test
    ## Returns
    formatted `Markdown` object containing results of hypothesis test.

    '''
    ret_str = ''
    p_flag = p < alpha
    for k,v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f'## Because $\\alpha$ {">" if p_flag else "<"} p,' + \
        f'we {"failed to " if ~(p_flag) else ""} reject $H_0$'
    return md(ret_str)
def t_to_md(p:float,t:float,alpha:float=.05,**kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-TEst
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p < alpha
    ret_str += f'## t = {t} \n\n'
    for k,v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p = {p} \n\n'
    ret_str +=\
         f'## Because t {">" if t_flag else "<"} 0 and $\\alpha$ {">" if p_flag else "<"} p,' + \
            f'we {"failed to " if ~(t_flag & p_flag) else ""} reject $H_0$'
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
    return p_to_md(p,alpha,r=r)
def plot_1(train:pd.DataFrame)->None:
    '''
    plots Calculated Square Feet vs Tax Value as well as a regression line.
    ## Parameters
    train: `pd.DataFrame` of unscaled training data
    ## Returns
    None
    '''
    sns.lmplot(data=train,x='calc_sqft',y='tax_value',scatter_kws={'color':'#40b7ad'},line_kws=\
    {'color':'#2E1F3B'})
    plt.show()

def levene_test(train:pd.DataFrame)->md:
    '''
    performs a levene test to check variance of train
    ## Parameters
    train: `pandas.DataFrame` of unscaled training data
    ## Returns
    an `IPython.display.Markdown` object containing results of levene test
    '''
    mu1 = train[train.fips == 6037].tax_value
    mu2 = train[train.fips == 6059].tax_value
    mu3 = train[train.fips == 6111].tax_value
    t,p = levene(mu1,mu2,mu3)
    return t_to_md(p,t)

def fips_v_tax(train:pd.DataFrame)->md:
    '''
    performs an Kruskal-Wallis test on training data set, grouped by FIPS code
    ## Parameters
    train: `pd.DataFrame` of training data set to be tested on
    ## Returns
    `IPython.display.Markdown` object with results of Kruskal-Wallis test.
    '''
    mu1 = train[train.fips == 6037].tax_value
    mu2 = train[train.fips == 6059].tax_value
    mu3 = train[train.fips == 6111].tax_value
    t,p = kruskal(mu1,mu2,mu3)
    return t_to_md(p,t)

def obligatory_pie_chart(train:pd.DataFrame)-> None:
    '''
    An obligatory pie chart to upset one of my instructors. Shows percentage of houses
    by FIPS code
    ## Parameters
    train: `pandas.DataFrame` object of training data
    ## Returns
    None
    '''
    fips = train.fips.value_counts().sort_values()
    plt.pie(fips,autopct='%1.0f',colors=['#2E1F3B','#348EA7','#8CDAB2'])
    plt.title('FIPS code in training data')
    plt.legend(train.fips.unique())
    plt.show()
def fips_graph(train:pd.DataFrame)->None:
    '''
    plots a histogram of property Tax Value vs their FIPS code
    ## Parameters
    train: `pandas.DataFrame` of training data
    ## Returns
    None
    '''
    sns.barplot(data=train,x='fips',y='tax_value',palette='mako')
    plt.xlabel('FIPS Code')
    plt.ylabel('Tax Value')
    plt.title('Avg. Tax Value by FIPS code')
    plt.show()

def bedrooms_v_tax_value(train:pd.DataFrame)->None:
    '''
    plots Bar Plot of the mean Tax Value of houses, separated by Bedroom Count
    ## Parameters
    train: `pandas.DataFrame` of training data
    ## Returns
    None
    '''
    sns.barplot(data=train,x='bed_count',y='tax_value')
def year_v_tax_value(train:pd.DataFrame)-> None:
    '''
    plots a linear regression plot of property Tax Value vs the Year Built
    ## Parameters
    train: `pandas.DataFrame` of training data
    ## Returns
    None
    '''
    sns.lmplot(data=train,x='year_built',y='tax_value',line_kws={'color':'#2e1e3b'},\
        scatter_kws={'color':'#40b7ad'})
