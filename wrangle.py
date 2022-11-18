from env import HOST,USERNAME,PASSWORD
from typing import Union
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import enum
def get_zillow_url()->str:
    '''
    returns URL for `zillow` database.
    # Parameters
    None
    # Returns
    Returns a formatted pymysql URL with the `HOST`, `USERNAME`, and `PASSWORD` imported from env.py using `zillow` database.
    '''
    return f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/zillow'

def get_zillow_from_sql()->pd.DataFrame:
    '''
    reads MySQL data from `zillow` database and returns `pandas.DataFrame` with raw data from query
    # Parameters
    None
    # Returns
    parsed DataFrame containing raw data from `zillow` database.
    '''
    query = '''
    SELECT taxvaluedollarcnt AS tax_value,
        bedroomcnt AS bed_count, 
        bathroomcnt AS bath_count, 
        fullbathcnt AS full_baths,
        calculatedfinishedsquarefeet AS calc_sqft,
        yearbuilt AS year_built,
        fips,
        latitude,
        longitude
        FROM properties_2017
        JOIN predictions_2017 USING(parcelid)
        JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc= "Single Family Residential"'''
    return pd.read_sql(query,get_zillow_url())
def df_from_csv(path:str)->Union[pd.DataFrame,None]:
    '''
    returns zillow DataFrame from .csv if it exists at `path`, otherwise returns None
    # Parameters
    path: string with path to .csv file
    # Returns
    `pd.DataFrame` if file exists at `path`, otherwise returns `None`.
    '''
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None
class RefreshType(enum.Flag):
    SQL = enum.auto()
    Reprepare = enum.auto()
    none = enum.auto()
    
def wrangle_zillow(flag:str='none')->pd.DataFrame:
    '''
    wrangles Zillow data from either a MySQL query or a `.csv` file, prepares the  (if necessary), and returns a `pandas.DataFrame` object
    containing the prepared Zillow data. If data is acquired from MySQL, return `DataFrame` is also encoded to `.csv` in both prepared and
    unprepared states
    ## Parameters
    refresh: if `True`, ignores any `.csv` files and pulls new data from the SQL database. default=False 
    ## Return
    parsed and prepared `pandas.DataFrame` with Zillow data from 2017.
    '''
    #aquire Zillow data from .csv if exists
    df = None
    if flag.lower() == 'none':
        df = df_from_csv('data/prepared_zillow.csv')
        if df is not None:
            return df
    if flag != 'sql':
        df = df_from_csv('data/zillow.csv')
    if df is None:
        #acquire zillow data from MySQL and caches to data/zillow.csv
        df = get_zillow_from_sql()
        df.to_csv('data/zillow.csv',index_label=False)
    df = prep_zillow(df)
    df.to_csv('data/prepared_zillow.csv',index_label=False)

    return df
def prep_zillow(df:pd.DataFrame)->pd.DataFrame:
    '''prepares zillow data set for modeling
    ## Parameters
    df: `DataFrame` containing zillow data set
    ## Returns
    prepared Zillow DataFrame
    '''
    #drop na values
    df = df.dropna(subset=['year_built','fips','tax_value','calc_sqft'])
     # drop any houses with 0 beds or 0 baths. That's not a house, that's a shed.
    df = df[df.bed_count > 0]
    df = df[df.bath_count > 0]
    df = df.drop_duplicates()
    # Fill values where full_bath is NaN with number of whole baths
    # Done because none of these houses have 3/4 baths, and no information is available on full baths
    df = df.reset_index(drop=True)
    na_bath = df[df.full_baths.isna()].index
    df.iloc[na_bath,3] = np.floor(df.iloc[na_bath,2])
    #set bath_count to be half_bath_count
    df['half_baths'] = (df.bath_count-df.full_baths) * 2
    df.full_baths = df.full_baths.astype(int)
    df.half_baths = df.half_baths.astype(int)
    df.bed_count = df.bed_count.astype(int)
    df.year_built = df.year_built.astype(int)
    df.fips = df.fips.astype(int)
    #discard outliers
    df = df[df.tax_value < 3500000]
    df = df[df.bed_count <9]
    df = df[df.bath_count < 8]
    df = df[df.half_baths != df.half_baths.max()]
    df.reset_index(drop=True)

    return df
def tvt_split(df:pd.DataFrame,stratify:str = None,tv_split:float = .2,validate_split:float= .3,sample:float = None):
    '''tvt_split takes a pandas DataFrame, a string specifying the variable to stratify over,
    as well as 2 floats where 0< f < 1 and returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    train_validate, test = train_test_split(df,test_size=tv_split,random_state=123,stratify=stratify)
    train, validate = train_test_split(train_validate,test_size=validate_split,random_state=123,stratify=stratify)
    if sample is not None:
        train = train.sample(frac=sample)
        validate = validate.sample(frac=sample)
        test = test.sample(frac=sample)
    return train,validate,test

if __name__ == "__main__":
    df = wrangle_zillow()
    print(df.info())