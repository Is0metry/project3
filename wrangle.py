'''wrangle contains helper functions to assist in data acquisition and preparation
in final_report.ipynb'''
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from env import HOST, PASSWORD, USERNAME


def get_zillow_url() -> str:
    '''
    returns URL for `zillow` database.
    # Parameters
    None
    # Returns
    A formatted pymysql URL with the `HOST`, `USERNAME`, and `PASSWORD` \
        imported from env.py using `zillow` database.
    '''
    return f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/zillow'


def get_zillow_from_sql() -> pd.DataFrame:
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
        fips
        FROM properties_2017
        JOIN predictions_2017 USING(parcelid)
        JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc= "Single Family Residential"'''
    return pd.read_sql(query, get_zillow_url())


def df_from_csv(path: str) -> Union[pd.DataFrame, None]:
    '''
    returns zillow DataFrame from .csv if it exists at `path`, otherwise returns None
    # Parameters
    path: string with path to .csv file
    # Returns
    `pd.DataFrame` if file exists at `path`, otherwise returns `None`.
    '''
    if os.path.exists(path):
        return pd.read_csv(path)

    return None


def wrangle_zillow(from_sql: bool = False, from_csv: bool = False) -> pd.DataFrame:
    '''
    wrangles Zillow data from either a MySQL query or a `.csv` file, prepares the  (if necessary)\
        , and returns a `pandas.DataFrame` object
    containing the prepared Zillow data. If data is acquired from MySQL,
     return `DataFrame` is also encoded to `.csv` in both prepared and
    unprepared states
    ## Parameters
    refresh: if `True`, ignores any `.csv` files and pulls new data from the SQL database,
    default=False.
    ## Return
    parsed and prepared `pandas.DataFrame` with Zillow data from 2017.
    '''
    # aquire Zillow data from .csv if exists
    ret_df = None
    if not from_sql and not from_csv:
        ret_df = df_from_csv('data/prepared_zillow.csv')
        if ret_df is not None:
            return ret_df
    if not from_sql:
        ret_df = df_from_csv('data/zillow.csv')
    if ret_df is None:
        # acquire zillow data from MySQL and caches to data/zillow.csv∏
        ret_df = get_zillow_from_sql()
        ret_df.to_csv('data/zillow.csv', index_label=False)
    ret_df = prep_zillow(ret_df)
    ret_df.to_csv('data/prepared_zillow.csv', index_label=False)

    return ret_df


def prep_zillow(zillow_df: pd.DataFrame) -> pd.DataFrame:
    '''prepares zillow data set for modeling
    ## Parameters
    df: `DataFrame` containing zillow data set
    ## Returns
    prepared Zillow DataFrame
    '''
    # drop na values
    zillow_df = zillow_df.dropna(subset=['year_built', 'fips', 'tax_value', 'calc_sqft'])
    # drop any houses with 0 beds or 0 baths. That's not a house, that's a shed.
    zillow_df = zillow_df[zillow_df.bed_count > 0]
    zillow_df = zillow_df[zillow_df.bath_count > 0]
    zillow_df = zillow_df[zillow_df.tax_value > 25000]
    zillow_df = zillow_df.drop_duplicates()
    # Fill values where full_bath is NaN with number of whole baths
    # Done because none of these houses have 3/4 baths,
    #  and no information is available on full baths
    zillow_df = zillow_df.reset_index(drop=True)
    na_bath = zillow_df[zillow_df.full_baths.isna()].index
    zillow_df.iloc[na_bath, 3] = np.floor(zillow_df.iloc[na_bath, 2])
    # set bath_count to be half_bath_count
    zillow_df['half_baths'] = (zillow_df.bath_count-zillow_df.full_baths) * 2
    zillow_df.full_baths = zillow_df.full_baths.astype(int)
    zillow_df.half_baths = zillow_df.half_baths.astype(int)
    zillow_df.bed_count = zillow_df.bed_count.astype(int)
    zillow_df.year_built = zillow_df.year_built.astype(int)
    zillow_df.fips = zillow_df.fips.astype(int)
    # discard outliers
    zillow_df = zillow_df[zillow_df.tax_value < 3500000]
    zillow_df = zillow_df[zillow_df.bed_count < 7]
    zillow_df = zillow_df[zillow_df.half_baths != zillow_df.half_baths.max()]
    zillow_df = zillow_df.drop(columns=['bath_count'])
    zillow_df.reset_index(drop=True)
    dummy_df = pd.get_dummies(zillow_df.fips,prefix='fips')
    zillow_df = pd.concat([zillow_df,dummy_df],axis=1)
    return zillow_df


def tvt_split(dframe: pd.DataFrame, stratify: Union[str, None] = None,
              tv_split: float = .2, validate_split: float = .3, \
                sample: Union[float, None] = None) -> \
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''tvt_split takes a pandas DataFrame, a string specifying the variable to stratify over,
    as well as 2 floats where 0< f < 1 and
    returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    train_validate, test = train_test_split(
        dframe, test_size=tv_split, random_state=123, stratify=stratify)
    train, validate = train_test_split(
        train_validate, test_size=validate_split, random_state=123, stratify=stratify)
    if sample is not None:
        train = train.sample(frac=sample)
        validate = validate.sample(frac=sample)
        test = test.sample(frac=sample)
    return train, validate, test


def get_scaled_copy(dframe: pd.DataFrame, x: List[str], scaled_data: np.ndarray) -> pd.DataFrame:
    '''copies `df` and returns a DataFrame with `scaled_data`
    ## Parameters
    df: `DataFrame` to be copied and scaled
    x: features in `df` to be scaled
    scaled_data: `np.ndarray` with scaled values
    ## Returns
    a copy of `df` with features replaced with `scaled_data`
    '''
    ret_df = dframe.copy()
    ret_df.loc[:, x] = scaled_data
    return ret_df


def scale_data(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame,
               x: List[str]) ->\
        Tuple[RobustScaler,pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    scales `train`,`validate`, and `test` data using a `method`
    ## Parameters
    train: `pandas.DataFrame` of training data
    validate: `pandas.DataFrame` of validate data
    test: `pandas.DataFrame` of test data
    x: list of str representing feature columns in the data
    method: `callable` of scaling function (defaults to `sklearn.RobustScaler`)
    ## Returns
    a tuple of scaled copies of train, validate, and test.
    '''
    xtrain = train[x]
    xvalid = validate[x]
    xtest = test[x]
    scaler = RobustScaler()
    scaler.fit(xtrain)
    scale_train = scaler.transform(xtrain)
    scale_valid = scaler.transform(xvalid)
    scale_test = scaler.transform(xtest)
    ret_train = get_scaled_copy(train, x, scale_train)
    ret_valid = get_scaled_copy(validate, x, scale_valid)
    ret_test = get_scaled_copy(test, x, scale_test)
    return scaler, ret_train, ret_valid, ret_test


if __name__ == "__main__":
    df = wrangle_zillow()
    print(df.info())
