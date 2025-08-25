import pandas as pd
from typing import Dict

def split_primary_data(
    features_m1: pd.DataFrame,
    features_m2: pd.DataFrame,
    params: Dict
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Splits feature data into development and test sets based on holdout days."""
    holdout_days = params["holdout_days"]
    
    # Split M1 features
    last_date_m1 = features_m1.index.max()
    split_date_m1 = last_date_m1 - pd.Timedelta(days=holdout_days)
    m1_dev = features_m1[features_m1.index <= split_date_m1]
    m1_test = features_m1[features_m1.index > split_date_m1]
    
    # Split M2 features
    last_date_m2 = features_m2.index.max()
    split_date_m2 = last_date_m2 - pd.Timedelta(days=holdout_days)
    m2_dev = features_m2[features_m2.index <= split_date_m2]
    m2_test = features_m2[features_m2.index > split_date_m2]
    
    return m1_dev, m1_test, m2_dev, m2_test
