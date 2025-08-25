import pandas as pd
import numpy as np
import holidays
import warnings

def preprocess_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw loan data.
    - Converts date column.
    - Removes duplicates.
    - Aggregates data by day.
    - Calculates the daily difference in total balance.
    """
    df = raw_data.copy()
    df['cob_dt'] = pd.to_datetime(df['cob_dt'], format='mixed')
    
    print(f'Before dropping duplicates: {df.shape}')
    df.drop_duplicates(keep='last', inplace=True)
    print(f'After dropping duplicates: {df.shape}')
    
    df = df.drop(columns=['brn']).groupby('cob_dt').sum()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    df = df[(df.index >= pd.Timestamp('2023-01-01'))]
    df['total_bal_diff'] = df['total_bal'].diff().bfill()
    
    loan_data = df[['total_bal', 'total_bal_diff']]
    loan_data.reset_index(inplace=True)
    
    print("Preprocessing complete.")
    return loan_data


def create_features_for_model1(df_input: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Creates time-series features for Model 1.
    """
    if not isinstance(df_input, pd.DataFrame):
        warnings.warn("Input 'df' is not a pandas DataFrame. Returning None.")
        return None
    if df_input.empty:
        warnings.warn("Input DataFrame 'df' is empty. Returning None.")
        return None

    df = df_input.copy()

    date_column = parameters["date_column"]
    balance_col = parameters["balance_col"]
    target_col_total_bal_diff = parameters["target_col"]

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in input DataFrame for create_features_for_model1.")

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    if df.empty:
        warnings.warn("DataFrame became empty after date processing in create_features_for_model1.")
        return None
    df = df.set_index(date_column).sort_index()
    
    df_features = df.copy()
    original_index_name = df_features.index.name 

    for col_to_convert in [balance_col, target_col_total_bal_diff]:
        df_features[col_to_convert] = pd.to_numeric(df_features[col_to_convert], errors='coerce')
    df_features[balance_col] = df_features[balance_col].ffill().bfill().fillna(0) 
    df_features[target_col_total_bal_diff] = df_features[target_col_total_bal_diff].fillna(0) 
    # --- Calendar Features (largely kept, but ensure robustness) ---
    idx = df_features.index
    df_features['dayofweek'] = idx.dayofweek.astype(np.uint8)
    df_features['dayofmonth'] = idx.day.astype(np.uint8)
    df_features['dayofyear'] = idx.dayofyear.astype(np.uint16)
    df_features['weekofyear'] = idx.isocalendar().week.astype(np.uint8)
    df_features['month'] = idx.month.astype(np.uint8)
    df_features['quarter'] = idx.quarter.astype(np.uint8)
    df_features['year'] = idx.year.astype(np.int16)
    df_features['is_month_start'] = (df_features['dayofmonth'] <= 3).astype(np.uint8)
    df_features['is_month_end_prox'] = (df_features.groupby(idx.to_period('M'))['dayofmonth'].transform('max') - df_features['dayofmonth'] <= 3).astype(np.uint8) 
    df_features['is_quarter_end_prox'] = (df_features.groupby(idx.to_period('Q'))['dayofyear'].transform('max') - df_features['dayofyear'] <= 7).astype(np.uint8) 

    near_end_threshold = 7 
    unique_years = df_features['year'].unique()
    valid_years = [y for y in unique_years if pd.notna(y) and isinstance(y, (int, np.integer))] 
    if len(valid_years) > 0:
        try:
            vn_holidays = holidays.Vietnam(years=valid_years, observed=True)
            if vn_holidays: 
                vn_holidays_set = set(pd.to_datetime(list(vn_holidays.keys())).tz_localize(None)) 
                df_features['is_holiday'] = idx.isin(vn_holidays_set).astype(np.uint8)
                
                min_days_to_holiday = pd.Series(np.iinfo(np.uint8).max, index=idx, dtype=np.uint8)
                for h_date in sorted(list(vn_holidays_set)): 
                    delta_days = (h_date - idx).days
                    min_days_to_holiday = np.where((delta_days >= 0) & (delta_days <= near_end_threshold) & (delta_days < min_days_to_holiday),
                                                   delta_days, min_days_to_holiday).astype(np.uint8)
                df_features['days_until_holiday'] = min_days_to_holiday
                df_features.loc[df_features['days_until_holiday'] == np.iinfo(np.uint8).max, 'days_until_holiday'] = 0 
            else:
                df_features['is_holiday'] = 0; df_features['days_until_holiday'] = 0
        except Exception as e:
            warnings.warn(f"Could not process holidays: {e}. Holiday features set to 0.")
            df_features['is_holiday'] = 0; df_features['days_until_holiday'] = 0
    else:
        df_features['is_holiday'] = 0; df_features['days_until_holiday'] = 0

    cond1 = (df_features['dayofmonth'] == 25) & (df_features['dayofweek'] < 5)
    cond2 = (df_features['dayofmonth'] == 26) & (df_features['dayofweek'] == 0)
    cond3 = (df_features['dayofmonth'] == 27) & (df_features['dayofweek'] == 0)
    df_features['is_payment_day'] = (cond1 | cond2 | cond3).astype(np.uint8)
    # Cyclical features (kept)
    year_days = np.where(df_features.index.is_leap_year, 366.0, 365.0)
    df_features['sin_dayofyear'] = np.sin(2 * np.pi * df_features['dayofyear'] / year_days)
    df_features['cos_dayofyear'] = np.cos(2 * np.pi * df_features['dayofyear'] / year_days)
    df_features['sin_dayofweek'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7.0)
    df_features['cos_dayofweek'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7.0)
    df_features['sin_month'] = np.sin(2 * np.pi * df_features['month'] / 12.0) 
    df_features['cos_month'] = np.cos(2 * np.pi * df_features['month'] / 12.0) 
    
    # --- Lag Features ---
    lag_periods_short = [1, 2, 3, 5, 7] 
    lag_periods_seasonal = [14, 21, 30, 60, 90, 150] 
    all_lags = sorted(list(set(lag_periods_short + lag_periods_seasonal)))
    if 1 not in all_lags: all_lags.insert(0, 1)

    for lag in all_lags:
        df_features[f'delta_lag_{lag}'] = df_features[target_col_total_bal_diff].shift(lag)
        df_features[f'balance_lag_{lag}'] = df_features[balance_col].shift(lag)
    # --- Momentum / Acceleration Features ---
    df_features['delta_diff_lag1'] = df_features[target_col_total_bal_diff].diff(1).shift(1) # Change from t-2 to t-1
    df_features['delta_diff_lag2'] = df_features[target_col_total_bal_diff].diff(1).shift(2) # Change from t-3 to t-2
    # Rate of change over a short window (e.g., 3 days)
    # (delta(t-1) - delta(t-4)) / 3
    if 'delta_lag_1' in df_features and 'delta_lag_4' in df_features: 
         df_features['delta_roc_3d'] = (df_features['delta_lag_1'] - df_features[target_col_total_bal_diff].shift(4)) / 3
    # --- Rolling Window Features (Shifted by 1 day to prevent leakage) ---
    window_sizes = sorted(list(set([3, 5, 7, 14, 30, 60, 90, 150]))) 
    min_periods_ratio = 0.7 
    delta_shifted_for_roll = df_features[target_col_total_bal_diff].shift(1)
    balance_shifted_for_roll = df_features[balance_col].shift(1)

    for w in window_sizes:
        min_p = max(1, int(w * min_periods_ratio))
        # Rolling features for DELTA
        if len(delta_shifted_for_roll.dropna()) >= min_p: 
            rolling_delta = delta_shifted_for_roll.rolling(window=w, min_periods=min_p)
            df_features[f'delta_roll_mean_{w}'] = rolling_delta.mean()
            df_features[f'delta_roll_std_{w}'] = rolling_delta.std() 
            df_features[f'delta_ewm_mean_{w}'] = delta_shifted_for_roll.ewm(span=w, min_periods=min_p, adjust=False).mean()
        # Rolling features for BALANCE 
        if len(balance_shifted_for_roll.dropna()) >= min_p:
            rolling_balance = balance_shifted_for_roll.rolling(window=w, min_periods=min_p)
            df_features[f'balance_roll_mean_{w}'] = rolling_balance.mean()
            df_features[f'balance_roll_std_{w}'] = rolling_balance.std() 
        # Ratio feature 
        balance_lag_1_series = df_features.get('balance_lag_1')
        balance_roll_mean_w_series = df_features.get(f'balance_roll_mean_{w}')
        epsilon = 1e-9

        if balance_lag_1_series is not None and balance_roll_mean_w_series is not None:
            ratio = pd.Series(1.0, index=df_features.index, dtype=np.float64)
            valid_idx = balance_lag_1_series.notna() & balance_roll_mean_w_series.notna()

            ratio.loc[valid_idx] = (
                balance_lag_1_series.loc[valid_idx] /
                (balance_roll_mean_w_series.loc[valid_idx].abs() + epsilon)
                * np.sign(balance_roll_mean_w_series.loc[valid_idx])
            )

            median_roll = balance_roll_mean_w_series.median()
            clip_limit = abs(median_roll) * 10 

            df_features[f'balance_lag1_div_roll_mean_{w}'] = ratio.clip(-clip_limit, clip_limit)

        else:
            df_features[f'balance_lag1_div_roll_mean_{w}'] = 1.0

    # Volatility Ratio 
    std5_series = df_features.get('delta_roll_std_5')
    std30_series = df_features.get('delta_roll_std_30')
    if std5_series is not None and std30_series is not None:
        vol_ratio = pd.Series(1.0, index=df_features.index, dtype=np.float64)
        epsilon = 1e-9
        valid_idx_vol = std5_series.notna() & std30_series.notna()
        vol_ratio.loc[valid_idx_vol] = std5_series.loc[valid_idx_vol] / (std30_series.loc[valid_idx_vol] + epsilon)
        median_vol = std30_series.median()
        clip_limit = abs(median_vol) * 10
        df_features['delta_volatility_ratio_5_30'] = vol_ratio.clip(0, clip_limit) 
    else:
        df_features['delta_volatility_ratio_5_30'] = 1.0

    abs_delta_shifted = df_features[target_col_total_bal_diff].abs().shift(1)
    for w_abs in [5, 14]: 
        min_p_abs = max(1, int(w_abs * min_periods_ratio))
        if len(abs_delta_shifted.dropna()) >= min_p_abs:
            df_features[f'delta_abs_roll_max_{w_abs}'] = abs_delta_shifted.rolling(window=w_abs, min_periods=min_p_abs).max()
            df_features[f'delta_abs_roll_mean_{w_abs}'] = abs_delta_shifted.rolling(window=w_abs, min_periods=min_p_abs).mean()
        else: 
            df_features[f'delta_abs_roll_max_{w_abs}'] = np.nan
            df_features[f'delta_abs_roll_mean_{w_abs}'] = np.nan
 
    # Interaction of recent delta mean with recent volatility
    if 'delta_roll_mean_5' in df_features and 'delta_roll_std_5' in df_features:
        df_features['delta_mean5_x_std5'] = df_features['delta_roll_mean_5'] * df_features['delta_roll_std_5']
    # --- Final Processing ---
    df_features.index.name = original_index_name   

    max_lookback_needed = 0
    if all_lags: max_lookback_needed = max(all_lags)
    
    if window_sizes: max_lookback_needed = max(max_lookback_needed, max(window_sizes) + 1)
    if 'delta_roc_3d' in df_features.columns: max_lookback_needed = max(max_lookback_needed, 4) 
    if 'delta_diff_lag2' in df_features.columns: max_lookback_needed = max(max_lookback_needed, 3)

    if max_lookback_needed > 0:
        if len(df_features) > max_lookback_needed:
            df_features = df_features.iloc[max_lookback_needed:].copy()
        else:
            warnings.warn(f"Input DataFrame (len {len(df_input)}) is shorter than or equal to "
                          f"max_lookback_needed ({max_lookback_needed}). Returning empty DataFrame.")
            empty_idx = pd.DatetimeIndex([], name=original_index_name, freq=df_features.index.freq) # Preserve freq if available
            return pd.DataFrame(columns=df_features.columns).set_index(empty_idx)
    if df_features.empty:
        warnings.warn("DataFrame became empty after feature creation and lookback trimming. Returning None.")
        return None

    # This is a general fillna; specific features might need more tailored imputation.
    numeric_cols = df_features.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col not in [balance_col, target_col_total_bal_diff]: 
            # Heuristic: if 'std' in col or 'volatility' in col, fillna(0) is common
            if 'std' in col.lower() or 'volatility' in col.lower():
                df_features[col] = df_features[col].fillna(0)
            # For ratios, filling with 1 if they represent 'no change' or 'baseline'
            elif 'ratio' in col.lower() or 'div' in col.lower():
                 df_features[col] = df_features[col].fillna(1.0) 
            else:
                df_features[col] = df_features[col].fillna(0)
 
    df_features = df_features.replace([np.inf, -np.inf], 0) 

    print("Feature engineering for Model 1 complete.")
    return df_features


def create_intraday_features_for_model2(df_raw_hourly_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Creates intraday features from hourly data for Model 2."""
    if df_raw_hourly_data is None or df_raw_hourly_data.empty:
        warnings.warn("M2_Features: Input df_raw_hourly_data is None or empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df_raw_hourly_data.copy()
    
    date_column = parameters["date_column"]

    if date_column not in df.columns:
        warnings.warn(f"M2_Features: Date column '{date_column}' missing. Returning empty DataFrame.")
        return pd.DataFrame()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    if df.empty:
        warnings.warn("M2_Features: Hourly DataFrame empty after date conversion/dropna. Returning empty DataFrame.")
        return pd.DataFrame()

    if 'time_point' not in df.columns:
        warnings.warn("M2_Features: 'time_point' column missing. Returning empty DataFrame.")
        return pd.DataFrame()

    if pd.api.types.is_object_dtype(df['time_point']) or pd.api.types.is_string_dtype(df['time_point']):
        try:
            df['hour_of_day'] = pd.to_datetime(df['time_point'], format='%H:%M:%S', errors='coerce').dt.hour
        except ValueError: # Handle cases where some time_points might not be H:M:S
             warnings.warn("M2_Features: Some 'time_point' values are not in H:M:S format. Attempting numeric conversion.")
             df['hour_of_day'] = pd.to_numeric(df['time_point'].str.split(':').str[0], errors='coerce')
    elif pd.api.types.is_numeric_dtype(df['time_point']):
        df['hour_of_day'] = df['time_point']
    else:
        warnings.warn("M2_Features: Unsupported 'time_point' format. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.dropna(subset=['hour_of_day'])
    if df.empty:
        warnings.warn("M2_Features: Hourly DataFrame empty after hour_of_day processing. Returning empty DataFrame.")
        return pd.DataFrame()
    df['hour_of_day'] = df['hour_of_day'].astype(int)

    df = df[(df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 15)]
    if df.empty:
        warnings.warn("M2_Features: No hourly data between 9 AM and 3 PM. Returning empty DataFrame.")
        return pd.DataFrame(index=pd.Index([], name=date_column)) 

    df = df.sort_values(by=[date_column, 'hour_of_day']).drop_duplicates(subset=[date_column, 'hour_of_day'], keep='first')

    sum_cols = ['loan_debit_sum', 'loan_credit_sum', 'deposit_debit_sum', 'deposit_credit_sum']
    for col in sum_cols:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in sum_cols:
        change_col_name = col.replace('_sum', '_hourly_change')
        df[change_col_name] = df.groupby(date_column)[col].diff().fillna(0.0)

    df['net_loan_flow_hourly_change'] = df['loan_debit_hourly_change'] - df['loan_credit_hourly_change']
    df['net_deposit_flow_hourly_change'] = df['deposit_credit_hourly_change'] - df['deposit_debit_hourly_change']
    df['total_net_flow_hourly_change'] = df['net_loan_flow_hourly_change'] + df['net_deposit_flow_hourly_change']
    
    stat_hourly_change_cols = [
        'net_loan_flow_hourly_change',
        'net_deposit_flow_hourly_change',
        'total_net_flow_hourly_change'
    ]

    unique_dates_in_hourly = df[date_column].unique()
    if len(unique_dates_in_hourly) == 0:
        warnings.warn("M2_Features: No unique dates in processed hourly data. Returning empty DataFrame.")
        return pd.DataFrame(index=pd.Index([], name=date_column))

    df_features = pd.DataFrame(index=pd.DatetimeIndex(unique_dates_in_hourly, name=date_column).sort_values())

    # --- Delta 9 AM to 3 PM Features ---
    pivoted_data_frames = {}
    # For sums at specific hours (9 and 15)
    for hour_val in [9, 15]:
        df_hour_data = df[df['hour_of_day'] == hour_val].set_index(date_column)
        cols_to_pivot = {col: df_hour_data[col] if col in df_hour_data else pd.Series(dtype='float64') for col in sum_cols}
        pivoted_data_frames[hour_val] = pd.DataFrame(cols_to_pivot).reindex(df_features.index)

    # Calculate delta 9-15 for original sum columns
    for col_sum in sum_cols:
        delta_col_name = col_sum.replace('_sum', '_d9_15') # Shorter name
        val_9am = pivoted_data_frames.get(9, pd.DataFrame(index=df_features.index)).get(col_sum, pd.Series(dtype='float64')).fillna(0)
        val_15pm = pivoted_data_frames.get(15, pd.DataFrame(index=df_features.index)).get(col_sum, pd.Series(dtype='float64')).fillna(0)
        df_features[delta_col_name] = val_15pm - val_9am
    
    # Calculate delta 9-15 for net flows
    df_features['net_loan_flow_d9_15'] = df_features['loan_debit_d9_15'] - df_features['loan_credit_d9_15']
    df_features['net_deposit_flow_d9_15'] = df_features['deposit_credit_d9_15'] - df_features['deposit_debit_d9_15']
    df_features['total_net_flow_d9_15'] = df_features['net_loan_flow_d9_15'] + df_features['net_deposit_flow_d9_15']

    df_intra_day_changes = df[(df['hour_of_day'] >= 10) & (df['hour_of_day'] <= 15)].copy()

    stat_functions = ['mean', 'std', 'sum', 'min', 'max', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'count']
    stat_names = ['mean', 'std', 'sum', 'min', 'max', 'median', 'p25', 'p75', 'count']

    if not df_intra_day_changes.empty:
        for col_hc in stat_hourly_change_cols:
            base_col_name_stat = col_hc.replace("_hourly_change", "_hc") 
            if col_hc in df_intra_day_changes.columns:
                try:
                    grouped_stats = df_intra_day_changes.groupby(date_column)[col_hc].agg(stat_functions)
                    grouped_stats.columns = [f'{s_name}_{base_col_name_stat}' for s_name in stat_names]
                    df_features = df_features.join(grouped_stats, how='left')

                except Exception as e_stat:
                    warnings.warn(f"M2_Features: Could not compute stats for {col_hc}. Error: {e_stat}")
                    for s_name in stat_names: df_features[f'{s_name}_{base_col_name_stat}'] = np.nan
            else:
                for s_name in stat_names: df_features[f'{s_name}_{base_col_name_stat}'] = np.nan
    else:
         for col_hc in stat_hourly_change_cols:
            base_col_name_stat = col_hc.replace("_hourly_change", "_hc")
            for s_name in stat_names: df_features[f'{s_name}_{base_col_name_stat}'] = np.nan

    if not df.empty: # Use the 9-15 df
        if 'total_net_flow_hourly_change' in df.columns:
            idx_max_flow = df.loc[df.groupby(date_column)['total_net_flow_hourly_change'].idxmax()][['hour_of_day']]
            idx_min_flow = df.loc[df.groupby(date_column)['total_net_flow_hourly_change'].idxmin()][['hour_of_day']]
            df_features = df_features.join(idx_max_flow.rename(columns={'hour_of_day': 'hour_max_total_net_flow_hc'}), how='left')
            df_features = df_features.join(idx_min_flow.rename(columns={'hour_of_day': 'hour_min_total_net_flow_hc'}), how='left')
        if 'total_net_flow_hourly_change' in df.columns:
            df_features['num_hrs_pos_total_net_flow_hc'] = df[df['total_net_flow_hourly_change'] > 1e-6].groupby(date_column).size()
            df_features['num_hrs_neg_total_net_flow_hc'] = df[df['total_net_flow_hourly_change'] < -1e-6].groupby(date_column).size()

    if isinstance(df_features.index, pd.DatetimeIndex):
        df_features['day_of_week_m2'] = df_features.index.dayofweek.astype(np.uint8)
        df_features['month_m2'] = df_features.index.month.astype(np.uint8)
        df_features['is_month_end_prox_m2'] = (df_features.index.to_series().groupby(df_features.index.to_period('M')).transform('max') - df_features.index.day <= 3).astype(np.uint8)
    else: 
        warnings.warn("M2_Features: df_features.index is not DatetimeIndex before adding calendar features.")
        df_features['day_of_week_m2'] = np.nan
        df_features['month_m2'] = np.nan
        df_features['is_month_end_prox_m2'] = np.nan

    df_features = df_features.fillna({
        col: 0 for col in df_features.columns if 'num_hrs_' in col or '_count_' in col
    })
    df_features = df_features.fillna({
        col: 12 for col in df_features.columns if 'hour_max_' in col or 'hour_min_' in col
    })
    df_features = df_features.fillna(0.0)

    for col in df_features.columns:
        if not pd.api.types.is_numeric_dtype(df_features[col]):
            warnings.warn(f"M2_Features: Column '{col}' is not numeric. Attempting conversion, then fillna(0).")
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0.0)

    print(f"M2_Features: Intraday features created. Shape: {df_features.shape}, Columns: {df_features.columns.tolist()[:5]}...")
    return df_features.sort_index()
