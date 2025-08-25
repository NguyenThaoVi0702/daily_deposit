import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from typing import Dict, Any, List, Callable
import os

from daily_deposit.pipelines.data_engineering.nodes import create_features_for_model1, create_intraday_features_for_model2

log = logging.getLogger(__name__)

def predict_recursive(
    historical_daily_df: pd.DataFrame,
    params_m1_features: Dict,
    params_prediction: Dict,
    #params_reporting: Dict,  
    **models: lgb.LGBMRegressor
) -> (pd.DataFrame, pd.DataFrame):
    """
    Predicts N days into the future, tracking both a mean and median-based scenario.
    """
    n_days = params_prediction["yearly_forecast_days"]
    m1_feature_lookback = 400
    
    models_m1 = {k: v for k, v in models.items() if 'model_1_prod' in k}
    models_m2 = {k: v for k, v in models.items() if 'model_2_prod' in k}

    correction_factors = params_reporting["m2_correction_factors", {'median': 0.8, 'mean':0.9}]
    
    forecast_mean_based = _run_one_recursion(
        n_days, historical_daily_df, params_m1_features, m1_feature_lookback, models_m1, models_m2,
        params_prediction["synthetic_actual_choice_mean"], correction_factors
    )
    forecast_median_based = _run_one_recursion(
        n_days, historical_daily_df, params_m1_features, m1_feature_lookback, models_m1, models_m2,
        params_prediction["synthetic_actual_choice_median"], correction_factors
    )
    return forecast_mean_based, forecast_median_based

def _run_one_recursion(n_days, historical_daily_df, params_m1, lookback, m1, m2, synthetic_choice, factors):
    """Helper function to run a single recursive simulation."""
    data_for_recursion = historical_daily_df.copy()
    data_for_recursion['cob_dt'] = pd.to_datetime(data_for_recursion['cob_dt'])
    last_known_date = data_for_recursion['cob_dt'].max()
    future_predictions = []
    
    log.info(f"--- Starting recursive loop for {n_days} days, based on '{synthetic_choice}' ---")

    for i in range(1, n_days + 1):
        prediction_date = last_known_date + pd.Timedelta(days=i)

        # Create feature creation slice
        start_index = max(0, len(data_for_recursion) - lookback)
        recent_history_slice = data_for_recursion.iloc[start_index:].copy()
        placeholder_row = pd.DataFrame([{'cob_dt': prediction_date}])
        data_for_feature_creation = pd.concat([recent_history_slice, placeholder_row], ignore_index=True)
        
        all_features = create_features_for_model1(data_for_feature_creation, params_m1)
        if all_features is None or prediction_date not in all_features.index:
            log.error(f"Feature creation failed for {prediction_date.date()}. Stopping recursion.")
            break
        X_m1_today = all_features.loc[[prediction_date]]
        
        # Predict with M1
        m1_preds = {}
        for k, v in m1.items():
            target_name = k.replace('model_1_prod_', '').replace('_', '.')
            pred_key = f"m1_pred_{target_name}"
            m1_preds[pred_key] = v.predict(X_m1_today[v.feature_name_])[0]
        
        # Predict M2 Error 
        m2_feature_row = {'m1_pred_interval_width': m1_preds['m1_pred_q0.9'] - m1_preds['m1_pred_q0.1']}
        m2_features_df = pd.DataFrame([m2_feature_row], index=[prediction_date])
        m2_error_preds = {}
        for m2_type, model in m2.items():
            corrector_type = m2_type.split('_')[-2] # mean or median
            X_m2_reindexed = m2_features_df.reindex(columns=model.feature_name_).fillna(0)
            m2_error_preds[f'm2_{corrector_type}_error_pred'] = model.predict(X_m2_reindexed)[0]

        # Combine and Finalize
        final_preds = {**m1_preds, **m2_error_preds, 'cob_dt': prediction_date}
        adj_median = final_preds['m2_median_error_pred'] * factors['median']
        for q in [0.1, 0.5, 0.9]:
            final_preds[f'final_pred_q{q}'] = final_preds[f'm1_pred_q{q}'] + adj_median
        adj_mean = final_preds['m2_mean_error_pred'] * factors['mean']
        final_preds['final_pred_mean'] = final_preds['m1_pred_mean'] + adj_mean
        future_predictions.append(final_preds)

        # 6. The Recursive Step
        synthetic_diff = final_preds[synthetic_choice]
        last_balance = data_for_recursion.iloc[-1][params_m1["balance_col"]]
        new_row = pd.DataFrame([{
            'cob_dt': prediction_date,
            params_m1["target_col"]: synthetic_diff,
            params_m1["balance_col"]: last_balance + synthetic_diff
        }])
        data_for_recursion = pd.concat([data_for_recursion, new_row], ignore_index=True)

    return pd.DataFrame(future_predictions)

def get_today_prediction(yearly_forecast: pd.DataFrame) -> Dict:
    """Extracts the first day's prediction from the yearly forecast."""
    if yearly_forecast.empty:
        return {"error": "Yearly forecast was empty."}
    return yearly_forecast.iloc[0].to_dict()

def predict_today_corrected(
    prediction_today_m1: Dict,
    hourly_data_today: pd.DataFrame,
    model_m2_median: lgb.LGBMRegressor,
    model_m2_mean: lgb.LGBMRegressor,
    params_reporting: Dict,
) -> Dict:
    """Applies the final M2 correction using today's live hourly data."""
    log.info("--- Applying 3:30 PM correction with live intraday data ---")
    
    # Create M2 features from today's hourly data
    df_m2_features_today = create_intraday_features_for_model2(hourly_data_today, {"date_column": "cob_dt"})
    
    # Add the M1 interval width feature
    df_m2_features_today['m1_pred_interval_width'] = prediction_today_m1['m1_pred_q0.9'] - prediction_today_m1['m1_pred_q0.1']
    
    # Predict the M2 error
    X_m2_final = df_m2_features_today.reindex(columns=model_m2_mean.feature_name_).fillna(0)
    m2_error_mean = model_m2_mean.predict(X_m2_final)[0]
    m2_error_median = model_m2_median.predict(X_m2_final)[0]
    
    # Apply correction
    factors = params_reporting["m2_correction_factors"]
    final_preds = prediction_today_m1.copy()
    
    adj_median = m2_error_median * factors['median']
    for q in [0.1, 0.5, 0.9]:
        final_preds[f'final_pred_q{q}'] = final_preds[f'm1_pred_q{q}'] + adj_median
        
    adj_mean = m2_error_mean * factors['mean']
    final_preds['final_pred_mean'] = final_preds['m1_pred_mean'] + adj_mean
    
    final_preds['correction_applied_at'] = pd.Timestamp.now().isoformat()
    return final_preds
    
    
def format_predictions_for_datalake(
    prediction_run_date: pd.Timestamp,
    mean_based_forecast: pd.DataFrame,
    median_based_forecast: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms the wide prediction DataFrames into a long format suitable for database storage.
    """
    log.info("--- Formatting prediction results for datalake ---")
    
    # Process mean-based forecast
    df_mean = mean_based_forecast.copy()
    df_mean = df_mean.melt(
        id_vars=['cob_dt'], 
        value_vars=[col for col in df_mean.columns if 'final_pred' in col],
        var_name='metric_type',
        value_name='metric_value'
    )
    df_mean['scenario'] = 'mean_based'

    # Process median-based forecast
    df_median = median_based_forecast.copy()
    df_median = df_median.melt(
        id_vars=['cob_dt'], 
        value_vars=[col for col in df_median.columns if 'final_pred' in col],
        var_name='metric_type',
        value_name='metric_value'
    )
    df_median['scenario'] = 'median_based'
    
    # Combine and format
    final_df = pd.concat([df_mean, df_median], ignore_index=True)
    final_df['prediction_run_date'] = pd.to_datetime(prediction_run_date)
    final_df = final_df.rename(columns={'cob_dt': 'target_date'})
    
    final_df = final_df.rename(columns={'prediction_run_date': 'cob_dt'})
    
    # Reorder for clarity
    final_df = final_df[['cob_dt', 'target_date', 'scenario', 'metric_type', 'metric_value']]
    return final_df


def predict_recursive_m1_only(
    history_update_signal: bool,
    historical_daily_df: pd.DataFrame,
    params_m1_features: Dict,
    params_prediction: Dict,
    **models_m1: lgb.LGBMRegressor
) -> (pd.DataFrame, pd.DataFrame, pd.Timestamp):
    """
    M1-ONLY VERSION. Predicts N days into the future using only M1.
    """
    n_days = params_prediction["yearly_forecast_days"]
    m1_feature_lookback = 400
    historical_daily_df['cob_dt'] = pd.to_datetime(historical_daily_df['cob_dt'])
    last_known_date = historical_daily_df['cob_dt'].max()
    log.info(f"Recursive prediction based on last known date: {last_known_date.date()}")
    
    # Run two simulations based on M1's mean and median predictions
    forecast_mean_based = _run_one_recursion_m1_only(
        n_days, historical_daily_df, params_m1_features, m1_feature_lookback, models_m1,
        params_prediction["synthetic_actual_choice_mean"]
    )
    forecast_median_based = _run_one_recursion_m1_only(
        n_days, historical_daily_df, params_m1_features, m1_feature_lookback, models_m1,
        params_prediction["synthetic_actual_choice_median"]
    )
    log.info(f"Recursive prediction based on last known date: {last_known_date.date()}")
    return forecast_mean_based, forecast_median_based, last_known_date


def _run_one_recursion_m1_only(n_days, historical_daily_df, params_m1, lookback, m1, synthetic_choice):
    """Helper function for a single M1-only recursive simulation."""
    data_for_recursion = historical_daily_df.copy()
    last_known_date = data_for_recursion['cob_dt'].max()
    future_predictions = []
    
    try:
        model_mean = next(v for k, v in m1.items() if 'mean' in  k)
        model_q01 = next(v for k, v in m1.items() if 'q0.1' in  k or 'q01' in k)
        model_q05 = next(v for k, v in m1.items() if 'q0.5' in  k or 'q05' in k)
        model_q09 = next(v for k, v in m1.items() if 'q0.9' in  k or 'q09' in k)
    except:
        log.error("A required M1 model (mean, q01, q05, q09) was not found in the input dictionary")
        raise ValueError("Could not find all required M1 models to run the prediction")

    log.info(f"--- Starting M1-ONLY recursive loop for {n_days} days, based on '{synthetic_choice}' ---")

    for i in range(1, n_days + 1):
        prediction_date = last_known_date + pd.Timedelta(days=i)
        
        data_for_feature_creation = pd.concat([
            data_for_recursion.iloc[max(0, len(data_for_recursion) - lookback):],
            pd.DataFrame([{'cob_dt': prediction_date}])
        ], ignore_index=True)
        all_features = create_features_for_model1(data_for_feature_creation, params_m1)
        if all_features is None or prediction_date not in all_features.index:
            log.error(f"M1 Feature creation failed for {prediction_date.date()}. Stopping recursion.")
            break
        X_m1_today = all_features.loc[[prediction_date]]
        

        m1_preds = {
            'm1_pred_mean': model_mean.predict(X_m1_today[model_mean.feature_name_])[0],
            'm1_pred_q0.1': model_q01.predict(X_m1_today[model_q01.feature_name_])[0],
            'm1_pred_q0.5': model_q05.predict(X_m1_today[model_q05.feature_name_])[0],
            'm1_pred_q0.9': model_q09.predict(X_m1_today[model_q09.feature_name_])[0]
        }

        final_preds = m1_preds.copy()
        final_preds['cob_dt'] = prediction_date

        for pred_key in m1_preds:
            final_pred_key = pred_key.replace("m1_pred", "final_pred")
            final_preds[final_pred_key] = m1_preds[pred_key]
        future_predictions.append(final_preds)

        # 5. The Recursive Step
        synthetic_diff = final_preds[synthetic_choice]
        last_balance = data_for_recursion.iloc[-1][params_m1["balance_col"]]
        new_row = pd.DataFrame([{
            'cob_dt': prediction_date,
            params_m1["target_col"]: synthetic_diff,
            params_m1["balance_col"]: last_balance + synthetic_diff
        }])
        data_for_recursion = pd.concat([data_for_recursion, new_row], ignore_index=True)

    return pd.DataFrame(future_predictions)



   
    
#old 
'''
def create_yesterday_history_record(
    historical_actuals: pd.DataFrame,
    last_days_mean_forecast: pd.DataFrame,
    last_days_median_forecast: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates a single-row DataFrame containing the actual value for yesterday
    and the prediction that was made for it on the previous day.
    """
    log.info("--- Creating historical performance record for yesterday ---")
    
    # 1. Get yesterday's actuals
    yesterday_actuals = historical_actuals.sort_values(by="cob_dt").iloc[-1]
    yesterday_date = pd.to_datetime(yesterday_actuals['cob_dt'])
    actual_value = yesterday_actuals['total_bal_diff']

    # 2. Get yesterday's prediction
    yesterday_pred_mean_row = last_days_mean_forecast.iloc[0]
    yesterday_pred_median_row = last_days_median_forecast.iloc[0]
    
    # 3. Ensure the dates align
    pred_date = pd.to_datetime(yesterday_pred_mean_row['cob_dt']).date()
    if pred_date != yesterday_date:
        log.warning(
            f"Date mismatch! Actual date is {yesterday_date} but prediction date is {pred_date}. "
            "This can happen on the very first run. Proceeding but record may be inaccurate."
        )
        return {}

    predicted_mean = yesterday_pred_mean_row['final_pred_mean']
    predicted_median = yesterday_pred_median_row['final_pred_q0.5']
    
    history_record = pd.DataFrame([{
        "target_date": yesterday_date,
        "actual_value": actual_value,
        "predicted_mean": predicted_mean,
        "predicted_median": predicted_median
    }])
    
    partition_id = yesterday_date.strftime("%Y-%m-%d")
    
    log.info(f"Successfully created history record for {yesterday_date}.")
    return {partition_id: history_record}
'''

# New
def create_yesterday_history_record(
    historical_actuals: pd.DataFrame,
    last_days_mean_forecast: pd.DataFrame,
    last_days_median_forecast: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates a history record for yesterday ONLY if it doesn't already exist.
    """
    log.info("--- Checking if yesterday's history record needs to be created ---")
    
    # 1. Get yesterday's date from the new actuals data
    yesterday_actuals = historical_actuals.sort_values(by="cob_dt").iloc[-1]
    yesterday_date = pd.to_datetime(yesterday_actuals['cob_dt'])
    partition_id = yesterday_date.strftime("%Y-%m-%d")
    
    # 3. Get yesterday's prediction
    yesterday_pred_mean_row = last_days_mean_forecast.iloc[0]
    pred_date = pd.to_datetime(yesterday_pred_mean_row['cob_dt'])

    if pred_date != yesterday_date:
        log.warning(
            f"Date mismatch! Actual date is {yesterday_date} but prediction date is {pred_date}. "
            "Cannot create a valid history record. This can happen on the very first run."
        )
        return {}

    # 4. Assemble the new record
    actual_value = yesterday_actuals['total_bal_diff']
    predicted_mean = yesterday_pred_mean_row['final_pred_mean']
    predicted_median = yesterday_pred_mean_row['final_pred_q0.5']
    
    history_record_df = pd.DataFrame([{
        "target_date": yesterday_date,
        "actual_value": actual_value,
        "predicted_mean": predicted_mean,
        "predicted_median": predicted_median,
        "prediction_run_date": yesterday_date - pd.Timedelta(days=1)
    }])
    
    log.info(f"Successfully created new history record for partition: {partition_id}.")
    return {partition_id: history_record_df}


def backfill_historical_predictions(
    full_historical_data: pd.DataFrame,
    params_m1_features: Dict,
    params_prediction: Dict,
    days_to_backfill: int,
    **models: lgb.LGBMRegressor
) -> Dict[str, pd.DataFrame]:
    """
    Simulates daily runs for past days, creating records ONLY for dates
    that are not already in the existing history.
    """
    log.info(f"--- Starting historical prediction backfill for the last {days_to_backfill} days ---")
    
    full_historical_data['cob_dt'] = pd.to_datetime(full_historical_data['cob_dt'])
    full_historical_data = full_historical_data.sort_values(by='cob_dt').reset_index(drop=True)

    latest_actual_date = full_historical_data['cob_dt'].max()
    partitions_to_save = {}
    m1_models = {k: v for k, v in models.items() if 'model_1_prod' in k}


    for i in range(days_to_backfill, 0, -1):
        target_date = latest_actual_date - pd.Timedelta(days=i-1)
        partition_id = target_date.strftime("%Y-%m-%d")
            
        log.info(f"Simulating run for missing target_date: {target_date}")
        
        # It masks data, creates features, and predicts for the target_date
        prediction_run_date = target_date - pd.Timedelta(days=1)
        data_for_this_run = full_historical_data[full_historical_data['cob_dt'] < target_date].copy()
        if data_for_this_run.empty: continue
        m1_feature_lookback = 400
        start_index = max(0, len(data_for_this_run) - m1_feature_lookback)
        data_for_feature_creation = pd.concat([
            data_for_this_run.iloc[start_index:],
            pd.DataFrame([{'cob_dt': target_date}])
        ], ignore_index=True)
        all_features = create_features_for_model1(data_for_feature_creation, params_m1_features)
        if all_features is None or target_date not in all_features.index:
            log.error(f"Feature creation failed for backfill date {target_date}. Skipping.")
            continue
        X_m1_today = all_features.loc[[target_date]]


        # 3. PREDICT 
        m1_preds = { 
            'm1_pred_q0.1': m1_models['model_1_prod_q01'].predict(X_m1_today[m1_models['model_1_prod_q01'].feature_name_])[0],
            'm1_pred_q0.5': m1_models['model_1_prod_q05'].predict(X_m1_today[m1_models['model_1_prod_q05'].feature_name_])[0],
            'm1_pred_q0.9': m1_models['model_1_prod_q09'].predict(X_m1_today[m1_models['model_1_prod_q09'].feature_name_])[0],
            'm1_pred_mean': m1_models['model_1_prod_mean'].predict(X_m1_today[m1_models['model_1_prod_mean'].feature_name_])[0]
        }

        # 4. GET THE ACTUAL VALUE for that day
        actual_value_row = full_historical_data[full_historical_data['cob_dt'] == target_date]
        if actual_value_row.empty:
            log.warning(f"Could not find actual value for {target_date}. Skipping.")
            continue
        actual_value = actual_value_row['total_bal_diff'].iloc[0]


        record_df = pd.DataFrame([{
            "target_date": target_date.date(),
            "actual_value": actual_value,
            "predicted_mean": m1_preds['m1_pred_mean'],
            "predicted_median": m1_preds['m1_pred_q0.5'],
            "prediction_run_date": prediction_run_date.date()
        }])
        
        partitions_to_save[partition_id] = record_df
        
    log.info(f"Generated {len(partitions_to_save)} new history partitions to be added.")
    return partitions_to_save

    
def update_history_file(
    new_record_df: pd.DataFrame,
    history_filepath: str
) -> pd.DataFrame:
    """
    - Read the existing CSV
    - Appends the new record
    - De-duplicates, keeping the original record in case of a re-run
    - Overwrites the file with the clean, combined data
    """
    if new_record_df.empty or 'target_date' not in new_record_df.columns:
        log.warning("No valid new history record to add")
        if os.path.exists(history_filepath):
            return pd.read_csv(history_filepath)
        return pd.DataFrame()
        
    if os.path.exists(history_filepath):
        log.info(f"Loading existing history from {history_filepath}")
        existing_history_df = pd.read_csv(history_filepath)
    else:
        log.info("History file not found. Creating a new one.")
        existing_history_df = pd.DataFrame()
        
    if not existing_history_df.empty:
        existing_history_df["target_date"] = pd.to_datetime(existing_history_df["target_date"])
        
    new_record_df["target_date"] = pd.to_datetime(new_record_df["target_date"])
    
    combined = pd.concat([existing_history_df, new_record_df], ignore_index=True)

    # Remove any duplicates, keeping the most recently generated record for a given date
    combined = combined.drop_duplicates(subset=['target_date'], keep='last')
    
    # Sort by date for a clean final file
    combined = combined.sort_values(by='target_date', ascending=True)
        
    log.info(f"Saving updated history with {len(combined)} records to {history_filepath}.")
    combined.to_csv(history_filepath, index=False)
        
    return combined


# --- Back Fill History ---
#Old
'''
def backfill_historical_predictions(
    full_historical_data: pd.DataFrame,
    params_m1_features: Dict,
    params_prediction: Dict,
    days_to_backfill: int,
    **models: lgb.LGBMRegressor
) -> Dict[str, pd.DataFrame]:
    """
    For each day, itfuture data and generates a prediction.
    """
    log.info(f"--- Starting historical prediction backfill for the last {days_to_backfill} days ---")
    
    full_historical_data['cob_dt'] = pd.to_datetime(full_historical_data['cob_dt'])
    full_historical_data = full_historical_data.sort_values(by='cob_dt').reset_index(drop=True)

    latest_actual_date = full_historical_data['cob_dt'].max()
    new_history_records = []
    partition_to_save = {}
    m1_models = {k: v for k, v in models.items() if 'model_1_prod' in k}
    
    
    for i in range(days_to_backfill, 0, -1):
        target_date = latest_actual_date - pd.Timedelta(days=i-1)
        prediction_run_date = target_date - pd.Timedelta(days=1)
        
        log.info(f"Simulating run for target_date: {target_date.date()}")

        # 1. MASK THE FUTURE
        data_for_this_run = full_historical_data[full_historical_data['cob_dt'] < target_date].copy()
        if data_for_this_run.empty:
            log.warning(f"Not enough historical data to predict for {target_date.date()}. Skipping.")
            continue
            
        # 2. GENERATE FEATURES 
        m1_feature_lookback = 400
        start_index = max(0, len(data_for_this_run) - m1_feature_lookback)
        data_for_feature_creation = pd.concat([
            data_for_this_run.iloc[start_index:],
            pd.DataFrame([{'cob_dt': target_date}])
        ], ignore_index=True)
        
        all_features = create_features_for_model1(data_for_feature_creation, params_m1_features)
        if all_features is None or target_date not in all_features.index:
            log.error(f"Feature creation failed for backfill date {target_date.date()}. Skipping.")
            continue
        X_m1_today = all_features.loc[[target_date]]

        # 3. PREDICT 
        m1_preds = { 
            'm1_pred_q0.1': m1_models['model_1_prod_q01'].predict(X_m1_today[m1_models['model_1_prod_q01'].feature_name_])[0],
            'm1_pred_q0.5': m1_models['model_1_prod_q05'].predict(X_m1_today[m1_models['model_1_prod_q05'].feature_name_])[0],
            'm1_pred_q0.9': m1_models['model_1_prod_q09'].predict(X_m1_today[m1_models['model_1_prod_q09'].feature_name_])[0],
            'm1_pred_mean': m1_models['model_1_prod_mean'].predict(X_m1_today[m1_models['model_1_prod_mean'].feature_name_])[0]
        }

        # 4. GET THE ACTUAL VALUE for that day
        actual_value_row = full_historical_data[full_historical_data['cob_dt'] == target_date]
        if actual_value_row.empty:
            log.warning(f"Could not find actual value for {target_date.date()}. Skipping.")
            continue
        actual_value = actual_value_row['total_bal_diff'].iloc[0]

        # 5. ASSEMBLE THE RECORD
        
#        new_history_records.append({
#            "target_date": target_date.date(),
#            "actual_value": actual_value,
#            "predicted_mean": m1_preds['m1_pred_mean'],
#            "predicted_median": m1_preds['m1_pred_q0.5'],
#           "prediction_run_date": prediction_run_date.date()
#        })
        

        record_df = pd.DataFrame([{
            "target_date": target_date.date(),
            "actual_value": actual_value,
            "predicted_mean": m1_preds['m1_pred_mean'],
            "predicted_median": m1_preds['m1_pred_q0.5'],
            "prediction_run_date": prediction_run_date.date()
        }])
        
        partition_id = target_date.strftime("%Y-%m-%d")
        partition_to_save[partition_id] = record_df

        
    log.info(f"Successfully generated {len(record_df)} new history records.")
    return partition_to_save
'''

def append_backfill_to_history(
    backfilled_history: pd.DataFrame,
    existing_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines newly backfilled history with existing history,
    removing duplicates and sorting.
    """
    
    if existing_history.empty:
        log.info("No existing history found. Creating new history file.")
        return backfilled_history

    log.info(f"Appending {len(backfilled_history)} new records to existing {len(existing_history)} history records.")
    
    existing_history['target_date'] = pd.to_datetime(existing_history['target_date'])
    backfilled_history['target_date'] = pd.to_datetime(backfilled_history['target_date'])
    
    combined = pd.concat([existing_history, backfilled_history], ignore_index=True)
    
    # Remove any duplicates, keeping the most recently generated record for a given date
    combined = combined.drop_duplicates(subset=['target_date'], keep='last')
    
    # Sort by date for a clean final file
    combined = combined.sort_values(by='target_date', ascending=True)
    
    log.info(f"Final history table contains {len(combined)} records.")
    return combined


def aggregate_partitions_into_dataframe(
    partitions: Dict[str, Callable[[], Any]]
) -> pd.DataFrame:
    """
    Take in a dictionary of partitions and concatenate them into a single pandas DataFrame.
    """
    
    log.info(f"Aggregating {len(partitions)} partitions into a single DataFrame.")
    if not partitions:
        return pd.DataFrame()
        
    all_dfs = [loader() for loader in partitions.values()]
    
    combine_df = pd.concat(all_dfs, ignore_index=True)
    combine_df['target_date'] = pd.to_datetime(combine_df['target_date'])
    combine_df = combine_df.drop_duplicates(subset=['target_date'], keep='first')
    combine_df = combine_df.sort_values(by="target_date", ascending=True)
    
    log.info(f"Successfully aggregated partition. Final shape: {combine_df.shape}.")
    
    return combine_df
    
    
def combine_and_deduplicate_history(
    new_history_record_df: pd.DataFrame,
    existing_history_df: pd.DataFrame
) -> pd.DataFrame:
    """
        Only add records for dates that do not already exist.
    """
    if new_history_record_df.empty or 'target_date' not in new_history_record_df.columns:
        log.warning("No new valid history record provide.")
        return existing_history_df
        
    if existing_history_df.empty:
        log.info("No existing history found. Using new records only.")
        return new_history_record_df.sort_values(by='target_date')
        
    existing_history_df['target_date'] = pd.to_datetime(existing_history_df['target_date'])
    new_history_record_df['target_date'] = pd.to_datetime(new_history_record_df['target_date'])
    combine_df = pd.concat([existing_history_df, new_history_record_df], ignore_index=True)
    combine_df = combine_df.drop_duplicates(subset=['target_date'], keep='first')
    combine_df = combine_df.sort_values(by='target_date')
    
    log.info(f"Combined history ready. Total records: {len(combine_df)}. New records added: {len(combine_df) - len(existing_history_df)}.")
    
    return combine_df
    
       
def scatter_df_to_partitions(
    final_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:

    if final_df.empty:
        return {}
        
    final_df['target_date'] = pd.to_datetime(final_df['target_date'])
    
    final_partitions = {
        row['target_date'].strftime('%Y-%m-%d'): pd.DataFrame([row])
        for index, row in final_df.iterrows()
    }
    
    return final_partitions
    
def generate_missing_history_predictions(
    full_historical_data: pd.DataFrame,
    existing_history_data: pd.DataFrame,
    days_to_check: int,
    params_m1_features: Dict,
    **models: lgb.LGBMRegressor
) -> pd.DataFrame:
    
    log.info(f"Checking for missing history in the last {days_to_check}.")
    
    full_historical_data['cob_dt'] = pd.to_datetime(full_historical_data['cob_dt'])
    full_historical_data = full_historical_data.sort_values(by='cob_dt').reset_index(drop=True)
    
    if not existing_history_data.empty:
        existing_history_data['target_date'] = pd.to_datetime(existing_history_data['target_date'])
        dates_already_predicted = set(existing_history['target_date'])
    else:
        dates_already_predicted = set()
        
        
    latest_actual_date = full_historical_data['cob_dt'].max()
    new_history_records = []
    m1_models = {k: v for k,v in models.items() if 'model_1_prod' in k}
    
    
   
def safely_load_or_create_empty_df(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        log.info(f"Found existing forecast file, loading: {filepath}")
        return pd.read_csv(filepath)
        
    else:
        log.warning(f"Forecast file not found: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["m1_pred_q01","m1_pred_q05", "m1_pred_q09", "m1_pred_mean", "cob_dt", "final_pred_mean", "final_pred_q0.1", "final_pred_q0.5",	"final_pred_q0.9"])
 
 
 
 

def update_prediction_history_log(
    historical_actuals: pd.DataFrame,
    last_days_mean_forecast: pd.DataFrame,
    last_days_median_forecast: pd.DataFrame
) -> (pd.DataFrame, bool):
    """
    1. Read the existing history log
    2. Identifies yesterday's actual and the prediction made for it
    3. Check if record for yesterday already exist
    4. If not, it appends the new record and return the full history
    5. If it exists, it returns the original history unchanged
    """
    
    log.info("---Starting the history log update---")
    
    try: 
        existing_history_df = pd.read_csv("data/10_datalake_output/prediction_history.csv")
        existing_history_df['target_date'] = pd.to_datetime(existing_history_df['target_date'])
        log.info(f"Loaded {len(existing_history_df)} existing history records.")
    except FileNotFoundError:
        log.info("No existing history file found!")
        existing_history_df = pd.DataFrame(columns = [
            "target_date", "actual_value", "predicted_mean", "predicted_median", "prediction_run_date"
        ])
    
    yesterday_actuals = historical_actuals.sort_values(by="cob_dt").iloc[-1]
    yesterday_date = pd.to_datetime(yesterday_actuals['cob_dt'])
    
    #If already have a record, do nothing
    if yesterday_date in existing_history_df['target_date'].values:
        log.info(f"History record for {yesterday_date} already exists. No update needed.")
        return existing_history_df, True
        
    try:
        yesterday_pred_mean_row = last_days_mean_forecast.iloc[0]
        yesterday_pred_median_row = last_days_median_forecast.iloc[0]
        pred_date = pd.to_datetime(yesterday_pred_mean_row['cob_dt'])
    except (IndexError, FileNotFoundError):
        log.warning("Could not read previous day's forecast, returning existing history.")
        return existing_history_df, True
        
    if pred_date != yesterday_date:
        log.warning(f"Date mismatch! Actual date is {yesterday_date} but prediction date is {pred_date}. Returning existing history.")
        return existing_history_df, True
        
    new_record = pd.DataFrame([{
        "target_date": yesterday_date,
        "actual_value": yesterday_actuals['total_bal_diff'],
        "predicted_mean": yesterday_pred_mean_row['final_pred_mean'],
        "predicted_median": yesterday_pred_mean_row['final_pred_q0.5'],
        "prediction_run_date": yesterday_date - pd.Timedelta(days=1)
    }])
    
    new_record['target_date'] = pd.to_datetime(new_record['target_date'])
    new_record['prediction_run_date'] = pd.to_datetime(new_record['prediction_run_date'])
    updated_history_df = pd.concat([existing_history_df, new_record], ignore_index=True)
    updated_history_df = updated_history_df.sort_values(by='target_date', ascending= True)
    
    log.info(f"Successfully append history for {yesterday_date}. Total records: {len(updated_history_df)}.")
    
    return updated_history_df, True
    
