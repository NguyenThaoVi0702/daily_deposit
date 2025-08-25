import pandas as pd
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


from daily_deposit.pipelines.data_science.nodes import (
    split_features_and_target,
    tune_hyperparameters,
    train_model
)
from daily_deposit.pipelines.prediction.nodes import (
    _run_one_recursion_m1_only 
)

def run_rolling_weekly_simulation(
    m1_full_features: pd.DataFrame,
    daily_full_preprocessed: pd.DataFrame,
    params_simulation: Dict,
    params_m1_training: Dict,
    params_m1_features: Dict,
    params_prediction: Dict
) -> pd.DataFrame:
    """
    Orchestrates a rolling weekly simulation of the training and prediction process.
    """
    daily_full_preprocessed['cob_dt'] = pd.to_datetime( daily_full_preprocessed['cob_dt']) 
    m1_full_features.index = pd.to_datetime(m1_full_features.index)

    start_date = pd.to_datetime(params_simulation["start_date"])
    window_days = params_simulation["prediction_window_days"]
    latest_available_date = m1_full_features.index.max()
    
    all_simulation_predictions = []
    current_train_end_date = start_date - pd.Timedelta(days=1)

    log.info(f"Starting rolling backfill from {start_date.date()} to {latest_available_date.date()}.")

    while current_train_end_date < latest_available_date:
        log.info("="*50)
        log.info(f"SIMULATION: Training on data up to {current_train_end_date.date()}")
        log.info("="*50)

        # 1. MASK THE FUTURE
        training_features_slice = m1_full_features[m1_full_features.index <= current_train_end_date]
        if len(training_features_slice) < 50: 
            log.warning(f"Skipping training for {current_train_end_date.date()} due to insufficient data ({len(training_features_slice)} rows).")
            current_train_end_date += pd.Timedelta(days=window_days)
            continue
        
        # 2. MIMIC THE TRAINING PIPELINE
        X_train, y_train, _ = split_features_and_target(training_features_slice, params_m1_training)
        
        trained_models_for_this_week = {}
        for pred_type in params_m1_training["prediction_targets"]:
            log.info(f"  - Tuning and training for target: {pred_type}")
            best_params = tune_hyperparameters(X_train, y_train, pred_type, params_m1_training)
            trained_model = train_model(X_train, y_train, best_params, pred_type)

            target_name = f"model_1_prod_{'q'+str(pred_type) if isinstance(pred_type, float) else pred_type}"
            trained_models_for_this_week[target_name] = trained_model

        # 3. MIMIC THE PREDICTION PIPELINE
        historical_data_slice =  daily_full_preprocessed[daily_full_preprocessed['cob_dt'] <= current_train_end_date]
        
        weekly_forecast_mean_based = _run_one_recursion_m1_only(
            n_days=window_days,
            historical_daily_df=historical_data_slice,
            params_m1=params_m1_features,
            lookback=400,
            m1=trained_models_for_this_week,
            synthetic_choice=params_prediction["synthetic_actual_choice_mean"]
        )
        weekly_forecast_median_based = _run_one_recursion_m1_only(
            n_days=window_days,
            historical_daily_df=historical_data_slice,
            params_m1=params_m1_features,
            lookback=400,
            m1=trained_models_for_this_week,
            synthetic_choice=params_prediction["synthetic_actual_choice_median"]
        )

        if not weekly_forecast_mean_based.empty and not weekly_forecast_median_based.empty:
            combined_week_df = weekly_forecast_mean_based[['cob_dt', 'final_pred_mean']].copy()
            combined_week_df = pd.merge(
                combined_week_df,
                weekly_forecast_median_based[['cob_dt', 'final_pred_q0.5']],
                on='cob_dt'
            )
            all_simulation_predictions.append(combined_week_df)
        
        # 4. Move to the next week
        current_train_end_date += pd.Timedelta(days=window_days)

    if not all_simulation_predictions:
        log.warning("Backfill simulation produced no predictions.")
        return pd.DataFrame()

    final_predictions_df = pd.concat(all_simulation_predictions, ignore_index=True)
    return final_predictions_df


def format_simulation_for_history_table(
    simulation_predictions: pd.DataFrame,
    full_preprocessed_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins simulation predictions with actuals and formats the result to match
    the prediction_history.csv schema.
    """
    if simulation_predictions.empty:
        return pd.DataFrame()

    preds = simulation_predictions.copy()
    preds = preds.rename(columns={
        "final_pred_mean": "predicted_mean",
        "final_pred_q0.5": "predicted_median",
        "cob_dt": "target_date"
    })
    preds['target_date'] = pd.to_datetime(preds['target_date'])
    
    actuals = full_preprocessed_data[['cob_dt', 'total_bal_diff']].copy()
    actuals = actuals.rename(columns={"cob_dt": "target_date", "total_bal_diff": "actual_value"})
    actuals['target_date'] = pd.to_datetime(actuals['target_date'])

    # Join to get the actuals
    history_df = pd.merge(preds, actuals, on='target_date', how='left')
    
    # Calculate the prediction_run_date
    history_df['prediction_run_date'] = history_df['target_date'] - pd.Timedelta(days=1)
    
    # Select and reorder columns to match the final table schema
    final_cols = ["target_date", "actual_value", "predicted_mean", "predicted_median", "prediction_run_date"]
    history_df = history_df[final_cols]
    
    history_df = history_df.dropna(subset=['actual_value'])
    history_df = history_df.sort_values(by='target_date').reset_index(drop=True)
    
    log.info(f"Formatted backfill history. Final shape: {history_df.shape}")
    return history_df
