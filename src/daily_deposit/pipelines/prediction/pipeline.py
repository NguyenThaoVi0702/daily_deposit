# src/daily_loan/pipelines/prediction/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
predict_recursive, get_today_prediction, 
predict_today_corrected, format_predictions_for_datalake, 
predict_recursive_m1_only, create_yesterday_history_record, 
backfill_historical_predictions, append_backfill_to_history,
aggregate_partitions_into_dataframe, combine_and_deduplicate_history,
scatter_df_to_partitions, update_history_file, update_prediction_history_log
)
from pathlib import Path
import pandas as pd

def create_prediction_9am_pipeline(**kwargs) -> Pipeline:
    """Pipeline for the 9am run: retrain and predict for the year."""
    return pipeline(
        [
            node(
                predict_recursive,
                inputs=dict(
                    historical_daily_df="preprocessed_loan_data",
                    params_m1_features="params:model_1_params",
                    params_prediction="params:prediction",
                    params_reporting="params:reporting",
                    model_1_prod_q0_1="model_1_prod_q0.1",
                    model_1_prod_q0_5="model_1_prod_q0.5",
                    model_1_prod_q0_9="model_1_prod_q0.9",
                    model_1_prod_mean="model_1_prod_mean",
                    model_2_prod_median_corrector="model_2_prod_median_corrector",
                    model_2_prod_mean_corrector="model_2_prod_mean_corrector",
                ),
                outputs=["prediction_yearly_trend_mean_based", "prediction_yearly_trend_median_based"],
                name="predict_recursive_node"
            ),
            node(
                get_today_prediction,
                inputs="prediction_yearly_trend_mean_based", 
                outputs="prediction_today",
                name="get_today_prediction_node"
            )
        ]
    )

def create_prediction_3pm_pipeline(**kwargs) -> Pipeline:
    """Pipeline for the 3:30pm run: load models and correct today's prediction."""
    return pipeline(
        [
            node(
                predict_today_corrected,
                inputs=dict(
                    prediction_today_m1="prediction_today",
                    hourly_data_today="raw_hourly_data_today",
                    model_m2_median="model_2_prod_median_corrector",
                    model_m2_mean="model_2_prod_mean_corrector",
                    params_reporting="params:reporting" 
                ),
                outputs="prediction_corrected_today",
                name="predict_corrected_node"
            )
        ]
    )
    

def create_daily_prediction_pipeline(**kwargs) -> Pipeline:
    """
    A pipeline that loads production models, runs the recursive forecast,
    and formats the data for the datalake.
    """
    return pipeline([
        node(
            predict_recursive,
            inputs=dict(
                historical_daily_df="preprocessed_loan_data",
                params_m1_features="params:model_1_params",
                params_prediction="params:prediction",
                model_1_prod_q01="model_1_prod_q0.1",
                model_1_prod_q05="model_1_prod_q0.5",
                model_1_prod_q09="model_1_prod_q0.9",
                model_1_prod_mean="model_1_prod_mean",
                model_2_prod_median_corrector="model_2_prod_median_corrector",
                model_2_prod_mean_corrector="model_2_prod_mean_corrector",
            ),
            outputs=["prediction_yearly_trend_mean_based", "prediction_yearly_trend_median_based"],
            name="predict_recursive_node"
        ),
        node(
            format_predictions_for_datalake,
            inputs={
                "prediction_run_date": "params:prediction_run_date", 
                "mean_based_forecast": "prediction_yearly_trend_mean_based",
                "median_based_forecast": "prediction_yearly_trend_median_based"
            },
            outputs="predictions_for_datalake",
            name="format_predictions_for_db"
        )
    ])


def create_history_update_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=update_prediction_history_log,
                inputs={
                    "historical_actuals": "preprocessed_loan_data",
                    "last_days_mean_forecast": "last_days_mean_forecast",
                    "last_days_median_forecast": "last_days_median_forecast"
                },
                outputs=["prediction_history_for_datalake", "history_update_complete_trigger"],
                name="update_history_log_node"
            )
        ]
    )


def create_m1_only_prediction_pipeline(**kwargs) -> Pipeline:
    """
    A pipeline that loads M1 production models, runs a recursive M1-only
    forecast, and formats the data for the datalake.
    """
    
    """
    preserve_branch = pipeline (
        [
            node (
                func=lambda x: x,
                inputs="last_days_mean_forecast",
                outputs="preserved_last_days_mean_forecast",
                name="preserved_last_days_mean_forecast_node"
            ),
            node (
                func=lambda x: x,
                inputs="last_days_median_forecast",
                outputs="preserved_last_days_median_forecast",
                name="preserve_last_days_mean_median_node"
            )
        ]
    )
    """
    
    forecast_branch = pipeline (
        [
            node(
                func=predict_recursive_m1_only,
                inputs=dict(
                    history_update_signal="history_update_complete_trigger",
                    historical_daily_df="preprocessed_loan_data",
                    params_m1_features="params:model_1_params",
                    params_prediction="params:prediction",
                    model_1_prod_q01="model_1_prod_q0.1",
                    model_1_prod_q05="model_1_prod_q0.5",
                    model_1_prod_q09="model_1_prod_q0.9",
                    model_1_prod_mean="model_1_prod_mean",
                ),
                outputs=["prediction_yearly_trend_mean_based", "prediction_yearly_trend_median_based", "prediction_run_date_from_data"],
                name="predict_recursive_m1_only_node"
            ),
            node(
                func=format_predictions_for_datalake,
                inputs={
                    "prediction_run_date": "prediction_run_date_from_data",
                    "mean_based_forecast": "prediction_yearly_trend_mean_based",
                    "median_based_forecast": "prediction_yearly_trend_median_based"
                },
                outputs="predictions_for_datalake",
                name="format_m1_predictions_for_db"
            )
        ]
    )
    
    """
    history_nodes = []
    
    try:
        history_actuals_df = pd.read_csv("data/02_intermediate/daily_diff.csv", parse_dates=["cob_dt"])
        yesterday_date = history_actuals_df.sort_values(by="cob_dt").iloc[-1]["cob_dt"].date()
        partition_path_to_check = Path(f"data/11_intermediate/prediction_history/{yesterday_date.strftime('%Y-%m-%d')}.csv")
        
        if not partition_path_to_check.exists():
            print(f"History for {yesterday_date} does not exist, adding creation node")
            
            history_nodes.append(
                node(
                    func=create_yesterday_history_record,
                    inputs={
                        "historical_actuals": "preprocessed_loan_data",
                        "last_days_mean_forecast": "preserved_last_days_mean_forecast",
                        "last_days_median_forecast": "preserved_last_days_median_forecast"
                    },
                    outputs="prediction_history_partitions",
                    name="create_history_record_partition_node"
                
                )
            )
            
        else:
            print(f"Info: History for {yesterday_date} already exists. Skipping creation node.")
    except (FileNotFoundError, IndexError):
        print(f"Warning: Could not find historical actuals. No history creation needed.")
            
    aggregation_node = node(
        func=aggregate_partitions_into_dataframe,
        inputs="prediction_history_partitions",
        outputs="predictions_history_for_datalake",
        name="aggregate_history_for_datalake_node"
    )
    
    history_branch = pipeline(history_nodes + [aggregation_node])
    """
    
    return forecast_branch 

    

def create_backfill_pipeline(**kwargs) -> Pipeline:
    """
    A pipeline to generate and append historical predictions.
    """
    return pipeline([
        node(
            func=backfill_historical_predictions,
            inputs=dict(
                full_historical_data="preprocessed_loan_data",
                params_m1_features="params:model_1_params",
                params_prediction="params:prediction",
                days_to_backfill="params:backfill.days_to_backfill",
                model_1_prod_q01="model_1_prod_q0.1",
                model_1_prod_q05="model_1_prod_q0.5",
                model_1_prod_q09="model_1_prod_q0.9",
                model_1_prod_mean="model_1_prod_mean",
            ),
            outputs="prediction_history_partitions", 
            name="backfill_predictions_partitions_node"
        ),
        node(
            func=aggregate_partitions_into_dataframe,
            inputs="prediction_history_partitions",
            outputs="prediction_history_for_datalake",
            name="aggregate_backfill_for_datalake_node"
        )
    ])


