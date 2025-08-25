from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_rolling_weekly_simulation, format_simulation_for_history_table

def create_pipeline(**kwargs) -> Pipeline:
    """Creates the rolling backfill simulation pipeline."""
    return pipeline([
        node(
            func=run_rolling_weekly_simulation,
            inputs=dict(
                m1_full_features="model_1_feature_data",
                daily_full_preprocessed="preprocessed_loan_data",
                params_simulation="params:backfill_simulation",
                params_m1_training="params:model_1_training",
                params_m1_features="params:model_1_params",
                params_prediction="params:prediction",
            ),
            outputs="raw_simulation_predictions", 
            name="run_rolling_simulation_node"
        ),
        node(
            func=format_simulation_for_history_table,
            inputs=["raw_simulation_predictions", "preprocessed_loan_data"],
            outputs="prediction_history_for_datalake", 
            name="format_simulation_results_node"
        )
    ])
