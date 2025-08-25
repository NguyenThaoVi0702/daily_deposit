from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_raw_data, create_features_for_model1, create_intraday_features_for_model2

def create_pipeline(**kwargs) -> Pipeline:
    """
    This pipeline preprocesses data and creates features for both Model 1 and Model 2.
    """
    return pipeline(
        [
            # --- Nodes for Model 1 Feature Path ---
            node(
                func=preprocess_raw_data,
                inputs="raw_loan_data",
                outputs="preprocessed_loan_data",
                name="preprocess_raw_data_node",
            ),
            node(
                func=lambda df: df, 
                inputs="preprocessed_loan_data",
                outputs="daily_actuals_for_datalake",
                name="save_actuals_for_datalake_node",
            ),
            node(
                func=create_features_for_model1,
                inputs=["preprocessed_loan_data", "params:model_1_params"],
                outputs="model_1_feature_data",
                name="create_model_1_features_node",
            ),
            node(
                func=create_intraday_features_for_model2,
                inputs=["raw_hourly_loan_data", "params:model_2_params"],
                outputs="model_2_feature_data",
                name="create_model_2_features_node",
            )
        ]
    )
