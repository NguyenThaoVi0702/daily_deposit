from kedro.pipeline import Pipeline, node
from .nodes import split_primary_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_primary_data,
            inputs=["model_1_feature_data", "model_2_feature_data", "params:primary_split"],
            outputs=[
                "model_1_features_dev", "model_1_features_test",
                "model_2_features_dev", "model_2_features_test"
            ],
            name="split_primary_feature_data"
        )
    ])
