from functools import partial, update_wrapper
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data_to_dev_and_test,
    split_features_and_target,
    tune_hyperparameters,
    train_model,
    create_oos_predictions,
    combine_predictions,
    create_model_2_training_data,
    create_model_2_common_features,
    create_error_target_for_model_2,
    tune_hyperparameters_m2,
    train_corrector_model_m2,
)

def create_primary_split_pipeline(**kwargs) -> Pipeline:
    """Pipeline that splits data into dev and test sets."""
    return pipeline([
        node(
            func=split_data_to_dev_and_test,
            inputs=["model_1_feature_data", "model_2_feature_data", "params:primary_split"],
            outputs=[
                "model_1_features_dev",
                "model_1_features_test",
                "model_2_features_dev",
                "model_2_features_test",
            ],
            name="primary_data_split"
        )
    ])


def create_m1_training_pipeline(**kwargs) -> Pipeline:
    prediction_targets = kwargs.get("prediction_targets", [0.1, 0.5, 0.9, "mean"])
    
    #input_dataset = "model_1_features_dev" if pipeline_mode == 'dev' else "model_1_feature_data"

    def partial_with_name(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func
    
    
    initial_split_node = node(
        func=split_features_and_target,
        inputs=["model_1_features_dev", "params:model_1_training"], # <-- Use the DEV dataset
        outputs=["X", "y", "model_1_feature_list"],
        name="split_data_node"
    )


    training_pipelines = []
    oos_prediction_outputs = []

    for pred_type in prediction_targets:
        target_name = f"q{pred_type}" if isinstance(pred_type, float) else str(pred_type)
        model_ds = f"model_1_{target_name}"
        best_params_ds = f"best_params_{target_name}"
        oos_preds_ds = f"oos_predictions_{target_name}"
        oos_prediction_outputs.append(oos_preds_ds)

        tune_func = partial(tune_hyperparameters, pred_type=pred_type)
        update_wrapper(tune_func, tune_hyperparameters)
        train_func = partial(train_model, pred_type=pred_type)
        update_wrapper(train_func, train_model)
        oos_func = partial(create_oos_predictions, pred_type=pred_type)
        update_wrapper(oos_func, create_oos_predictions)
        target_pipeline = pipeline([
            node(func=tune_func, inputs={"X": "X", "y": "y", "params": "params:model_1_training"}, outputs=best_params_ds, name=f"tune_hyperparameters_{target_name}"),
            node(func=train_func, inputs={"X": "X", "y": "y", "best_params": best_params_ds}, outputs=model_ds, name=f"train_model_{target_name}"),
            node(func=oos_func, inputs={"X": "X", "y": "y", "model": model_ds, "params": "params:model_1_training"}, outputs=oos_preds_ds, name=f"create_oos_predictions_{target_name}")
        ])
        training_pipelines.append(target_pipeline)
    combine_node = node(func=combine_predictions, inputs=oos_prediction_outputs, outputs="model_1_oos_predictions", name="combine_all_oos_predictions")
    return pipeline([initial_split_node]) + sum(training_pipelines, pipeline([])) + pipeline([combine_node])



def create_m2_training_pipeline(
    pipeline_mode: str = 'dev', **kwargs
) -> Pipeline:
    targets_to_correct = kwargs.get("targets_to_correct", ["mean", "median"])
    
    input_intraday_features = "model_2_features_dev" if pipeline_mode == 'dev' else "model_2_feature_data"

    preamble_pipeline = pipeline([
        node(
            func=create_model_2_training_data,
            inputs={
                "m1_oos_predictions": "model_1_oos_predictions",
                "daily_actuals": "preprocessed_loan_data",
                "intraday_features": "model_2_features_dev", 
                "params": "params:model_2_training"
            },
            outputs="m2_base_training_data",
            name="create_m2_base_data"
        ),
        node(
            func=create_model_2_common_features,
            inputs="m2_base_training_data",
            outputs=["X_m2_common", "m2_common_feature_list"], 
            name="create_m2_common_features"
        )
    ])


    training_pipelines = []
    for target_type in targets_to_correct:

        create_target_func = partial(create_error_target_for_model_2, target_type=target_type)
        update_wrapper(create_target_func, create_error_target_for_model_2)
        
        tune_func = partial(tune_hyperparameters_m2, target_type=target_type)
        update_wrapper(tune_func, tune_hyperparameters_m2)

        train_func = partial(train_corrector_model_m2, target_type=target_type)
        update_wrapper(train_func, train_corrector_model_m2)

        if pipeline_mode == 'prod':
            model_ds = f"model_2_prod_{target_type}_corrector"
            feature_list_ds = f"model_2_feature_list_prod_{target_type}"
        else: 
            model_ds = f"model_2_{target_type}_corrector"
            feature_list_ds = f"model_2_feature_list_{target_type}"

        target_pipeline = pipeline([
            node(
                func=create_target_func,
                inputs="m2_base_training_data",
                outputs=f"y_m2_{target_type}",
                name=f"create_error_target_{target_type}"
            ),
            node(
                func=tune_func,
                inputs={"X": "X_m2_common", "y": f"y_m2_{target_type}", "params": "params:model_2_training"},
                outputs=f"best_params_m2_{target_type}",
                name=f"tune_hyperparameters_m2_{target_type}"
            ),
            node(
                func=lambda x: x, 
                inputs="m2_common_feature_list",
                outputs=feature_list_ds,
                name=f"save_feature_list_m2_{target_type}"
            ),
            node(
                func=train_func,
                inputs={"X": "X_m2_common", "y": f"y_m2_{target_type}", "best_params": f"best_params_m2_{target_type}"},
                outputs=model_ds,
                name=f"train_corrector_model_m2_{target_type}"
            )
        ])
        training_pipelines.append(target_pipeline)
    
    return preamble_pipeline + sum(training_pipelines, pipeline([]))
