# src/daily_loan/pipelines/reporting/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from functools import partial, update_wrapper
from .nodes import (
    generate_m1_only_holdout_predictions, create_m1_only_evaluation_table,
    plot_m1_only_forecast_comparison, generate_m1_only_docx_report,
    analyze_point_forecasts, analyze_interval_forecasts, analyze_quantile_losses,
    plot_feature_importance
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the final reporting pipeline with explicit execution stages."""

    def partial_with_name(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    all_models = {
        "m1_q01": "model_1_q0.1", "m1_q05": "model_1_q0.5", "m1_q09": "model_1_q0.9", "m1_mean": "model_1_mean",
        "m2_median": "model_2_median_corrector", "m2_mean": "model_2_mean_corrector"
    }

    # Stage 1: Generate predictions and the main results table
    prediction_pipeline = pipeline([
        node(
            generate_final_holdout_predictions,
            inputs=dict(
                m1_features_test="model_1_features_test",
                m2_features_test="model_2_features_test",
                **{f"model_{k.replace('.', '')}": v for k, v in all_models.items()}
            ),
            outputs="test_set_predictions",
            name="generate_holdout_predictions"
        ),
        node(
            create_evaluation_table,
            inputs=dict(
                predictions="test_set_predictions",
                daily_test_set="model_1_features_test",
                params_m1_training="params:model_1_training",
                params_reporting="params:reporting"
            ),
            outputs="evaluation_results_table",
            name="create_final_evaluation_table"
        )
    ])

    # Stage 2: Calculate all metrics from the results table
    analysis_pipeline = pipeline([
        node(analyze_point_forecasts, "evaluation_results_table", "point_forecast_metrics"),
        node(analyze_interval_forecasts, "evaluation_results_table", "interval_metrics"),
        node(analyze_quantile_losses, "evaluation_results_table", "quantile_loss_metrics"),
    ])

    # Stage 3: Generate all plots
    fi_plot_nodes = []
    fi_plot_outputs_for_docx = {}
    for model_name, model_ds in all_models.items():
        sanitized_name = model_name.replace('.', '')
        fig_ds = f"feature_importance_{sanitized_name}_fig"
        png_ds = f"feature_importance_{sanitized_name}_png"
        
        fi_plot_nodes.append(
            node(
                func=partial_with_name(plot_feature_importance, model_name=model_name),
                inputs=model_ds,
                outputs=[fig_ds, png_ds], 
                name=f"plot_fi_{sanitized_name}"
            )
        )
        fi_plot_outputs_for_docx[f"fi_{sanitized_name}"] = fig_ds
    
    comparison_plot_nodes = [
        node(
            func=partial_with_name(plot_forecast_comparison, forecast_type='median'),
            inputs={"results_df": "evaluation_results_table", "factors": "params:reporting.m2_correction_factors"},
            outputs="forecast_comparison_plot_median",
            name="plot_comparison_median"
        ),
        node(
            func=partial_with_name(plot_forecast_comparison, forecast_type='mean'),
            inputs={"results_df": "evaluation_results_table", "factors": "params:reporting.m2_correction_factors"},
            outputs="forecast_comparison_plot_mean",
            name="plot_comparison_mean"
        ),
    ]
    
    plotting_pipeline = pipeline(fi_plot_nodes + comparison_plot_nodes)

    # Stage 4: Assemble the final document from all created artifacts
    report_generation_pipeline = pipeline([
        node(
            generate_docx_report,
            inputs=dict(
                point_metrics="point_forecast_metrics",
                interval_metrics="interval_metrics",
                quantile_losses="quantile_loss_metrics",
                plot_comp_median="forecast_comparison_plot_median",
                plot_comp_mean="forecast_comparison_plot_mean",
                **fi_plot_outputs_for_docx
            ),
            outputs=None, 
            name="generate_final_report_node"
        )
    ])


    # Chain all stages together explicitly
    return prediction_pipeline + analysis_pipeline + plotting_pipeline + report_generation_pipeline
    
    

def create_m1_only_reporting_pipeline(**kwargs) -> Pipeline:
    """Creates a reporting pipeline that only evaluates M1 performance."""
    
    def partial_with_name(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func
        
    m1_models = {
        "m1_q01": "model_1_q0.1", "m1_q05": "model_1_q0.5", 
        "m1_q09": "model_1_q0.9", "m1_mean": "model_1_mean"
    }

    # Stage 1: Generate predictions and results table
    prediction_pipeline = pipeline([
        node(
            generate_m1_only_holdout_predictions,
            inputs=dict(
                m1_features_test="model_1_features_test",
                **{f"model_{k}": v for k, v in m1_models.items()}
            ),
            outputs="m1_test_set_predictions",
            name="generate_m1_holdout_predictions"
        ),
        node(
            create_m1_only_evaluation_table,
            inputs=["m1_test_set_predictions", "model_1_features_test", "params:model_1_training"],
            outputs="evaluation_results_table",
            name="create_m1_evaluation_table"
        )
    ])

    # Stage 2: Calculate metrics (these nodes can be reused)
    analysis_pipeline = pipeline([
        node(analyze_point_forecasts, "evaluation_results_table", "point_forecast_metrics"),
        node(analyze_interval_forecasts, "evaluation_results_table", "interval_metrics"),
        node(analyze_quantile_losses, "evaluation_results_table", "quantile_loss_metrics"),
    ])

    # Stage 3: Generate plots
    fi_plot_nodes = []
    fi_plot_outputs_for_docx = {}
    for model_name, model_ds in m1_models.items():
        sanitized_name = model_name.replace('.', '')
        fig_ds = f"feature_importance_{sanitized_name}_fig"
        png_ds = f"feature_importance_{sanitized_name}_png"
        
        fi_plot_nodes.append(
            node(
                func=partial_with_name(plot_feature_importance, model_name=model_name),
                inputs=model_ds,
                outputs=[fig_ds, png_ds], 
                name=f"plot_fi_{sanitized_name}"
            )
        )
        fi_plot_outputs_for_docx[f"fi_{sanitized_name}"] = fig_ds
        
    comparison_plot_nodes = [
        node(
            partial_with_name(plot_m1_only_forecast_comparison, forecast_type='median'),
            "evaluation_results_table", "forecast_comparison_plot_median"
        ),
        node(
            partial_with_name(plot_m1_only_forecast_comparison, forecast_type='mean'),
            "evaluation_results_table", "forecast_comparison_plot_mean"
        ),
    ]
    plotting_pipeline = pipeline(fi_plot_nodes + comparison_plot_nodes)

    # Stage 4: Assemble the final docx report
    report_generation_node = node(
        generate_m1_only_docx_report,
        inputs=dict(
            point_metrics="point_forecast_metrics",
            interval_metrics="interval_metrics",
            quantile_losses="quantile_loss_metrics",
            fi_m1_q01="feature_importance_m1_q01_fig",
            fi_m1_q05="feature_importance_m1_q05_fig",
            fi_m1_q09="feature_importance_m1_q09_fig",
            fi_m1_mean="feature_importance_m1_mean_fig",
            plot_comp_median="forecast_comparison_plot_median",
            plot_comp_mean="forecast_comparison_plot_mean"
        ),
        outputs=None, # Saves the file itself
        name="generate_m1_only_report_node"
    )
    
    return prediction_pipeline + analysis_pipeline + plotting_pipeline + pipeline([report_generation_node])
    

    

    
    
