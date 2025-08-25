from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from daily_deposit.pipelines import data_ingestion as di
from daily_deposit.pipelines import data_engineering as de
from daily_deposit.pipelines import data_science as ds
from daily_deposit.pipelines import reporting as rp
from daily_deposit.pipelines import prediction as pred
from daily_deposit.pipelines import backfill as bf 

def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines for different operational modes."""
    
    p_ingestion = di.create_pipeline()
    p_engineering = de.create_pipeline()
    p_primary_split = ds.create_primary_split_pipeline()

    p_m1_training_dev = ds.create_m1_training_pipeline()

    p_reporting_m1_only = rp.create_m1_only_reporting_pipeline()

    p_backfill_simulation = bf.create_pipeline()
    m1_training_logic_for_prod = ds.create_m1_training_pipeline()
    p_m1_training_prod = pipeline(
        pipe=m1_training_logic_for_prod,
        inputs={"model_1_features_dev": "model_1_feature_data"},
        outputs={
            "model_1_q0.1": "model_1_prod_q0.1",
            "model_1_q0.5": "model_1_prod_q0.5",
            "model_1_q0.9": "model_1_prod_q0.9",
            "model_1_mean": "model_1_prod_mean",
            "model_1_oos_predictions": None, 
        },
        parameters={"model_1_training": "params:model_1_training"},
        namespace="m1_prod_training"
    )

    p_history_update = pred.create_history_update_pipeline()
    p_prediction_m1_only = pred.create_m1_only_prediction_pipeline()

    p_weekly_train_and_report_m1 = (
        p_ingestion
        + p_engineering
        + p_primary_split          # STEP 1: Create model_1_features_dev & _test
        + p_m1_training_dev        # STEP 2: Train DEV models on model_1_features_dev
        + p_m1_training_prod       # STEP 3: Train PROD models on model_1_feature_data (full set)
        + p_reporting_m1_only      # STEP 4: Create report using DEV models and model_1_features_test
    )


    p_daily_predict_m1 = (p_ingestion + p_engineering + p_history_update + p_prediction_m1_only)
    
    dev_train_and_report_only = (
        p_ingestion
        + p_engineering
        + p_primary_split
        + p_m1_training_dev
        + p_reporting_m1_only
    )
    
    p_backfill_history = pred.create_backfill_pipeline()

    return {
        "__default__": dev_train_and_report_only,
        
        "dev_train_and_report_only": dev_train_and_report_only,
        
        "weekly_train_and_report_m1": p_weekly_train_and_report_m1,
        
        "history_update": p_history_update,
        
        "daily_predict_m1": p_daily_predict_m1,
        
        # Utilities for testing individual parts
        "ingest_only": p_ingestion,
        
        "backfill_history": p_backfill_history,

        "backfill_simulation": p_backfill_simulation
    }
