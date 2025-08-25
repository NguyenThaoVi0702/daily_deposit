from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_data_from_trino

def create_pipeline(**kwargs) -> Pipeline:
    """Creates the data ingestion pipeline."""
    return pipeline(
        [
            node(
                func=fetch_data_from_trino,
                inputs=["params:data_ingestion.daily_balance_query"],
                outputs="raw_loan_data", 
                name="fetch_daily_loan_data"
            )
        ]
    )
