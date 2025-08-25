import pandas as pd
import logging
from typing import Dict, Any
from sqlalchemy import create_engine
from trino.auth import BasicAuthentication

log = logging.getLogger(__name__)

def fetch_data_from_trino(query: str) -> pd.DataFrame:
    """
    Connects to Trino, executes a query, and returns the result as a DataFrame.
    """
    log.info(f"Connecting to Trino database to execute query...")
    
    try:
        engine = create_engine(
            "trino://****/datalake",
            connect_args={
                "auth": BasicAuthentication("ai_user","****"),
                "http_scheme": "https",
                "verify": False
            }
        )
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
            
        log.info(f"Successfully fetched {len(df)} rows.")
        return df

    except Exception as e:
        log.error(f"Failed to fetch data from Trino: {e}")
        raise
