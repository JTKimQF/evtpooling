import logging
import os

import pandas as pd

from ..constants import *
from .extract import extract_file
from .load import load_file
from .transform import transform_data

logger = logging.getLogger(__name__)


def etl_pipeline(file_path: str, save_file: bool = False, **kwargs) -> pd.DataFrame:
    """
    ETL pipeline to extract, transform, and load data from a file.

    Parameters
    ----------
    file_path : str
        Path to the input file.
    **kwargs : dict
        Additional parameters for extraction and transformation.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame ready for analysis.
    """
    try:
        unexpected_keys = set(kwargs) - ALL_ALLOWED_KWARGS
        if unexpected_keys:
            raise ValueError(f"Unexpected keyword argument(s): {unexpected_keys}")

        # Merge default etl kwargs with user kwargs
        extract_kwargs = {k: kwargs[k] for k in DEFAULT_EXTRACT_KWARGS if k in kwargs}
        extract_final = DEFAULT_EXTRACT_KWARGS.copy()
        extract_final.update(extract_kwargs)

        transform_kwargs = {k: kwargs[k] for k in DEFAULT_TRANSFORM_KWARGS if k in kwargs}
        transform_final = DEFAULT_TRANSFORM_KWARGS.copy()
        transform_final.update(transform_kwargs)

        load_kwargs = {k: kwargs[k] for k in DEFAULT_LOAD_KWARGS if k in kwargs}
        load_final = DEFAULT_LOAD_KWARGS.copy()
        load_final.update(load_kwargs)

        # Extract, transform, and load data
        raw_stock_data = extract_file(file_path, **extract_final)
        clean_data_week, _ = transform_data(raw_stock_data, **transform_final)
        directory = os.path.dirname(file_path)
        week_filepath = os.path.join(directory, "losses_week.xlsx")
        loaded_data = load_file(clean_data_week, week_filepath)
        logging.info("Successfully extracted, transformed and loaded data.")

        if not save_file:
            os.remove(week_filepath)

        return loaded_data
    # Handle exceptions, log messages
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise


# Configure logging ONLY when run directly (not when imported)
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
    )
