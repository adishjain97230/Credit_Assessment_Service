import pandas as pd
from config import constants, switch_properties
import config.logging_config as logging_config

module_properties = switch_properties.SWITCH_PROPERTIES[constants.data_cleaning]
logger = logging_config.get_logger(__name__)


def convertTermToNumeric(series):
    """Convert values like '36 months', '60 months' to int/float. Ignores leading whitespace."""
    extracted = series.astype(str).str.extract(r'^\s*(\d+(?:\.\d+)?)\s*months?')[0]
    return pd.to_numeric(extracted, errors='coerce')

def getRawData():
    path = module_properties[constants.dataset_path]

    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        logger.error("CSV file not found at %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file at %s is empty", path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV at %s: %s", path, e)
        raise
    else:
        logger.info("Loaded CSV from %s with shape %s", path, df.shape)
        return df

def selectColumns(df):
    target_column = list(module_properties[constants.target_column])[0]
    target_column_mapping = module_properties[constants.target_column][target_column]
    df["y"] = df[target_column].map(target_column_mapping)
    columns = module_properties[constants.selected_columns] + ["y"]
    return df[columns].copy().dropna(how="any")

def convertStringColumnsToNumeric(df): #TODO clean
    df["term"] = convertTermToNumeric(df["term"])
    print(df.shape)
    df = df.dropna(how="any")
    df = df.copy()
    print(df.shape)
    return df

def main(): #TODO clean
    df = getRawData()
    df = selectColumns(df)
    df = convertStringColumnsToNumeric(df)
    print(df.head())

if __name__ == "__main__":
    main()