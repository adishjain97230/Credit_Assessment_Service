import pandas as pd
from config import constants, switch_properties
import config.logging_config as logging_config

module_properties = switch_properties.SWITCH_PROPERTIES[constants.data_cleaning]
logger = logging_config.get_logger(__name__)

def getRawData():
    path = module_properties[constants.dataset_path]

    try:
        df = pd.read_csv(path)
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
    return df[columns]


def main():
    df = getRawData()
    df = selectColumns(df)
    df = df.dropna(how="any")

if __name__ == "__main__":
    main()