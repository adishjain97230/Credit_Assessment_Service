from config import switch_properties, logging_config, constants
import pandas as pd
from models import split

logger = logging_config.get_logger(__name__)
module_properties = switch_properties.SWITCH_PROPERTIES[constants.models]

def encodeCategoricalColumns(df):
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df

def main():
    logger.info("Logistic Regression Model")
    
    df = pd.read_parquet(module_properties[constants.dataset_path])

    df = encodeCategoricalColumns(df)

    y = df["y"]
    X = df.drop(columns=["y"])

    split_data = split.SplitData(X, y, stratify=y)

    split_data.encodeCategoricalColumns()

    





if __name__ == "__main__":
    main()