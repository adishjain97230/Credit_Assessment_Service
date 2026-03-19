import pandas as pd
from config import constants, switch_properties
import config.logging_config as logging_config
import re
import os

module_properties = switch_properties.SWITCH_PROPERTIES[constants.data_cleaning]
logger = logging_config.get_logger(__name__)

def convertTermToNumeric(df, column):
    """Convert values like '36 months', '60 months' to int/float. Ignores leading whitespace."""
    series = df[column]
    extracted = series.astype(str).str.extract(r'^\s*(\d+(?:\.\d+)?)\s*months?')[0]
    df[column] = pd.to_numeric(extracted, errors='coerce')
    return df.dropna(how="any").copy()

def convertEmpLengthToNumeric(df, column):
    """
    Convert emp_length (e.g. '< 1 year', '1 year', '10+ years', '36 months') to months.
    - year/years -> value * 12; month/months -> value unchanged.
    - '< 1 year' -> 6 (months).
    - If string contains '+' (e.g. '10+ years'), adds column {column}_10_plus = 1.
    - Drops rows where conversion fails.
    """
    df = df.copy()
    series = df[column].astype(str).str.strip()

    # Flag column for "+" (e.g. 10+ years)
    df[column + "_10_plus"] = series.str.contains(r"\+", regex=True).astype(int)

    def to_months(s):
        if pd.isna(s) or str(s).strip() == "" or str(s).strip().lower() == "nan":
            return float("nan")
        s = str(s).strip()
        # "< 1 year" or "<1 year" -> 6 months
        if re.match(r"^\s*<\s*1\s*year", s, re.IGNORECASE):
            return 6.0
        # Match: number, optional +, then year(s) or month(s)
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*\+?\s*(year|month)", s, re.IGNORECASE)
        if not m:
            return float("nan")
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("year"):
            num *= 12
        return num

    df[column] = series.apply(to_months)
    return df.dropna(how="any").copy()

def ConvertEarliestCrLineToNumeric(df, column):
    """
    Convert 'Oct-1998', 'Sep-2002' style column to two int columns: month (1-12) and year.
    Uses column name as prefix to avoid overwriting (e.g. earliest_cr_line_month, earliest_cr_line_year).
    Rows that don't parse become NaN; drop with dropna if needed.
    """
    df = df.copy()
    dt = pd.to_datetime(df[column], format="%b-%Y", errors="coerce")
    df[f"{column}_month"] = dt.dt.month.astype("Int64")   # nullable int
    df[f"{column}_year"] = dt.dt.year.astype("Int64")
    return df.drop(columns=[column]).dropna(how="any").copy()

def getRawData(n = None):
    path = module_properties[constants.dataset_path]

    try:
        if (n == None): df = pd.read_csv(path, low_memory=False)
        else: df = pd.read_csv(path, nrows=n, low_memory=False)
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
        return df.copy()

def selectColumns(df):
    target_column = list(module_properties[constants.target_column])[0]
    target_column_mapping = module_properties[constants.target_column][target_column]
    df["y"] = df[target_column].map(target_column_mapping)
    columns = module_properties[constants.selected_columns] + ["y"]
    return df[columns].dropna(how="any").copy()


columnSpecificCleaning = {
    "term": convertTermToNumeric,
    "emp_length": convertEmpLengthToNumeric,
    "earliest_cr_line": ConvertEarliestCrLineToNumeric,
}

def convertStringColumnsToNumeric(df):
    for key, value in columnSpecificCleaning.items():
        df = value(df, key)

    return df

def main(**kwargs):
    repeat_all_parts = kwargs.get("repeat_all_parts", True)

    if not(repeat_all_parts) and os.path.exists(module_properties[constants.post_dataset_path]):
        logger.info("Data cleaning already completed. Skipping...")
        return;
    
    df = getRawData()
    df = selectColumns(df)
    df = convertStringColumnsToNumeric(df)
    df.to_parquet(module_properties[constants.post_dataset_path], index=False)

if __name__ == "__main__":
    main()