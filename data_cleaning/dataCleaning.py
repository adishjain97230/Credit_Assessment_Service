import pandas as pd
from config import constants, switch_properties

def main():
    df = pd.read_csv(switch_properties.SWITCH_PROPERTIES[constants.data_cleaning][constants.dataset_path])
    print(df.head())

if __name__ == "__main__":
    main()