from config import logging_config
from data_cleaning import dataCleaning

def main():
    logging_config.setup_logger()
    dataCleaning.main()



if __name__ == "__main__":
    main()