from config import logging_config
from data_cleaning import dataCleaning
from config import constants
import argparse

def strToBool(value):
    return value.lower() in ("true", "1", "t", "y")

def getCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{constants.repeat_all_parts}", "-r", type=strToBool, default=True, help="If it is False, then it will check whether a part has been completed and if it has, it will not repeat it.")
    return parser.parse_args()

def main():
    args = getCommandLineArguments()
    logging_config.setup_logger()
    dataCleaning.main(repeat_all_parts=args.repeat_all_parts)



if __name__ == "__main__":
    main()