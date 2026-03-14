import json
import csv
import pandas as pd
from pathlib import Path
from config import constants

def getSwitchProperties():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / constants.switch_properties
    with open(file_path, "r") as f:
        switch_properties = json.load(f)
    return switch_properties
    

if __name__ == "__main__":
    switch_properties = getSwitchProperties()
    