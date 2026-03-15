# config/switch_properties.py
import json
import config.constants as constants
from pathlib import Path

_PATH = constants.switch_properties

with open(_PATH, "r") as f:
    SWITCH_PROPERTIES = json.load(f)