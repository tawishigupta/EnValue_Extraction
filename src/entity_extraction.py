# In entity_extraction.py

import re
from constants import entity_unit_map

def extract_entity_value(text, entity_name):
    allowed_units = entity_unit_map.get(entity_name, set())
    pattern = r'(\d+(\.\d+)?)\s*(' + '|'.join(allowed_units) + ')'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        value, _, unit = matches[0]
        return f"{float(value):.2f} {unit.lower()}"
    return ""