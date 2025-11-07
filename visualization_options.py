"""
visualization_options.py
Small helper to map text-based options to visualization functions.
"""
from typing import Optional


VALID_MAIN_OPTIONS = {"1": "scatterplot", "2": "boxplot", "3": "multivariate"}


def get_main_choice(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    return VALID_MAIN_OPTIONS.get(raw)
