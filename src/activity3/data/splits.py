"""
Temporal splits (no leakage). Dates strictly increase across splits.
"""
import pandas as pd

from src.activity3.config import TRAIN_END, VAL_END


def temporal_split(dates):
    """
    Given a pandas DatetimeIndex / array of dates, return three boolean
    masks (train, val, test) honoring the global config TRAIN_END/VAL_END.
    """
    dates = pd.to_datetime(pd.Series(dates).values)
    train_end = pd.to_datetime(TRAIN_END)
    val_end   = pd.to_datetime(VAL_END)

    train_mask = dates <= train_end
    val_mask   = (dates > train_end) & (dates <= val_end)
    test_mask  = dates > val_end
    return train_mask, val_mask, test_mask
