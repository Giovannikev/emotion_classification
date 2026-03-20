from pathlib import Path
from typing import Optional

import pandas as pd

from src import PROJECT_ROOT


def get_data_path() -> Path:
    return PROJECT_ROOT / "data" / "tweets.csv"


def load_raw_tweets(n_rows: Optional[int] = None) -> pd.DataFrame:
    csv_path = get_data_path()
    column_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        names=column_names,
        nrows=n_rows,
    )
    return df
