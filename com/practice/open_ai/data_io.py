# from future import annotations

import io
import csv
from typing import Optional, Dict, Any

import pandas as pd

def load_df(file_bytes: bytes,encoding: str = "utf-8", delimiter: str=None) -> pd.DataFrame:
    if delimiter is None:
        try:
            sample = file_bytes[: 64 * 1024].decode(encoding, errors="ignore")
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","  #
        # Wrap bytes in a buffer for pandas
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(
        buf,
        sep=delimiter,
        encoding=encoding,
        on_bad_lines="skip",
        low_memory=False
    )
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    print(df)
    return df.to_csv(index=False).encode("utf-8")