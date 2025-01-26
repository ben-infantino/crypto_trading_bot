import pandas as pd


def ingest_csv(file_path, format="pandas"):
    """
    Ingests a single CSV file and returns it in the specified format.

    :param file_path: Path to the CSV file.
    :param format: Desired output format ("pandas" or "numpy").
    :return: Data in the specified format.
    """
    headers = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    data = pd.read_csv(file_path, names=headers)

    if format == "numpy":
        return data.to_numpy()
    elif format == "pandas":
        return data
    else:
        raise ValueError("Invalid format specified. Choose 'pandas' or 'numpy'.")

