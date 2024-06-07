import pandas as pd
import sqlite3
from typing import Union

class IngestData:
    """
    A class used to ingest data from various sources such as CSV files and SQL databases.

    Methods:
    --------
    __init__() -> None:
        Initializes the IngestData class.
    
    get_data(source: str, source_type: str = 'csv', table_name: str = '') -> pd.DataFrame:
        Reads data from the specified source, which can be a CSV file or a SQL database.
    """

    def __init__(self) -> None:
        """
        Initializes the IngestData class.
        """
        self.df = None

    def get_data(self, source: str, source_type: str = 'csv', table_name: str = '') -> pd.DataFrame:
        """
        Reads data from the specified source, which can be a CSV file or a SQL database.

        Parameters:
        -----------
        source : str
            The path to the CSV file or the database connection string.
        
        source_type : str, optional
            The type of the data source (default is 'csv').
            Options: 'csv', 'sql'.
        
        table_name : str, optional
            The table name in the SQL database (required if source_type is 'sql').

        Returns:
        --------
        pd.DataFrame
            The ingested dataset.
        """
        if source_type == 'csv':
            self.df = pd.read_csv(source)
            return self.df
        elif source_type == 'sql':
            if not table_name:
                raise ValueError("Table name must be provided when source_type is 'sql'")
            conn = sqlite3.connect(source)
            self.df = pd.read_sql(f"SELECT * FROM {table_name}", con=conn)
            return self.df




        