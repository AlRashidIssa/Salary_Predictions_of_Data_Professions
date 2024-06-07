from sklearn.model_selection import train_test_split
import pandas as pd

class SplitData:
    """
    A class used to split the dataset into training and testing sets.

    Attributes:
    -----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Testing feature set.
    y_train : pd.Series
        Training target set.
    y_test : pd.Series
        Testing target set.

    Methods:
    --------
    __init__() -> None:
        Initializes the SplitData class.

    split(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = None) -> tuple:
        Splits the dataset into training and testing sets.
    """

    def __init__(self) -> None:
        """
        Initializes the SplitData class.
        """
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()

    def split(self, df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = None) -> tuple:
        """
        Splits the dataset into training and testing sets.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be split.
        
        target : str
            The target column name in the dataset.
        
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        
        random_state : int, optional
            The random seed used by the random number generator (default is None).
        
        Returns:
        --------
        tuple:
            A tuple containing the training feature set, testing feature set, training target set, and testing target set.
        """
        X = df.drop(columns=[target])
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test