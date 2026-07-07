import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class Inspector:

    def __init__(self):
        self.line = '─' * 170
    

    def get_columns_types(self,dataframe,car_th=20,cat_th=10):
        """
        Classifies all dataframe columns into four groups based on their data type and cardinality.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to be analyzed.
        car_th : int, optional
            Cardinality threshold above which an object/category column is treated as
            high-cardinality (cardinal) rather than categorical (default: 20).
        cat_th : int, optional
            Unique-value threshold below which a numeric column is treated as
            categorical rather than continuous (default: 10).

        Returns
        -------
        tuple : (cat_cols, num_cols, num_but_cat, cat_but_car)
            cat_cols     : list of str — categorical columns (including low-cardinality numerics).
            num_cols     : list of str — continuous numeric columns.
            num_but_cat  : list of str — numeric columns that look categorical (few unique values).
            cat_but_car  : list of str — object/category columns with too many unique values
                        to be treated as categorical (high-cardinality).
        """
        STRING_TYPES = {'object', 'bool', 'category', 'str', 'string'}
        cat_cols = [col for col in dataframe.columns if
                    str(dataframe[col].dtype) in STRING_TYPES]
        num_but_cat = [col for col in dataframe.columns if
                       pd.api.types.is_numeric_dtype(dataframe[col]) and dataframe[
                           col].nunique() < cat_th]
        cat_but_car = [col for col in dataframe.columns if
                       str(dataframe[col].dtype) in STRING_TYPES and dataframe[
                           col].nunique() > car_th]
        c_c = [col for col in num_but_cat if col not in cat_cols ]
        cat_cols = cat_cols + c_c
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        num_cols = [col for col in dataframe.columns if
                    pd.api.types.is_numeric_dtype(dataframe[col]) and col not in num_but_cat]
        return cat_cols, num_cols, num_but_cat, cat_but_car

    def check_dataframe(self,dataframe, n=5):
        """
        Prints a general overview of the dataframe: head, random sample, tail,
        shape, dtypes/info, missing-value ratios, and duplicate row count.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to inspect.
        n : int, optional
            Number of rows shown for head, sample, and tail sections (default: 5).

        Returns
        -------
        None
            All output is printed to stdout.
        """
        print(f'\n{self.line}')
        print(' Head '.center(170))
        print(self.line)
        print(dataframe.head(n))
        print(f'\n{self.line}')
        print(' Sample '.center(170))
        print(self.line)
        print(dataframe.sample(n))
        print(f'\n{self.line}')
        print(' Tail '.center(170))
        print(self.line)
        print(dataframe.tail(n))
        print(f'\n{self.line}')
        print(' Shape '.center(170))
        print(self.line)
        print('Rows: ', dataframe.shape[0])
        print('Columns: ', dataframe.shape[1])
        print(f'\n{self.line}')
        print(' Info '.center(170))
        print(self.line)
        print(dataframe.info())
        print(f'\n{self.line}')
        print(' NA '.center(170))
        print(self.line)
        print(dataframe.isnull().mean())
        print(f'\n{self.line}')
        print(' Duplicate Values '.center(170))
        print(self.line)
        print('Count: ', dataframe.duplicated().sum())
        print('Ratio: ', (dataframe.duplicated().sum()) / (dataframe.shape[0]))    
