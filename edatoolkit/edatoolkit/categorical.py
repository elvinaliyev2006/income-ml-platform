import pandas as pd
import matplotlib.pyplot as plt 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class CategoricalAnalyzer:
    def __init__(self):

      self.line = '─' * 170


    def cat_summary(self, dataframe, cat_cols, plot=False, width_for_graph=13, height_for_graph=5):
        """
        Prints value counts and percentage ratios for every categorical column,
        and optionally renders a bar chart for each one.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing the categorical columns.
        cat_cols : list of str
            List of categorical column names to summarize.
        plot : bool, optional
            If True, displays a labelled bar chart for each column (default: False).
        width_for_graph : int, optional
            Width of each figure in inches (default: 13).
        height_for_graph : int, optional
            Height of each figure in inches (default: 5).

        Returns
        -------
        None
            All output is printed to stdout; charts are displayed inline.

        Raises
        ------
        ValueError
            If cat_cols is empty.
        """
        if cat_cols:
            print(f'\n{self.line}')
            print(' Categorical Variable Summary '.center(170))
            print(self.line)
            for col in cat_cols:
                val_c = pd.DataFrame({
                    'Count': dataframe[col].value_counts(),
                    'Ratio': 100 * dataframe[col].value_counts() / len(dataframe)})
                print(f"\nColumn: {col}")
                print(val_c)

                if plot:
                    fig, ax = plt.subplots(figsize=(width_for_graph, height_for_graph))
                    fig.patch.set_facecolor('white')
                    ax.set_facecolor('white')
                    ax.grid(False)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#cccccc')

                    order = dataframe[col].value_counts().index
                    bars = ax.bar(order, dataframe[col].value_counts()[order],
                                  color='#2A9D8F', alpha=0.85, edgecolor='white')
                    for bar in bars:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + max(dataframe[col].value_counts()) * 0.01,
                                str(int(bar.get_height())),
                                ha='center', va='bottom', fontsize=10)
                    ax.set_title(col, fontsize=13, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()

            print(self.line)
        else:
            raise ValueError('! cat_cols is empty')