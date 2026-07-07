import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class NormalityAnalyzer:
    
    def __init__(self):
        self.line = '─' * 170
    

    def descriptive_analysis(self,dataframe,num_cols):
        """
        Prints an extended descriptive statistics table for all numerical columns.

        In addition to the standard describe() output, the table includes the median,
        coefficient of variation (CV%), skewness, and kurtosis for each column.
        Percentiles reported: 1%, 5%, 25%, 50%, 75%, 95%, 99%.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing the columns to summarize.
        num_cols : list of str
            List of numerical column names to include in the analysis.

        Returns
        -------
        None
            Results are printed to stdout as a transposed DataFrame.

        Raises
        ------
        ValueError
            If num_cols is empty.
        """
        if num_cols:
            df_desc = dataframe[num_cols].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            cv = (dataframe[num_cols].std() / dataframe[num_cols].mean().replace(0,
                                                                                                     np.nan) * 100).round(
                3)
            cv = cv.replace([np.inf, -np.inf], np.nan)
            median = dataframe[num_cols].median()
            skew = dataframe[num_cols].skew()
            kurtosis = dataframe[num_cols].kurtosis()
            cv_col = pd.DataFrame(cv).T
            cv_col.index = ['cv%']
            median_col = pd.DataFrame(median).T
            median_col.index = ['median']
            skew_col = pd.DataFrame(skew).T
            skew_col.index = ['skewness']
            kurtosis_col = pd.DataFrame(kurtosis).T
            kurtosis_col.index = ['kurtosis']
            df_desc = pd.concat([df_desc.T, median_col, cv_col, skew_col, kurtosis_col]).T
            df_desc.columns = ['count', 'mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max',
                               'median', 'cv%', 'skewness', 'kurtosis']
            print(f'\n{self.line}')
            print(' Descriptive Analysis '.center(170))
            print(self.line)
            print(df_desc)
            print(self.line)
        else:
            raise ValueError('! num_cols is empty')
        

    def check_num(self, dataframe, num_cols, alpha=0.05, plot=False, width_for_graph=15, height_for_graph=5):
        """
        Runs normality diagnostics for each numerical column and optionally
        displays visual plots.

        For each column the method:
        - Optionally renders a Q-Q plot, histogram, and box plot side-by-side.
        - Applies Shapiro-Wilk test for n ≤ 2500, or D'Agostino K² test for n > 2500.
        - Prints the test statistic, p-value, and a plain-language conclusion.

        The statistical result should always be cross-checked against the visual plots
        before a final normality decision is made. Use the returned list together with
        num_summary() to record overrides.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to analyze.
        num_cols : list of str
            List of numerical column names to test.
        alpha : float, optional
            Significance level for the normality test (default: 0.05).
        plot : bool, optional
            If True, displays Q-Q plot, histogram, and box plot for each column
            (default: False).
        width_for_graph : int, optional
            Width of each figure in inches (default: 15).
        height_for_graph : int, optional
            Height of each figure in inches (default: 5).

        Returns
        -------
        list of str
            Column names that the statistical test flagged as non-normal.
            Columns that passed the test are not included, even if visually suspect.

        Raises
        ------
        ValueError
            If num_cols is empty.

        Example
        -------
        non_normals = eda.check_num(plot=True)
        eda.num_summary(result_dict={'age': 'Normal', 'salary': 'Non-normal'})
        """

        if num_cols:
            print(f'\n{self.line}')
            print(' Numerical Variable Summary '.center(170))
            print(self.line)
            result = []
            for col in num_cols:
                data = dataframe[col].dropna()

                if plot:
                    fig, axes = plt.subplots(1, 3, figsize=(width_for_graph, height_for_graph))
                    fig.patch.set_facecolor('white')
                    for ax in axes:
                        ax.set_facecolor('white')
                        ax.grid(False)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#cccccc')
                    fig.suptitle(col, fontsize=14, fontweight='bold')

                    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(data)
                    axes[0].scatter(osm, osr, color='#4682B4', s=15, alpha=0.7)
                    axes[0].plot(osm, slope * np.array(osm) + intercept, color='red', linewidth=2)
                    axes[0].set_title('Q-Q Plot')
                    axes[0].set_xlabel('Theoretical Quantiles')
                    axes[0].set_ylabel(f'{col} (Sample Quantiles)')

                    axes[1].hist(data, bins='auto', color='#4682B4', alpha=0.7, edgecolor='white')
                    axes[1].set_title('Histogram')
                    axes[1].set_xlabel(col)
                    axes[1].set_ylabel('Frequency')

                    axes[2].boxplot(data, vert=False, patch_artist=True,
                                    boxprops=dict(facecolor='#4682B4', alpha=0.7),
                                    medianprops=dict(color='red', linewidth=2),
                                    whiskerprops=dict(color='#4682B4'),
                                    capprops=dict(color='#4682B4'),
                                    flierprops=dict(marker='o', color='#4682B4', alpha=0.5))
                    axes[2].set_title('Box Plot')
                    axes[2].set_xlabel(col)
                    axes[2].set_yticks([])

                    plt.tight_layout()
                    plt.show()

                n = len(data)
                if n <= 2500:
                    test_stat, p_value = scipy.stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                else:
                    test_stat, p_value = scipy.stats.normaltest(data)
                    test_name = "D'Agostino K²"
                print(f"Column: {col}")
                print(f"Test: {test_name}")
                print(f"Test Statistic: {test_stat:.4f}, p-value: {p_value:.4f}")
                if p_value > alpha:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} > {alpha}), "
                          f"\nthe sample appears Gaussian. However, please verify using the visuals above before making a final decision. "
                          f"\nTo override, pass result_dict={{'col_name': 'Normal/Non-normal'}} to num_summary().")
                else:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} ≤ {alpha}), "
                          f"\nthe sample does not appear Gaussian. However, please verify using the visuals above before making a final decision. "
                          f"\nTo override, pass result_dict={{'col_name': 'Normal/Non-normal'}} to num_summary().")
                    result.append(col)
                print(self.line)
            print(f"\nNote: The results above are based on statistical tests only. "
                  f"\nPlease verify using the visuals before making a final decision. "
                  f"\nTo manually set normality, pass result_dict={{'col_name': 'Normal/Non-normal'}} "
                  f"\nto num_summary().")
            print(self.line)
            if result:
                print(f"\nColumns flagged as Non-normal by the test — please verify visually: {result}")
            else:
                print(f"\nAll columns appear Gaussian according to the test.")
            return result
        else:
            raise ValueError('! num_cols is empty')
        

    def num_summary(self,num_cols, result_dict):
        """
         Builds the normality summary DataFrame that downstream methods rely on
        (outlier detection, target correlation method selection).

        Columns not present in result_dict are assumed to be normally distributed.
        Always run check_num() first to identify non-normal columns before
        constructing result_dict.

        Parameters
        ----------
        num_cols : list of str
            Full list of numerical column names in the dataset.
        result_dict : dict
            Mapping of {column_name: 'Normal' | 'Non-normal'} for columns whose
            normality you wish to specify explicitly. Any column not listed here
            defaults to 'Normal'.

        Returns
        -------
        pd.DataFrame
            A two-column DataFrame with columns ['Column', 'Result'], where Result
            is either 'Normal' or 'Non-normal' for each entry in num_cols.

        Example
        -------
        eda.check_num()
        eda.num_summary(result_dict={
            'age'    : 'Normal',
            'salary' : 'Non-normal',
            'height' : 'Non-normal'
        })
         """
        result_df = pd.DataFrame({
            'Column': num_cols,
            'Result': [result_dict.get(col, 'Normal') for col in num_cols]
        })
        return  result_df 