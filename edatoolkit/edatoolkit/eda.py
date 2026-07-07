from .inspector import Inspector
from .normality import NormalityAnalyzer
from .outliers import OutlierAnalyzer
from .categorical import CategoricalAnalyzer
from .target import TargetAnalyzer
from .correlation import Correlation


class EDA:
    """
    Initializes the EDA orchestrator with the dataset and analysis configuration.
    
    Automatically classifies all columns into categorical, numerical,
    num_but_cat, and cat_but_car groups upon construction.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to analyze.
    target_col : str
        Name of the target (dependent) variable.
    cat_th : int, optional
        Unique-value threshold for treating a numeric column as categorical
        (default: 10).
    car_th : int, optional
        Cardinality threshold above which an object column is treated as
        high-cardinality (default: 20).
    alpha : float, optional
        Global significance level used across all hypothesis tests (default: 0.05).      
    """
    def __init__(self, dataframe, target_col, cat_th=10, car_th=20, alpha=0.05):
        self.dataframe=dataframe
        self.target_col=target_col
        self.car_th=car_th
        self.cat_th=cat_th
        self.alpha=alpha
        self.inspector=Inspector()
        self.normality=NormalityAnalyzer()
        self.outliers=OutlierAnalyzer()
        self.categorical=CategoricalAnalyzer()
        self.target=TargetAnalyzer()
        self.correlation=Correlation()
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.inspector.get_columns_types(
                                                                                        dataframe=self.dataframe, car_th=self.car_th, cat_th=self.cat_th)
        self.num_summary_df = None
        self.outlier_report = None
    
    def update_dataframe(self,new_df):
        """
        Replaces the active dataframe and recalculates all column type classifications.

        Use this after any external transformation (e.g., encoding, scaling, merging)
        to keep the EDA instance in sync with the current data state.
        Resets num_summary_df to None, requiring num_summary() to be re-run.

        Parameters
        ----------
        new_df : pd.DataFrame
            The updated dataset to assign to the instance.

        Returns
        -------
        None
            Prints a confirmation message to stdout.
        """
        self.dataframe=new_df
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.inspector.get_columns_types(dataframe=self.dataframe, 
                                                                                                            car_th=self.car_th, cat_th=self.cat_th)
        self.num_summary_df = None
        print("Dataframe and column types have been successfully updated.")  

    def check_dataframe(self,n=5):
        """
         Delegates to Inspector.check_dataframe().
        Prints head, sample, tail, shape, dtypes, missing-value ratios,
        and duplicate row count for the active dataframe.

        Parameters
        ----------
        n : int, optional
            Number of rows to display for head, sample, and tail (default: 5). 
        """
        self.inspector.check_dataframe(dataframe=self.dataframe, n=n)
    
    def get_columns_types(self):
        """
        Delegates to Inspector.get_columns_types().
        Returns the current column-type classification for the active dataframe.

        Returns
        -------
        tuple : (cat_cols, num_cols, num_but_cat, cat_but_car)
        """
        return self.inspector.get_columns_types(dataframe=self.dataframe,car_th=self.car_th,cat_th=self.cat_th)

    def descriptive_analysis(self):
        """
        Delegates to NormalityAnalyzer.descriptive_analysis().
        Prints extended descriptive statistics for all numerical columns
        (percentiles, median, CV%, skewness, kurtosis).
        """
        self.normality.descriptive_analysis(dataframe=self.dataframe,num_cols=self.num_cols)
    
    def check_num(self, plot=False, width_for_graph=15, height_for_graph=5):
        """
        Delegates to NormalityAnalyzer.check_num().
        Runs normality tests and optionally plots Q-Q, histogram, and box plots
        for all numerical columns.

        Parameters
        ----------
        plot : bool, optional
            If True, displays diagnostic plots (default: False).
        width_for_graph : int, optional
            Figure width in inches (default: 15).
        height_for_graph : int, optional
            Figure height in inches (default: 5).

        Returns
        -------
        list of str
            Columns flagged as non-normal by the statistical test.
        """
        return self.normality.check_num(dataframe=self.dataframe,num_cols=self.num_cols,alpha=self.alpha,plot=plot,width_for_graph=width_for_graph,
                                 height_for_graph=height_for_graph)
    
    def num_summary(self,result_dict):
        """
        Delegates to NormalityAnalyzer.num_summary() and stores the result
        internally as self.num_summary_df for use by outlier and target methods.

        Parameters
        ----------
        result_dict : dict
            {column_name: 'Normal' | 'Non-normal'} overrides. Columns not listed
            default to 'Normal'.

        Returns
        -------
        pd.DataFrame
            Normality summary with columns ['Column', 'Result'].
        """
        self.num_summary_df=self.normality.num_summary(num_cols=self.num_cols,result_dict=result_dict)
        return self.num_summary_df
    
    def check_outlier(self, q1_th=0.25, q3_th=0.75, iqr_th=1.5, z_score_th=3, remove=False, cap=False):
        """
        Delegates to OutlierAnalyzer.check_outlier().
        Detects outliers using Z-score (normal columns) or IQR (non-normal columns).
        If remove=True or cap=True, updates self.dataframe in place and
        resets self.num_summary_df to None.

        Parameters
        ----------
        q1_th : float, optional
            Quantile used as the first quartile (Q1) when applying the IQR method.
            Must be between 0 and 1 (default: 0.25).
        q3_th : float, optional
            Quantile used as the third quartile (Q3) when applying the IQR method.
            Must be between 0 and 1 (default: 0.75).
        iqr_th : float, optional
            IQR multiplier (default: 1.5).
        z_score_th : int, optional
            Z-score boundary in standard deviations (default: 3).
        remove : bool, optional
            Drop rows containing outliers (default: False).
        cap : bool, optional
            Clip outlier values to boundary limits (default: False).

        Returns
        -------
        dict or tuple
            Outlier report, or (outlier_report, cleaned_df) if remove/cap is True.
        """
        if self.num_summary_df is None:
            raise RuntimeError("Run num_summary() first.")
        result=self.outliers.check_outlier(dataframe=self.dataframe,num_cols=self.num_cols,num_summary_df=self.num_summary_df,
                                           q1_th=q1_th, q3_th=q3_th,iqr_th=iqr_th,z_score_th=z_score_th,remove=remove,cap=cap)
        if isinstance(result, tuple):
            outlier_report, self.dataframe = result
            self.num_summary_df = None
            return outlier_report, self.dataframe
        return result
    
    def cat_summary(self,plot=False,width_for_graph=13, height_for_graph=5 ):
        """
        Delegates to CategoricalAnalyzer.cat_summary().
        Prints value counts and ratios for all categorical columns, with
        optional bar charts.

        Parameters
        ----------
        plot : bool, optional
            Display bar chart per column (default: False).
        width_for_graph : int, optional
            Figure width in inches (default: 13).
        height_for_graph : int, optional
            Figure height in inches (default: 5).
        """
        self.categorical.cat_summary(dataframe=self.dataframe,cat_cols=self.cat_cols,plot=plot,
                                     width_for_graph=width_for_graph,height_for_graph=height_for_graph)

    def target_summary_with_cat(self, plot=False, width_for_graph=13, height_for_graph=5):
        """
        Delegates to TargetAnalyzer.target_summary_with_cat().
        Analyzes the relationship between the target and each categorical column
        using Chi-Square + Cramer's V (categorical target) or group comparison
        tests (numerical target).

        Parameters
        ----------
        plot : bool, optional
            Display charts per column (default: False).
        width_for_graph : int, optional
            Figure width in inches (default: 13).
        height_for_graph : int, optional
            Figure height in inches (default: 5).
        """
        self.target.target_summary_with_cat(dataframe=self.dataframe,cat_cols=self.cat_cols,num_cols=self.num_cols,
                                            target_col=self.target_col,alpha=self.alpha,plot=plot,width_for_graph=width_for_graph,height_for_graph=height_for_graph)
        
    def target_summary_with_num(self, plot=False, width_for_graph=13, height_for_graph=5):
        """
        Delegates to TargetAnalyzer.target_summary_with_num().
        Analyzes the relationship between the target and each numerical column
        using group comparison tests (categorical target) or Pearson/Spearman
        correlation (numerical target).

        Parameters
        ----------
        plot : bool, optional
            Display charts per column (default: False).
        width_for_graph : int, optional
            Figure width in inches (default: 13).
        height_for_graph : int, optional
            Figure height in inches (default: 5).
        """
        self.target.target_summary_with_num(dataframe=self.dataframe,num_cols=self.num_cols,cat_cols=self.cat_cols,
                                            target_col=self.target_col,num_summary_df=self.num_summary_df,alpha=self.alpha,plot=plot,
                                            width_for_graph=width_for_graph,height_for_graph=height_for_graph)

    def correlation_heatmap(self, method="spearman",  width_for_graph=9, height_for_graph=9):
        """
        Delegates to Correlation.correlation_heatmap().
        Displays an annotated correlation heatmap for all numerical columns.

        Parameters
        ----------
        method : str, optional
            Correlation method: 'spearman', 'pearson', or 'kendall' (default: 'spearman').
        width_for_graph : int, optional
            Figure width in inches (default: 9).
        height_for_graph : int, optional
            Figure height in inches (default: 9).
        """
        self.correlation.correlation_heatmap(dataframe=self.dataframe,num_cols=self.num_cols,method=method,
                                            width_for_graph=width_for_graph,height_for_graph=height_for_graph )
