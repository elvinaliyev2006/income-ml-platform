import scipy
import numpy as np

def get_groups(dataframe, group_col, value_col):
    """
    Splits a dataframe column into per-group arrays and tests each group for normality.

    Shapiro-Wilk is used for groups with 2500 or fewer observations;
    D'Agostino K² is used for larger ones.

    Parameters
    ----------
    dataframe : pd.DataFrame
    group_col : str
        Column that defines the groups (e.g. a categorical variable).
    value_col : str
        Numeric column whose values are being compared across groups.

    Returns
    -------
    groups : list of pd.Series
        One array per unique value in group_col.
    normality_pvals : np.ndarray
        Normality test p-value for each group, in the same order.``
    """
    groups = []
    normality_pvals = []
    for val in dataframe[group_col].unique():
        g = dataframe[dataframe[group_col] == val][value_col]
        groups.append(g)
        _, p = (scipy.stats.shapiro(g) if len(g) <= 2500 
                else scipy.stats.normaltest(g))
        normality_pvals.append(p)
    return groups, np.array(normality_pvals)

def select_group_test(groups, normality_pvals, alpha):
    """
    Picks and runs the right significance test based on group count and normality.

    Decision logic:
        2 groups, normal + n > 30  → Welch's t-test
        2 groups, otherwise        → Mann-Whitney U
        3+ groups, normal + n > 30 + equal variance → One-way ANOVA
        3+ groups, otherwise       → Kruskal-Wallis

    Parameters
    ----------
    groups : list of array-like
        Per-group data arrays, typically from get_groups().
    normality_pvals : np.ndarray
        Normality p-values for each group, also from get_groups().
    alpha : float
        Significance level used for both the normality check and Levene's test.

    Returns
    -------
    stat : float
        Test statistic.
    p : float
        p-value.
    test_name : str
        One of 'ttest', 'mannwhitneyu', 'anova', 'kruskal'.
    """
    is_normal = bool((normality_pvals > alpha).all())
    n = len(groups)
    
    if n == 2:
        if is_normal and min(len(g) for g in groups) > 30:
            test_name = 'ttest'
            stat, p = scipy.stats.ttest_ind(*groups, equal_var=False)
        else:
            test_name = 'mannwhitneyu'
            stat, p = scipy.stats.mannwhitneyu(*groups, alternative='two-sided')
    elif n >= 3:
        if is_normal and min(len(g) for g in groups) > 30:
            _, p_levene = scipy.stats.levene(*groups)
            if p_levene > 0.05:
                test_name='anova'
                stat, p= scipy.stats.f_oneway(*groups)
            else:
                test_name='kruskal'
                stat,p= scipy.stats.kruskal(*groups)
        else:
            test_name='kruskal'
            stat,p= scipy.stats.kruskal(*groups)
    return stat, p , test_name

def calculate_advanced_effect_size(test_name,stat,groups_data):
    """
    Computes and prints an effect size metric for a completed group comparison test.

    Each test maps to its appropriate metric:
        t-test / Mann-Whitney U → r (correlation-based effect size)
        ANOVA                   → eta-squared (η²)
        Kruskal-Wallis          → epsilon-squared (ε²)

    Magnitude thresholds follow Cohen's conventions:
        r:               small < 0.3, medium < 0.5, large ≥ 0.5
        η² and ε²:       small < 0.06, medium < 0.14, large ≥ 0.14

    Parameters
    ----------
    test_name : str
        One of 'ttest', 'mannwhitneyu', 'anova', 'kruskal'.
    stat : float
        The test statistic returned by select_group_test().
    groups_data : list of array-like
        The same group arrays passed to select_group_test().

    Returns
    -------
    None
        Prints a single interpretive sentence to stdout.
    """
    flattened=np.concatenate(groups_data)
    N=len(flattened)
    k=len(groups_data)

    if test_name == 'mannwhitneyu':
        metric="r"
        n1,n2 = len(groups_data[0]), len(groups_data[1])
        mu_u= (n1*n2)/2
        sigma_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        z_stat = abs((stat - mu_u) / sigma_u)
        value = z_stat / np.sqrt(N)
        result = "Small" if value < 0.3 else "Medium" if value < 0.5 else "Large"

    elif test_name == "ttest":
        metric="r"
        df = N - 2
        value = np.sqrt(stat**2 / (stat**2 + df))
        result = "Small" if value < 0.3 else "Medium" if value < 0.5 else "Large"
        
    elif test_name == 'anova':
        metric="eta_squared"
        df_between = k - 1
        df_within = N - k
        value = (stat * df_between) / (stat * df_between + df_within)
        result = "Small" if value < 0.06 else "Medium" if value < 0.14 else "Large"

    elif test_name =='kruskal':
        metric="epsilon_squared"
        value = (stat * (N + 1)) / (N**2 - 1)
        result='Small' if value<0.06 else 'Medium' if value< 0.14 else 'Large'
    
    if result == "Small":
        report = f"Effect size {metric} = {round(value, 4)}/{result} — the difference is statistically significant but has minimal practical importance."
    elif result == "Medium":
        report = f"Effect size {metric} = {round(value, 4)}/{result} — the difference is both statistically significant and practically meaningful."
    elif result == "Large":
        report = f"Effect size {metric} = {round(value, 4)}/{result} — the difference is statistically significant and has substantial real-world importance."

    print(report)

