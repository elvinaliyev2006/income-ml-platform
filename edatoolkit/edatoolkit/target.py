import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .helpers import get_groups, select_group_test ,calculate_advanced_effect_size


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class TargetAnalyzer:

    def __init__(self):

        self.line = '─' * 170

    def target_summary_with_cat(self, dataframe, cat_cols, num_cols, target_col, alpha=0.05, plot=False, width_for_graph=13,
                                height_for_graph=5):
        """
        Analyzes the relationship between the target column and each categorical column.

        Behavior depends on the target column type:

        Target is categorical:
            - Computes a crosstab with row-percentage normalization.
            - Runs a Chi-Square test of independence.
            - Calculates Cramer's V to quantify association strength.
            - Optionally displays a grouped bar chart and a percentage heatmap.

        Target is numerical:
            - Computes mean, median, and count of the target per category.
            - Assesses within-group normality using Shapiro-Wilk (n ≤ 2500)
            or D'Agostino K² (n > 2500).
            - Selects the appropriate significance test automatically:
                2 groups, normal  → Welch's t-test
                2 groups, non-normal → Mann-Whitney U
                3+ groups, normal + equal variance → One-way ANOVA
                3+ groups, otherwise → Kruskal-Wallis
            - Optionally displays a box plot + overlapping histogram per categorical column.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to analyze.
        cat_cols : list of str
            Categorical column names to compare against the target.
        num_cols : list of str
            Numerical column names (used to determine target column type).
        target_col : str
            Name of the target column.
        alpha : float, optional
            Significance level for all hypothesis tests (default: 0.05).
        plot : bool, optional
            If True, displays charts for each categorical column (default: False).
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
            print(' Target Analysis with Categorical Variables '.center(170))
            print(self.line)
            if target_col in cat_cols:
                cols_to_analyze = [col for col in cat_cols if col != target_col]
                teal_palette = ['#355c7d', '#43aa8b', '#c77dff', '#f67280', '#f8961e', '#ef476f', '#00b4d8', '#9b5de5']
                for col in cols_to_analyze:
                    ct = pd.crosstab(dataframe[target_col].astype(str), dataframe[col])
                    ct_pct = pd.crosstab(dataframe[target_col].astype(str), dataframe[col],
                                         normalize='index') * 100
                    chi2, p, dof, expected = scipy.stats.chi2_contingency(ct)

                    if plot:
                        fig, axes = plt.subplots(1, 2, figsize=(width_for_graph, height_for_graph))
                        fig.patch.set_facecolor('white')
                        for ax in axes:
                            ax.set_facecolor('white')
                            ax.grid(False)
                            for spine in ax.spines.values():
                                spine.set_edgecolor('#cccccc')
                        fig.suptitle(f'Relationship: {target_col} vs {col}', fontsize=13, fontweight='bold')

                        target_cats = dataframe[target_col].unique()
                        col_cats = dataframe[col].unique()
                        x = np.arange(len(target_cats))
                        width = 0.8 / len(col_cats)
                        for i, cat in enumerate(col_cats):
                            counts = [dataframe[
                                          (dataframe[target_col] == t) & (dataframe[col] == cat)].shape[
                                          0]
                                      for t in target_cats]
                            axes[0].bar(x + i * width, counts, width=width,
                                        label=str(cat), color=teal_palette[i % len(teal_palette)], alpha=0.85)
                        axes[0].set_xticks(x + width * (len(col_cats) - 1) / 2)
                        axes[0].set_xticklabels([str(t) for t in target_cats], rotation=45, ha='right')
                        axes[0].set_title('Countplot')
                        axes[0].set_xlabel(target_col)
                        axes[0].set_ylabel('Count')
                        axes[0].legend(title=col)

                        sns.heatmap(ct_pct.round(1), annot=True, fmt='.1f', cmap='YlGnBu',
                                    vmin=0, vmax=100, ax=axes[1],
                                    linewidths=0.5, linecolor='white',
                                    annot_kws={'size': 10})
                        axes[1].set_title('Crosstab Heatmap Percentage')
                        axes[1].set_xlabel(col)
                        axes[1].set_ylabel(target_col)

                        plt.tight_layout()
                        plt.show()

                    print(f"\n{self.line}")
                    print(' Chi-Square Test '.center(170))
                    print(self.line)
                    print(f"χ²      = {chi2:.4f}")
                    print(f"p-value = {p:.6f}")
                    print(f"df      = {dof}")
                    print(f"\nExpected Values:")
                    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(1)
                    print(expected_df, '\n')
                    print('Result:')
                    if p < alpha:
                        print(
                            f"\n→ H₀ REJECTED: There is a significant relationship between {target_col} and {col} (p < 0.05)")
                    else:
                        print("\n→ H₀ ACCEPTED: No statistically significant difference found")
                    print(f"\n{self.line}")
                    print(" Cramer's V ".center(170))
                    print(self.line)
                    n = ct.values.sum()
                    min_dim = min(ct.shape) - 1
                    if min_dim > 0:
                        cramers_v = np.sqrt(chi2 / (n * min_dim))
                    else:
                        cramers_v = 0
                    print(f"V = {cramers_v:.4f}")
                    print('Result:')
                    if cramers_v < 0.1:
                        strength = "Very weak"
                    elif cramers_v < 0.3:
                        strength = "Weak-moderate"
                    elif cramers_v < 0.5:
                        strength = "Moderate-strong"
                    else:
                        strength = "Strong"
                    print(f"\nAssociation strength: {strength}\n")
                    print(self.line)

            elif target_col in num_cols:
                for col in cat_cols:
                    df_pivot = dataframe.pivot_table(index=col, values=target_col,
                                                          aggfunc=['mean', 'median', 'count'], observed=False)
                    print(df_pivot)
                    groups, normality_pvals = get_groups(dataframe, col, target_col)
                    if len(groups) < 2:
                        print(f"Skipping {col}: Not enough groups for comparison.")
                        continue

                    if plot:
                        teal_palette = ['#355c7d', '#43aa8b', '#c77dff', '#f67280', '#f8961e', '#ef476f', '#00b4d8',
                                        '#9b5de5']
                        categories = dataframe[col].dropna().unique()
                        palette = {cat: teal_palette[i % len(teal_palette)] for i, cat in enumerate(categories)}

                        fig, (ax_box, ax_hist) = plt.subplots(
                            1, 2,
                            figsize=(width_for_graph * 1.4, height_for_graph),
                            gridspec_kw={'width_ratios': [1, 1]}
                        )
                        fig.patch.set_facecolor('white')
                        for ax in (ax_box, ax_hist):
                            ax.set_facecolor('white')
                            ax.grid(False)
                            for spine in ax.spines.values():
                                spine.set_edgecolor('#cccccc')

                        sns.boxplot(
                            data=dataframe, x=col, y=target_col,
                            palette=palette, ax=ax_box, hue=col,
                            boxprops=dict(alpha=0.85),
                            medianprops=dict(color='red', linewidth=2)
                        )
                        ax_box.set_title(f'{target_col} by {col}', fontsize=12, fontweight='bold')
                        ax_box.set_xlabel(col)
                        ax_box.set_ylabel(target_col)
                        plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right')
                        bin_count = min(40, max(10, int(np.sqrt(len(dataframe)))))
                        all_vals = dataframe[target_col].dropna()
                        bins = np.linspace(all_vals.min(), all_vals.max(), bin_count + 1)

                        for i, cat in enumerate(categories):
                            vals = dataframe.loc[dataframe[col] == cat, target_col].dropna()
                            ax_hist.hist(
                                vals, bins=bins,
                                color=teal_palette[i % len(teal_palette)],
                                alpha=0.45, label=str(cat),
                                edgecolor='white', linewidth=0.4
                            )
                           
                            if len(vals) > 5:
                                kde_x = np.linspace(all_vals.min(), all_vals.max(), 300)
                                kde = scipy.stats.gaussian_kde(vals, bw_method='scott')
                                ax_hist_twin = ax_hist.twinx()
                                ax_hist_twin.plot(
                                    kde_x, kde(kde_x),
                                    color=teal_palette[i % len(teal_palette)],
                                    linewidth=2, alpha=0.9
                                )
                                ax_hist_twin.set_yticks([])
                                ax_hist_twin.set_facecolor('white')

                        ax_hist.set_title(f'Distribution of {target_col} by {col}', fontsize=12, fontweight='bold')
                        ax_hist.set_xlabel(target_col)
                        ax_hist.set_ylabel('Count')
                        ax_hist.legend(title=col, framealpha=0.7)

                        fig.suptitle(f'{target_col}  ×  {col}', fontsize=13, fontweight='bold', y=1.01)
                        plt.tight_layout()
                        plt.show()

                    print(f"\n{self.line}")
                    stat,p_value,test_name = select_group_test(groups, normality_pvals, alpha)
                    if p_value < alpha:
                        print(f"P-value: {p_value:.6f}.\n H₀ REJECTED: Significant difference found.")
                        calculate_advanced_effect_size(test_name,stat,groups)
                    else:
                        print(f"P-value: {p_value:.6f}.\n H₀ ACCEPTED: No significant difference.")
                    print(self.line)
        else:
            raise ValueError('! cat_cols is empty')


    def target_summary_with_num(self,dataframe, num_cols, cat_cols, target_col, num_summary_df, alpha=0.05, plot=False, width_for_graph=13, height_for_graph=5):
        """
        Analyzes the relationship between the target column and each numerical column.

        Behavior depends on the target column type:

        Target is categorical:
            - Computes mean, median, and count of each numerical column per target class.
            - Assesses per-group normality using Shapiro-Wilk (n ≤ 2500)
            or D'Agostino K² (n > 2500).
            - Selects the appropriate significance test automatically (same logic as
            target_summary_with_cat for the numerical-target case).
            - Optionally displays a box plot + overlapping histogram for each numerical column.

        Target is numerical:
            - Requires num_summary_df (run num_summary() first).
            - Selects Pearson correlation if both columns are normal,
            otherwise uses Spearman correlation.
            - Reports the correlation coefficient (ρ), p-value, and a
            qualitative strength label (Negligible / Weak / Moderate /
            Strong / Very strong) with direction (positive / negative).
            - Optionally displays a scatter plot with a fitted regression line.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset to analyze.
        num_cols : list of str
            Numerical column names to compare against the target.
        cat_cols : list of str
            Categorical column names (used to determine target column type).
        target_col : str
            Name of the target column.
        num_summary_df : pd.DataFrame or None
            Normality summary from num_summary(). Required when the target is
            numerical; pass None otherwise.
        alpha : float, optional
            Significance level for all hypothesis tests (default: 0.05).
        plot : bool, optional
            If True, displays charts for each numerical column (default: False).
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
            If num_cols is empty.
        RuntimeError
            If target is numerical and num_summary_df is None.
        """
        if num_cols:
            print(f'\n{self.line}')
            print(' Target Analysis with Numerical Variables '.center(170))
            print(self.line)
            if target_col in cat_cols:
                for col in num_cols:
                    df_pivot = dataframe.pivot_table(index=target_col, values=col,
                                                          aggfunc=['mean', 'median', 'count'], observed=False)
                    print(df_pivot)
                    groups, normality_pvals = get_groups(dataframe, target_col, col)
                    if len(groups) < 2:
                        print(f"Skipping {col}: Not enough groups for comparison.")
                        continue

                    if plot:
                        teal_palette = ['#355c7d', '#43aa8b', '#c77dff', '#f67280', '#f8961e', '#ef476f', '#00b4d8',
                                        '#9b5de5']
                        categories = dataframe[target_col].dropna().unique()
                        palette = {cat: teal_palette[i % len(teal_palette)] for i, cat in enumerate(categories)}

                        fig, (ax_box, ax_hist) = plt.subplots(
                            1, 2,
                            figsize=(width_for_graph * 1.4, height_for_graph),
                            gridspec_kw={'width_ratios': [1, 1]}
                        )
                        fig.patch.set_facecolor('white')
                        for ax in (ax_box, ax_hist):
                            ax.set_facecolor('white')
                            ax.grid(False)
                            for spine in ax.spines.values():
                                spine.set_edgecolor('#cccccc')

                        sns.boxplot(
                            data=dataframe, x=target_col, y=col,
                            palette=palette, ax=ax_box, hue=target_col,
                            boxprops=dict(alpha=0.85),
                            medianprops=dict(color='red', linewidth=2)
                        )
                        ax_box.set_title(f'{col} by {target_col}', fontsize=12, fontweight='bold')
                        ax_box.set_xlabel(target_col)
                        ax_box.set_ylabel(col)
                        plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right')
                        bin_count = min(40, max(10, int(np.sqrt(len(dataframe)))))
                        all_vals = dataframe[col].dropna()
                        bins = np.linspace(all_vals.min(), all_vals.max(), bin_count + 1)

                        for i, cat in enumerate(categories):
                            vals = dataframe.loc[dataframe[target_col] == cat, col].dropna()
                            ax_hist.hist(
                                vals, bins=bins,
                                color=teal_palette[i % len(teal_palette)],
                                alpha=0.45, label=str(cat),
                                edgecolor='white', linewidth=0.4
                            )
                            if len(vals) > 5:
                                kde_x = np.linspace(all_vals.min(), all_vals.max(), 300)
                                kde = scipy.stats.gaussian_kde(vals, bw_method='scott')
                                ax_hist_twin = ax_hist.twinx()
                                ax_hist_twin.plot(
                                    kde_x, kde(kde_x),
                                    color=teal_palette[i % len(teal_palette)],
                                    linewidth=2, alpha=0.9
                                )
                                ax_hist_twin.set_yticks([])
                                ax_hist_twin.set_facecolor('white')

                        ax_hist.set_title(f'Distribution of {col} by {target_col}', fontsize=12, fontweight='bold')
                        ax_hist.set_xlabel(col)
                        ax_hist.set_ylabel('Count')
                        ax_hist.legend(title=target_col, framealpha=0.7)

                        fig.suptitle(f'{col}  ×  {target_col}', fontsize=13, fontweight='bold', y=1.01)
                        plt.tight_layout()
                        plt.show()

                    print(f"\n{self.line}")
                    stat, p_value, test_name = select_group_test(groups, normality_pvals, alpha)
                    if p_value < alpha:
                        print(f"P-value: {p_value:.6f}.\n H₀ REJECTED: Significant difference found.")
                        calculate_advanced_effect_size(test_name,stat,groups)
                    else:
                        print(f"P-value: {p_value:.6f}.\n H₀ ACCEPTED: No significant difference.")
                    print(self.line)

            elif target_col in num_cols:
                if num_summary_df is None:
                    raise RuntimeError("Run num_summary() first.")
                cols_to_analyze = [col for col in num_cols if col != target_col]
                for col in cols_to_analyze:
                    is_normal_target = \
                        num_summary_df[num_summary_df['Column'] == target_col]['Result'].values[
                            0] == 'Normal'
                    is_normal_col = num_summary_df[num_summary_df['Column'] == col]['Result'].values[
                                        0] == 'Normal'
                    if is_normal_target and is_normal_col:
                        method = 'Pearson'
                        corr_value, p_value = scipy.stats.pearsonr(dataframe[target_col], dataframe[col])
                    else:
                        method = 'Spearman'
                        corr_value, p_value = scipy.stats.spearmanr(dataframe[target_col],
                                                                    dataframe[col])
                    print(f'\nTarget: {target_col} ←→ {col}')
                    print(f'Method: {method} Correlation')
                    print(f'ρ: {corr_value}')
                    print(f'p-value: {p_value}')
                    if p_value < alpha:
                        print(f"→ H₀ REJECTED: Significant correlation (p < {alpha})")
                    else:
                        print(f"→ H₀ ACCEPTED: No significant correlation")
                    if abs(corr_value) < 0.1:
                        strength = "Negligible"
                    elif abs(corr_value) < 0.3:
                        strength = "Weak"
                    elif abs(corr_value) < 0.5:
                        strength = "Moderate"
                    elif abs(corr_value) < 0.7:
                        strength = "Strong"
                    else:
                        strength = "Very strong"
                    direction = 'positive' if corr_value > 0 else 'negative'
                    print(f'Strength: {strength} {direction} correlation\n')

                    if plot:
                        fig, ax = plt.subplots(figsize=(width_for_graph, height_for_graph))
                        fig.patch.set_facecolor('white')
                        ax.set_facecolor('white')
                        ax.grid(False)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#cccccc')
                        ax.scatter(dataframe[target_col], dataframe[col],
                                   color='#4682B4', alpha=0.5, s=20)
                        m, b = np.polyfit(dataframe[target_col], dataframe[col], 1)
                        x_line = np.linspace(dataframe[target_col].min(),
                                             dataframe[target_col].max(), 200)
                        ax.plot(x_line, m * x_line + b, color='red', linewidth=3)
                        ax.set_title(f'{target_col} vs {col} | Method: {method} ρ={corr_value:.3f}',
                                     fontsize=13, fontweight='bold')
                        ax.set_xlabel(target_col)
                        ax.set_ylabel(col)
                        plt.tight_layout()
                        plt.show()

                    print(self.line)
        else:
            raise ValueError('! num_cols is empty ')