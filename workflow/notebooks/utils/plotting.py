"""
utils for jupyter notebook plotting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce


def fix_count_names(df):
    """
    transform into FL-B format
    @param df:
    @return:
    """
    df['NAME'] = [x.replace(".label", "") for x in df['NAME']]
    df['NAME'] = [x.replace("res/label_locs/", "") for x in df['NAME']]
    df['NAME'] = [x.replace(".vote2", "") for x in df['NAME']]
    df['NAME'] = [x.replace("1U_Epo", "1unitEpo") for x in df['NAME']]
    df['NAME'] = [x.replace("U_Epo", "unitsEpo") for x in df['NAME']]
    return df

def fix_count_name(name):
    """
    transform into FL-B format
    @param name: 1U_Epo_2-Stitching-02.1
    @return: 1unitEpo_2-Stitching-02.1
    """
    name = name.replace(".label", "")
    name = name.replace("res/label_locs/", "")
    name = name.replace("1U_Epo", "1unitEpo")
    name = name.replace("U_Epo", "unitsEpo")
    return name


def get_epo_concentration(name):
    """
    designed for FL-B
    @param name:
    @return:
    """
    if '1unitEpo' in name:
        return 1.0
    elif 'point5unitsEpo' in name:
        return 0.5
    elif 'point25unitsEpo' in name:
        return 0.25
    elif 'point125unitsEpo' in name:
        return 0.125
    elif 'point0625unitsEpo' in name:
        return 0.0625
    elif 'NoEpo' in name:
        return 0.0
    else:
        print(name, 'has no known Epo concentration')
        return None  # If no concentration found


def get_replicate(label):
    for key, value in {'Epo_1': '1', 'Epo_2': '2', 'Epo_3': '3', 'Epo_4': '4',
                       'Epo1': '1', 'Epo2': '2', 'Epo3': '3', 'Epo4': '4'}.items():
        if key in label:  # allow partial match
            return value
    return None


def get_sceneIndex(name):
    if '.0' in name:
        return '0'
    elif '.1' in name:
        return '1'
    elif '.2' in name:
        return '2'
    elif '.3' in name:
        return '3'


def create_pairplot(df, bins=20):
    """
    all df columns should be counts, no NAME column

    print(m2.head(2))
       COUNT-A  COUNT-J  COUNT-L
    0       23       21       29
    1        6        2        4
    """
    global_max = df.max().max() * 1.1
    g = sns.pairplot(df, diag_kind="hist", diag_kws={"bins": bins}, corner=True)

    # Set x-axis limits for all subplots
    for ax in g.axes.flat:  # Iterate through all axes
        if ax is not None:  # Avoid empty subplots
            ax.set_xlim(right=global_max)
            ax.set_ylim(top=global_max)

    # Add correlation coefficients to the upper triangle
    for i, row in enumerate(df.columns):
        for j, col in enumerate(df.columns):
            if i > j:  # lower triangle
                # Calculate correlation
                corr = df[row].corr(df[col])

                # Add text annotation to the pairplot
                ax = g.axes[i, j]
                ax.annotate('corr: ' + f'{corr:.2f}',
                            xy=(0.2, 0.9),
                            xycoords='axes fraction',
                            ha='center', va='center', fontsize=10, color='red')
            if i == j:  # diagnal
                # Calculate correlation
                total = sum(df[row])

                # Add text annotation to the pairplot
                ax = g.axes[i, j]
                ax.annotate("total count: " + f'{total}',
                            xy=(0.3, 0.9),
                            xycoords='axes fraction',
                            ha='center', va='center', fontsize=10, color='black')
    return (g)


def create_corr_heatmap(df):
    """
    all df columns should be counts, no NAME column

    print(m2.head(2))
       COUNT-A  COUNT-J  COUNT-L
    0       23       21       29
    1        6        2        4
    """
    corr_matrix = df.corr(method='pearson')  # {‘pearson’, ‘kendall’, ‘spearman’}
    g = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Pearson Correlation Heatmap')
    return (g)


def create_epo_curve(df_melted,
                     custom_palette={'i-count-A': '#666666'},
                     linestyle_dict={'i-count-A': ':'}
                     ):
    """
    df_melted example:

          Epo replicate Count_Type  Count
    0   0.0000         3    COUNT-A      0
    1   0.0625         1    COUNT-A     45
    2   0.0625         2    COUNT-A     48
    3   0.0625         3    COUNT-A     42
    4   0.0625         4    COUNT-A     73
    ..     ...       ...        ...    ...
    58  0.5000         4    COUNT-L    132
    59  1.0000         1    COUNT-L    211
    60  1.0000         2    COUNT-L    131
    61  1.0000         3    COUNT-L    143
    62  1.0000         4    COUNT-L    164

    [63 rows x 4 columns]

    custom_palette = {
        'i-count-A': '#666666',  # Gray1
        'i-count-J': '#999999',  # Gray2
        'i-count-L': '#BBBBBB',  # Gray3
        'm-count': '#0072B2',   # Colorblind-friendly blue for c-count
        'c-count-AF1-P0.1': '#D55E00',  # Colorblind-friendly red
    }


    linestyle_dict = {
    'i-count-A': ':',
    'i-count-J': ':',
    'i-count-L': ':',
    'm-count': '--',
    'c-count-AF1-P0.1': '-.',
    }

    """
    plt.figure(figsize=(6, 5))

    pointplot = sns.pointplot(
        data=df_melted,
        x='Epo', y='Count',
        hue='Count_Type', palette=custom_palette,
        dodge=True,
        markers='o',
        linestyles='--', # -  --  :  -.
        errorbar='se'  # se, sd, ci, pi
    )

    # Apply different linestyles by modifying the lines manually
    for line, (_, count_type) in zip(pointplot.lines[1::3], enumerate(df_melted['Count_Type'].unique())):
        lineStyle = linestyle_dict.get(count_type, '-')
        print(count_type)
        print(lineStyle)
        line.set_linestyle(lineStyle)


    plt.title('Epo Concentration Curve with Error Bars (SE)')
    plt.xlabel('Epo Concentration')
    plt.ylabel('Count')
    plt.legend(title='Count Type', loc='best')

    plt.show()

    return pointplot
