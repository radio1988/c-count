"""
Input: res/eval/xxx.txt  # multiple files
Output:
- res/plot/saturation_analysis.pdf
- res/plot/saturation_analysis.pdf.csv

Usage example:
python workflow/scripts/plot_saturation.py \
res/3_evaluation_on_validationSet/0.125.rep1.txt .. \
res/3_evaluation_on_validationSet/1.0.rep9.txt \
res/plots/saturation_analysis.pdf  \
&>res/plots/saturation_analysis.pdf.log
"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_eval_txts_into_df(files):
    """
    Read all txt files in directory, default 'res/eval' from train.Snakefile workflow
    Output a sorted df like this:
        #        Precision  Recall     F1
        # Name
        # 0.125      63.41   84.78  72.56
        # 0.25       84.81   72.83  78.36
        # 0.5        68.33   89.13  77.36
        # 1.0        74.75   80.43  77.49
    """
    data = []
    pattern = r"Precision: ([\d.]+)%, Recall: ([\d.]+)%, F1: ([\d.]+)%"
    pattern2 = r"([\d.]+)\.(rep[\d]+)\.txt"
    p_auc_pr = r"AUC-PR: ([\d.]+)"
    p_mcc_max = r"MCC-MAX: ([\d.]+)"

    for filename in files:
        filepath = filename
        match2 = re.search(pattern2, filename)
        if match2:
            proportion, rep = map(str, match2.groups())
        else:
            sys.exit("Error: filename does not match pattern")

        with open(filepath, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    precision, recall, f1 = map(float, match.groups())
                match3 = re.search(p_auc_pr, line)
                if match3:
                    auc_pr = float(match3.group(1))
                match4 = re.search(p_mcc_max, line)
                if match4:
                    mcc_max = float(match4.group(1))

            data.append(
                {"Name": filename.replace('.txt', ''),
                 "Proportion": proportion,
                 "Rep": rep,
                 "Precision": precision,
                 "Recall": recall,
                 "F1": f1,
                 "AUC-PR": auc_pr,
                 "MCC-MAX": mcc_max
                 })

    # Create a DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by="Name").set_index("Name")
    return df


def plot_training_saturation(df, mode='F1', out_keyword='saturation_analysis'):
    """
    Obsolete
    Parse txt files, get metric scores, plot saturation over training subset proportions
    Simply plot them as a scatterplot with connecting lines, not collapsing to proportion
    This function is obsolete

    @param df: output from <read_eval_txts_into_df>
    @param mode: F1, Precision, Recall, or All (metric score types)
    @param out_keyword: name of the saturation: selectede metric score change over proportion
    @return: plot object
    """

    if mode not in ['F1', 'Precision', 'Recall', 'All']:
        raise Exception('mode not recognized')

    plt.figure(figsize=(6, 5))

    if mode in ['All', 'F1']:
        plt.plot(df.index, df["F1"], marker='o', label="F1", color='red')
    if mode in ['All', 'Precision']:
        plt.plot(df.index, df["Precision"], marker='o', label="Precision", color='lightblue')
    if mode in ['All', 'Recall']:
        plt.plot(df.index, df["Recall"], marker='o', label="Recall", color='lightgreen')

    # Add labels and title
    plt.title("Saturation Analysis")
    plt.ylabel("Scores")
    plt.xlabel("Percent of Training Data Used")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_keyword + '.' + mode + '.pdf')
    plt.show()


def create_saturation_curve(df_melted):
    """
    Create saturation plot using boxplots to summarize the distribution (CI added)
    and add jitter to show individual data points.

    @param df_melted: DataFrame with columns Proportion, Rep, ScoreType, Value.
               Proportion   Rep ScoreType  Value
    0       0.125  rep1        F1  81.32
    1       0.125  rep2        F1  72.54
    2       0.125  rep3        F1  70.65
    8        0.25  rep1        F1  75.80
    9        0.25  rep2        F1  68.83
    10       0.25  rep3        F1  79.78

    @return: plt object
    """
    plt.figure(figsize=(6, 5))

    pointplot = sns.pointplot(
        data=df_melted,
        x='Proportion', y='Value',
        # hue='ScoreType',
        dodge=True,
        markers='o',
        color='red',
        linestyles='--',  # -  --  :  -.
        errorbar='ci'  # se, sd, ci, pi
    )

    # boxplot = sns.boxplot(
    #     data=df_melted,
    #     x='Proportion', y='Value',
    #     color='lightgray',  # Neutral color for the boxplot
    #     showfliers=False,   # Hide outliers to reduce visual clutter
    #     width=0.5,          # Narrower boxes for better layout
    #     linewidth=1.5       # Thicker boxplot lines
    # )

    stripplot = sns.stripplot(
        data=df_melted,
        x='Proportion', y='Value',
        color='black',  # Adjust color for visibility
        alpha=0.6,  # Transparency for better layering
        jitter=True,  # Add jitter
        dodge=True  # Align with pointplot
    )

    plt.title('Saturation Analysis with Error Bars (CI) and Jitter')
    plt.xlabel('Proportion of Training Data Used')
    plt.ylabel('F1 Score')
    plt.legend(title='F1 Score', loc='best')

    return plt


def create_saturation_curve_jittered_boxplot(df_melted):
    """
    Create saturation plot using boxplots to summarize the distribution
    and add jitter to show individual data points.

    @param df_melted: DataFrame with columns Proportion, Rep, ScoreType, Value.
               Proportion   Rep ScoreType  Value
        0       0.125  rep1        F1  81.32
        1       0.125  rep2        F1  72.54
        2       0.125  rep3        F1  70.65
        8        0.25  rep1        F1  75.80
        9        0.25  rep2        F1  68.83
        10       0.25  rep3        F1  79.78

    @return: plt object
    """
    plt.figure(figsize=(6, 5))

    # Boxplot for distribution and summary stats
    boxplot = sns.boxplot(
        data=df_melted,
        x='Proportion', y='Value',
        color='lightgray',  # Neutral color for the boxplot
        showfliers=False,  # Hide outliers to reduce visual clutter
        width=0.5,  # Narrower boxes for better layout
        linewidth=1.5  # Thicker boxplot lines
    )

    # Stripplot for individual points with jitter
    stripplot = sns.stripplot(
        data=df_melted,
        x='Proportion', y='Value',
        hue='ScoreType',
        color='blue',  # Blue for data points
        alpha=0.6,  # Transparency for layering
        jitter=True,  # Add jitter
        dodge=True  # Align with boxplot
    )

    plt.title('Saturation Analysis with Boxplots and Jitter')
    plt.xlabel('Proportion of Training Data Used')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping

    return plt


def main():
    files = sys.argv[0:-1]
    df = read_eval_txts_into_df(files)
    print(df)
    os.makedirs(os.path.dirname(sys.argv[-1]), exist_ok=True)
    df.to_csv(sys.argv[-1] + ".csv")

    melt = df.melt(
        id_vars=['Proportion', 'Rep'],
        # value_vars=['Precision', 'Recall', 'F1'],
        value_vars=['F1'],
        var_name='ScoreType', value_name='Value')

    # print(melt.shape)
    # print(melt)

    saturation_plot = create_saturation_curve(melt)
    saturation_plot.savefig(sys.argv[-1], dpi=300, bbox_inches='tight')  # Save to PDF


if __name__ == '__main__':
    main()
