import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt


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
    # pattern2 = r"([\d.]+).(rep[\d.]+).txt"
    pattern2 = r"([\d.]+)\.(rep[\d]+)\.txt"
    # Iterate over all files in the directory
    for filename in files:
        if filename.endswith(".txt"):  # Ensure it's a .txt file
            filepath = filename
            match2 = re.search(pattern2, filename)

            with open(filepath, 'r') as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        # Extract values
                        precision, recall, f1 = map(float, match.groups())
                        proportion, rep = map(str, match2.groups())
                        data.append(
                            {"Name": filename.replace('.txt', ''),
                             "Proportion": proportion,
                             "Rep": rep,
                             "Precision": precision,
                             "Recall": recall,
                             "F1": f1})
                        break  # Stop reading after finding the line

    # Create a DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by="Name").set_index("Name")
    return df


def plot_training_saturation(df, mode='F1', out_keyword='saturation_analysis'):
    """
    Input df from <read_eval_txts_into_df>
    mode: F1, Precision, Recall, All
    Plot: saturation: Metric Score change over proportion
    """
    if mode not in ['F1', 'Precision', 'Recall', 'All']:
        raise Exception('mode not recognized')

    plt.figure(figsize=(6, 5))

    if mode in ['All', 'F1']:
        plt.plot(df.index, df["F1"], marker='o', label="F1", color='red')
    if mode in ['All', 'Precision']:
        plt.plot(df.index, df["Precision"], marker='o', label="Precision", color = 'lightblue')
    if mode in ['All', 'Recall']:
        plt.plot(df.index, df["Recall"], marker='o', label="Recall", color = 'lightgreen')

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
    '''
    '''
    import seaborn as sns
    plt.figure(figsize=(6, 5))

    pointplot = sns.pointplot(
        data=df_melted,
        x='Proportion', y='Value',
        #hue='ScoreType',
        dodge=True,
        markers='o',
        color='red',
        linestyles='--',  # -  --  :  -.
        errorbar='ci'  # se, sd, ci, pi
    )

    plt.title('Saturation Analysis with Error Bars (CI)')
    plt.xlabel('Proportion of Training Data Used')
    plt.ylabel('F1 Score')
    plt.legend(title='F1 Score', loc='best')

    return (pointplot)


files = sys.argv[0:-1]
df = read_eval_txts_into_df(files)
print(df)


melt = df.melt(
    id_vars=['Proportion', 'Rep'],
    # value_vars=['Precision', 'Recall', 'F1'],
    value_vars=['F1'],
    var_name='ScoreType', value_name='Value')

print(melt.shape)
# melt.to_csv('count_plate_level_melt.csv', index=0)
melt

saturation_plot = create_saturation_curve(melt)
saturation_plot.figure.savefig(sys.argv[-1])  # Save to PDF
