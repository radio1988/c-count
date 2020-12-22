import sys

def get_countDF_from_count_log(fname="classification1/COUNT1.txt"):
    """
    Read log file: classification1/COUNT1.txt
    Output data frame of label and colony count
        label	colony_count
    0	E2f4_CFUe_KO_1-Stitching-01.0	171
    1	E2f4_CFUe_KO_1-Stitching-01.1	105
    2	E2f4_CFUe_KO_1-Stitching-01.2	132
    3	E2f4_CFUe_KO_1-Stitching-01.3	114
    4	E2f4_CFUe_KO_2-Stitching-02.0	190
    ...	...	...
    61	E2f4_CFUe_WT3_3-Stitching-20.0	127
    62	E2f4_CFUe_WT3_3-Stitching-20.1	70
    63	E2f4_CFUe_WT3_3-Stitching-20.2	95
    64	E2f4_CFUe_WT3_3-Stitching-20.3	165
    65	E2f4_CFUe_WT3_3_Top-Stitching-21.0	141
    """
    import re
    import pandas as pd

    file = open(fname, "r")
    labels = []
    colony_counts = []
    for line in file:
        label = re.search("[\w\.\-]+log:Predictions", line)
        if label:
            label = re.sub(".log.Predictions", "", label.group())
            yes = re.search("count_yes: \d+", line)
            yes = int(yes.group().replace("count_yes: ", ""))
            labels.append(label)
            colony_counts.append(yes)
    df = pd.DataFrame(
        list(zip(labels, colony_counts)),
        columns=['label', 'colony_count']
                      )
    return(df)

print("usage: get_counts.py COUNT1.txt output_csv_fname")
print("example: python workflow/get_counts.py res/classification1/COUNT1.txt res/COUNT.csv")
if len(sys.argv) < 3:
    sys.exit("cmd error")
df1 = get_countDF_from_count_log(sys.argv[1])
print(df1.head())
df1.to_csv(sys.argv[2])
