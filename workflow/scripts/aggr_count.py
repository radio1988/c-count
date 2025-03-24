"""
Read count.txt and aggregate counts
"""

import sys
import os
import pandas as pd

def aggr_count_info(inputs, label = int(1)):
    names = []
    counts = []
    for f in inputs:
        name = os.path.basename(f)
        name = name.replace('.crops.clas.txt', '')
        c = pd.read_csv(f, header=None)
        c = c.astype(int)
        count = sum(c.values == 1)[0]
        print(f, count)
        names.append(name)
        counts.append(count)
    return pd.DataFrame(zip(names, counts), columns=['NAME', 'COUNT'])


print(">>> usage: aggr_count.py COUNT-1.txt COUNT-2.txt ... COUNT-n.txt output.csv")
print(">>> example: python workflow/scripts/aggr_count_info.py.py \
    x.clas.txt y.clas.txt res/COUNT.csv")

if len(sys.argv) < 3:
    sys.exit("cmd error")

inputs = sys.argv[1:-1]
print(">>> inputs:", inputs)
output = sys.argv[-1]

count_df = aggr_count_info(inputs, label=int(1))
print(">>> output", output, ":\n", count_df)

count_df.to_csv(output)
