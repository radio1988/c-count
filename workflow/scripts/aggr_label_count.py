import sys
import pandas as pd
import os


def extract_yes_count(text):
    """Extracts the number before 'Yes' from the given text.

    Args:
      text: The text to extract the number from.

    Returns:
      The extracted number as an integer, or None if not found.
    """
    import re

    match = re.search(r'(\d+) Yes', text)
    if match:
        return int(match.group(1))
    else:
        sys.exit("Yes not found, format error in LABEL.npy.gz.txt")


def aggr_label_count(inputs):
    names = []
    counts = []
    for f in inputs:
        name = os.path.basename(f)
        name = name.replace('.LABEL.npy.gz.txt', '')
        name = name.replace('.log.txt', '')
        with open(f, 'r') as file:
            text = file.read()
            count = extract_yes_count(text)
            count = int(count)
        names.append(name)
        counts.append(count)
    return pd.DataFrame(zip(names, counts), columns=['NAME', 'COUNT'])


print(">>> usage: aggr_count_info.py COUNT-1.txt COUNT-2.txt ... COUNT-n.txt output.csv")
print(">>> example: python workflow/scripts/aggr_count_info.py.py \
    x.clas.txt y.clas.txt res/COUNT.csv")

if len(sys.argv) < 3:
    sys.exit("cmd error")

inputs = sys.argv[1:-1]
print(">>> inputs:", inputs)
output = sys.argv[-1]

count_df = aggr_label_count(inputs)
print(">>> output", output, ":\n", count_df)
count_df.to_csv(output)
