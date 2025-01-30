#!/bin/bash
# run with:
# bsub -q long -W 144:00 -R rusage[mem=4000] 'bash submit.train.sh'

source /home/rui.li-umw/anaconda3/etc/profile.d/conda.sh
conda activate mamba > train.log 2>&1

snakemake \
-s workflow/count.Snakefile \
--rerun-triggers mtime \
-p -k --jobs 999 \
--ri --restart-times 1 \
--cluster 'bsub -q long -o lsf.log -R "rusage[mem={resources.mem_mb}]" -n {threads} -R span[hosts=1] -W 144:00' > train.log 2>&1
