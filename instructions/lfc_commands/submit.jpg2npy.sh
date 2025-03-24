#!/bin/bash
# run with:
# bsub -q long -W 144:00 -R rusage[mem=4000] 'bash submit.jpg2npy.sh'

source /home/rui.li-umw/anaconda3/etc/profile.d/conda.sh
conda activate mamba > jpg2npy.log 2>&1

snakemake \
-s workflow/jpg2npy.Snakefile \
--use-singularity --singularity-args "--bind ~/github/ccount/workflow/" \
--rerun-triggers mtime \
-p -k --jobs 999 \
--ri --restart-times 1 \
--cluster 'bsub -q short -o lsf.log -R "rusage[mem={resources.mem_mb}]" -n {threads} -R span[hosts=1] -W 4:00' > jpg2npy.log 2>&1

snakemake -j 1 --report report.html -s workflow/train.Snakefile  >> jpg2npy.log  2>&1
