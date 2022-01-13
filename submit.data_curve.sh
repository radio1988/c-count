#!/bin/bash
# run with:
# bsub -q long -W 144:00 -R rusage[mem=4000] 'bash submit.data_curve.sh && echo yung5'

# snakemake -np -s workflow/Snakefile_data_curve
source activate ccount-gpu
snakemake \
-s workflow/Snakefile_data_curve \
-p -k --jobs 999 \
--latency-wait 120 \
--ri --restart-times 0 \
--cluster 'bsub -q gpu -o lsf.log -R "rusage[mem={resources.mem_mb}]" -n {threads} -R span[hosts=1]  -R "select[rh=8 && ncc>=7.0]" -W 24:00' > snakemake.log 2>&1
