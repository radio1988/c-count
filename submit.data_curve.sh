#!/bin/bash
# run with:
# bsub -q long -W 144:00 -R rusage[mem=4000] 'bash submit.sh && echo yung'

# snakemake -np -s workflow/Snakefile_data_curve
source activate ccount-env
snakemake \
-s workflow/Snakefile_data_curve \
-p -k --jobs 999 \
--latency-wait 120 \
--ri --restart-times 0 \
--cluster 'bsub -q long -o lsf.log -R "rusage[mem={resources.mem_mb}]" -n {threads} -R span[hosts=1]  -R select[rh=8] -W 144:00' > snakemake.log 2>&1
