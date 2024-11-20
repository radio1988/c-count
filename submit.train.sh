#!/bin/bash
# run with:
# bsub -q long -W 144:00 -R rusage[mem=4000] 'bash submit.train.sh'

source activate ccount-env

snakemake \
-s workflow/train.Snakefile \
-p -k --jobs 999 \
--latency-wait 120 \
--ri --restart-times 0 \
--cluster 'bsub -q long -o lsf.log -R "rusage[mem={resources.mem_mb}]" -n {threads} -R span[hosts=1] -W 144:00' > snakemake.log 2>&1
