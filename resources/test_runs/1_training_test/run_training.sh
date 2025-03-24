conda activate c-count-env
snakemake -s workflow/train.Snakefile -j1 reset  # remove all previous results
snakemake -s workflow/train.Snakefile -pk --ri -j1  # run training on example data
snakemake -s workflow/train.Snakefile -j 1 --report report.html  # generate report on the run-time
