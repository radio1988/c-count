conda activate c-count-env
snakemake -s workflow/count.Snakefile -j1 reset  # remove all previous results
snakemake -s workflow/count.Snakefile -pk --ri -j1  # run counting on example data
snakemake -s workflow/count.Snakefile -j 1 --report report.html  # generate report on the run-time
