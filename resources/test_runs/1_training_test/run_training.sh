conda activate c-count-env
snakemake -s workflow/train.Snakefile -pk --ri -j1
snakemake -s workflow/train.Snakefile -j 1 --report report.html
