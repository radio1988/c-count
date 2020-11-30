#cd workdir
#ln -s $path/scripts
#cp scripts/config.yaml ./ # then edit

nohup snakemake -s scripts/Snakefile -k --jobs 999 --latency-wait 60 \
--cluster 'bsub -q short -o lsf.log -R "rusage[mem={params.mem}]" -n {threads} -R span[hosts=1] -W 4:00' && snakemake -j 1 --report report.html &
