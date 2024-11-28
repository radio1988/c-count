rule czi2img:
    input:
        os.path.join(config['DATA_DIR'], "{s}.czi")
    output:
        touch("log/img/{s}.done")
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 16000  # ~10.5G for '2019'
    log:
        "log/img/{s}.log"
    benchmark:
        "log/img/{s}.benchmark"
    shell:
        """
        python workflow/scripts/czi2img.py -i {input} -c config.yaml -odir res/img &> {log}
        """
