import os
import subprocess

def extract_unique_reads(fastq_id, reference, threads=8, keep_temp=False):
    """
    Run the FASTQ → Bowtie2 → SAMtools → unique reads filtering pipeline.

    Parameters
    ----------
    fastq_id : str
        FASTQ file prefix (ex. SAMPLE then SAMPLE.Fastq input file is found)
    reference : str
        Bowtie2 index prefix (ex. /BiO/NIPT_V3/NIPT_V3/ref_genomom/UCSC_hg19/index)
    threads : int
        Bowtie2 / samtools multi-thread option
    keep_temp : bool
        Intermediate file preservation
    """

    fastq_file = f"{fastq_id}.Fastq"
    fq_trim = f"{fastq_id}.fq.fastq"
    sam = f"{fq_trim}.sam"
    bam = f"{sam}.bam"
    bam_sort = f"{bam}.sort.bam"
    bam_rmdup = f"{bam_sort}.rmdup.bam"
    sam_rmdup = f"{bam_rmdup}.sam"
    unique_file = f"{fastq_id}.Fastq.mapped"

    cmds = [f"cat {fastq_file} | sed 's/\\(.\{{35\\}}\\).*/\\1/' > {fq_trim}", ## 1. Fastq 35nt trimming
            f"bowtie2 -p {threads} -x {reference} {fq_trim} -S {sam}", # 2. bowtie2 mapping
            f"wc -l {sam} > {fastq_id}.rawread", # 3. Raw read count
            f"samtools view -@{threads} -Sb {sam} > {bam}", # 4. SAM → BAM → sort → index
            f"samtools sort -@{threads} {bam} -o {bam_sort}",
            f"samtools index -@{threads} {bam_sort}",
            f"samtools rmdup -s {bam_sort} {bam_rmdup}", # 5. rmdup
            f"samtools view -@{threads} {bam_rmdup} > {sam_rmdup}",
            f"cat {sam_rmdup} | grep 'AS:i:0' | awk '$5>=36 && $5<=42 {{print $0}}' > {unique_file}", # 6. Unique filtering: Range parameters 36–42 can be modified as needed
            ]


    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

    
    print(f"[DONE] Unique reads written to {unique_file}")

    # clean up
    if not keep_temp:
        for f in [fq_trim, sam, bam, f"{fastq_id}.rawread",
                  bam_sort, bam_rmdup, f"{bam_sort}.bai", sam_rmdup]:
            if os.path.exists(f):
                os.remove(f)
        print("[CLEANUP] Temporary files removed.")

    return unique_file