# Synthetic Data Generation for Fetal Chromosomal Aneuploidy Detection

## Introduction
This repository implements the synthetic data generation process described in *"Synthetic data-driven AI approach for fetal chromosomal aneuploidies detection"* (TBA).

The methodology addresses the critical limitation of scarce positive samples in Non-Invasive Prenatal Testing (NIPT) by generating virtually unlimited synthetic datasets with over 99.9% similarity to real-world data. This enables accurate detection of both Autosomal Chromosome Aneuploidies (ACA) and Sex Chromosome Aneuploidies (SCA).

## Environment Setup
- OS: Ubuntu 22.04.4 LTS
- The following script installs the required dependencies **bowtie2** and **samtools**:

  - bowtie2:
    ```bash
    sudo apt-get update
    sudo apt-get install -y bowtie2
    ```

  - samtools (v1.19.2):
    ```
    wget https://github.com/samtools/samtools/releases/download/1.19.2/samtools-1.19.2.tar.bz2
    tar -xf samtools-1.19.2.tar.bz2 
    cd samtools-1.19.2/
    ./configure 
    make
    sudo make install
    ```



- Python version: 3.11
  - Install the required python libraries using:
    ```
    pip install -r requirements.txt
    ```

- hg Reference Setting:
  ```
  # To run the demo code without modification, proceed within the UCSC_hg19 folder
  mkdir ./UCSC_hg19
  cd ./UCSC_hg19
  wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz
  gunzip hg19.fa.gz
  bowtie2-build hg19.fa index
  ```

## Input Data Requirements

### Sample Information File
Create a `Sample_FF_info.csv` file containing sample paths and fetal fractions:
```csv
sample_id,FF
./Demo_samples/Sample_male_1.Fastq,32.45
./Demo_samples/Sample_male_2.Fastq,31.23
./Demo_samples/Sample_male_3.Fastq,29.87
```

### FASTQ Preprocessing
The preprocessing pipeline uses the `extract_unique_reads` function from `preprocess_fastq.py`:

1. **Input**: Single-end `.Fastq` files from NIPT sequencing
2. **Trimming**: Reads trimmed to first 35 nucleotides
3. **Alignment**: Trimmed reads aligned to hg19 reference using Bowtie2
4. **SAM/BAM Processing**: Convert SAM to BAM, sort and index
5. **Duplicate Removal**: Remove duplicates using `samtools rmdup -s`
6. **Unique Read Filtering**: Extract uniquely mapped reads to `.unique.Fastq` files
7. **Quality Control**: Samples filtered for ≥2M unique reads, GC content 39.5-43%, FF ≥4%

The pipeline automatically generates `.unique.Fastq` files in the same directory as input files if they don't exist.  

## Synthetic Data Generation

### 1. Synthetic Negative Dataset (`make_synthetic_negative.py`)

Generates synthetic negative (normal) samples by combining multiple real FASTQ samples:

**Key Features:**
- **Sample Combination**: Randomly merges 2+ real FASTQ files from male/female groups
- **Weighted FF Calculation**: `Cff = (M1×M1ff + ... + Mn×Mnff) / (M1 + ... + Mn)`
- **Data Processing**: Shuffles combined reads and downsamples to control size
- **Output**: CSV files with chromosome read counts, GC content, and metadata

**Usage:**
```bash
python make_synthetic_negative.py
```

**Output Directory:** `./synthetic_negatives/`

### 2. Synthetic ACA Dataset (`make_synthetic_positive_ACA.py`)

Generates synthetic Autosomal Chromosome Aneuploidy (Trisomy) data using paper's Equation (1):

**Mathematical Foundation:**
- **Trisomy Formula**: `T_iRC = C_iRC + (C_iRC × FF) / 2`
- **Target Chromosomes**: Autosomes 1-22 (T13, T18, T21, etc.)

**Key Features:**
- **Internal Reference**: Uses last sample in combination as ACA reference source
- **Consistent Processing**: Same weighted FF calculation and merging as negative generation
- **Dual Output**: Both theoretical T-values and actual chromosome counts
- **Quality Control**: Automatic outlier filtering and GC content validation

**Usage:**
```bash
python make_synthetic_positive_ACA.py
```

**Output Directory:** `./synthetic_positive_ACA/`

### 3. Synthetic SCA Dataset (`make_synthetic_positive_SCA.py`)

Generates synthetic Sex Chromosome Aneuploidy data implementing four SCA equations:

**Mathematical Foundation:**
- **XO (Turner)**: `CX_RC = CXY_RC - CY_RC + Cy_RC` (Equation 3)
- **XYY (Jacob's)**: `CXYY_RC = CXY_RC + CY_RC - Cy_RC` (Equation 4)  
- **XXY (Klinefelter)**: `CXXY_RC = CXX_RC + CY_RC` (Equation 5)
- **XXX (Triple X)**: `CXXX_RC = CXX_RC + (CXX_RC × FF) / 2` (Equation 2)

**Regression Models:**
- **Misaligned Y Reads (XO, XYY)**: `Cy_RC = β0 + β1 · TRC(UR) + ε` (Equation 3.1)
  - **Coefficients**: β0 = 8.73854394, β1 = 3.76076E-05, σ = 14
- **XXY Y Chromosome Reads**: `Y_count = 0.000183 · TRC(UR) + 72.46617 · FF - 575.88`
  - **Standard deviation**: σ = 82 (>95% correlation with UR and FF)

**Key Features:**
- **Four SCA Types**: XO, XYY, XXY, XXX conditions
- **Regression Integration**: Models misaligned Y chromosome reads
- **Self-Contained**: No external Y reference file dependencies

**Gender-Specific Sample Requirements:**
- **XO (Turner) and XYY (Jacob's)**: Use **MALE** fetus samples only
- **XXY (Klinefelter) and XXX (Triple X)**: Use **FEMALE** fetus samples only

**Note**: The provided demo samples in this repository contain only **MALE** fetus samples. To generate XXY and XXX synthetic data, you will need to provide your own female fetus samples.

**Usage:**
```bash
python make_synthetic_positive_SCA.py
```

**Output Directory:** `./synthetic_positive_SCA/`

## ACA/SCA Logistic Regression Training

This section summarizes the ACA (Trisomy 13/18/21) and SCA (sex chromosome aneuploidies) Logistic Regression pipelines.

### Pipeline Overview
- ACA
  - Load ACA/normal datasets and merge male/female cohorts
  - Build features including chrN_count_3m (3M normalization handled automatically in code)
  - Build features: GC (scaled ×100), snp_FF, chrN_count_3m
  - Standardize features using training statistics only
  - Train Logistic Regression (class_weight='balanced', C=100)
  - Evaluate via ROC with Youden’s J threshold; show confusion matrix/metrics
  - Save per-chromosome model/scaler artifacts
- SCA
  - Load SCA backdata and harmonize labels (M/F → XY/XX)
  - Build features: GC (scaled ×100), UR, snp_FF, chr23_count, chr24_count
  - Standardize features; train multiclass Logistic Regression (class_weight='balanced')
  - Report macro metrics (accuracy/precision/recall) and confusion matrix
  - Save model/scaler artifacts

- Both training and testing CSV inputs must be produced by this project’s synthetic data generation pipeline and then preprocessed before use.
- Required feature columns:
  - ACA
    - Positive (ACA) CSV: `sample_id, result, GC, snp_FF, T13_UR, T18_UR, T21_UR, T13_RC, T18_RC, T21_RC`
    - Normal CSV: `sample_id, result, GC, snp_FF, ATRC, chr13_count, chr18_count, chr21_count`
    - The training pipeline automatically renames `T{N}_RC → chr{N}_count` and computes 3M-normalized counts; ensure the required columns exist. `GC` scaling (×100) is also handled in code.
  - SCA
    - CSV: `result, GC, UR, snp_FF, chr23_count, chr24_count`
    - Labels `M/F` are normalized to `XY/XX`. `GC` scaling (×100) is handled in code.

> Note: Do not feed raw synthetic outputs directly. Ensure columns and 3M normalization align with the feature columns above prior to training/testing.

### Scripts
- `train_ACA_LR_model.py`
  - Binary LR per chromosome (13/18/21)
  - Outputs: `lr_model_T{13|18|21}.joblib`, `lr_scaler_T{13|18|21}.joblib`
- `train_SCA_LR_model.py`
  - Multiclass LR for SCA (XY/XX/XO/XXY/XYY/XXX)
  - Outputs: `sca_lr_model.joblib`, `sca_lr_scaler.joblib`

### Usage
1) Install dependencies
```bash
pip install -r requirements.txt
```
2) Generate and preprocess synthetic data according to the schemas above

3) Run training
```bash
# ACA
python train_ACA_LR_model.py

# SCA
python train_SCA_LR_model.py
```

### Outputs
- Joblib artifacts for models and scalers
- Console confusion matrices and key metrics (ACA: ROC/Youden’s J-based; SCA: multiclass macro metrics)

## Contact

For questions or technical support regarding this implementation, please contact: **hch8357@naver.com**
