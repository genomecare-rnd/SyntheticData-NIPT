#!/home/albatross/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Synthetic Positive SCA (Sex Chromosome Aneuploidies) Data Generator

This module implements the synthetic positive SCA data generation methodology described in:
"Synthetic Data-based Artificial Intelligence Approach for Fetal Chromosomal Aneuploidies Detection"

Implements the following SCA equations from the paper:
- XO (Turner syndrome):  Equation (3): CX_RC = CXY_RC - CY_RC + Cy_RC
- XYY (Jacob's syndrome): Equation (4): CXYY_RC = CXY_RC + CY_RC - Cy_RC  
- XXY (Klinefelter):     Equation (5): CXXY_RC = CXX_RC + CY_RC
- XXX (Triple X):        Equation (2): CXXX_RC = CXX_RC + (CXX_RC × FF) / 2

Regression model for misaligned Y chromosome reads (Equations 3.1, 3.2):
- Cy_RC = β0 + β1 · TRC + ε  (3.1)
- Cy_RC′ = β0 + β1 · TRC    (3.2)
where β0 = 8.73854394, β1 = 3.76076E-05, σ = 14

Features:
- Applies systematic sample combination approach from negative generation
- Uses weighted average FF calculation based on unique read counts
- Handles all four major SCA conditions using internal Y chromosome reference
- Dynamically sets Y reference from last sample in each combination
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from utils.preprocess_fastq import extract_unique_reads


def get_filename_from_path(file_path):
    """
    Extract filename without extension from full file path
    
    Parameters
    ----------
    file_path : str
        Full file path (e.g., /SyntheticDataGeneration/Demo_samples/Sample_male_1.Fastq)
        
    Returns
    -------
    str
        Filename without extension (e.g., Sample_male_1)
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def make_combined_SCA(fastq_dfs, filenames, sca_type):
    """
    Generate synthetic positive SCA data by combining multiple samples
    
    This function implements the paper's SCA generation methodology:
    1. Combines multiple samples using weighted averaging (identical to negative generation)
    2. Weights each sample's contribution by its unique read count (UR)
    3. Sets up Y chromosome reference using the last sample in the combination
    4. Applies SCA-specific transformations based on paper's equations (2-5)
    
    The weighted FF calculation follows the paper's formula:
    Cff = (M1 × M1ff + ... + Mi × Miff + ... + Mn × Mnff) / (M1 + ... + Mi + ... + Mn)
    
    Y chromosome reference strategy:
    - Uses the last sample in the combination as Y chromosome reference source
    - Eliminates dependency on external Y_ref files
    - Ensures consistency and reproducibility across different environments
    
    Parameters
    ----------
    fastq_dfs : dict
        Dictionary containing sample data and FF values
        { filename: {"DATA": DataFrame, "FF": float} }
        where DATA contains unique mapped reads with columns: Chr, Pos, Seq
    filenames : list
        List of sample filenames to combine for synthetic SCA generation
    sca_type : str
        Type of SCA to generate: 'XO', 'XYY', 'XXY', or 'XXX'
        Each corresponds to specific equations from the paper
        
    Returns
    -------
    tuple
        (final_df, filename_new) - processed dataframe and output filename
        final_df contains chromosome read counts after SCA transformation
    """
    
    # Step 1: Sample combination using weighted averaging (identical to negative generation)
    dfs = []
    comb_UR = 0
    weighted_FF = 0
    
    # Collect each unique-read table and its FF; accumulate weighted terms
    # UR (number of unique reads) serves as the weight for FF averaging
    # Implements paper's formula: Cff = (M1×M1ff + ... + Mn×Mnff) / (M1 + ... + Mn)
    for filename in filenames:
        df = fastq_dfs[filename]['DATA']
        FF = fastq_dfs[filename]['FF']
        UR = df.shape[0]  # Mi: number of unique reads in ith FASTQ file
        
        dfs.append(df)
        comb_UR += UR  # Accumulate total read count denominator
        weighted_FF += (UR * FF)  # Accumulate weighted FF numerator (Mi × Miff)
    
    # Calculate weighted average fetal fraction of the mixture
    FF_new = weighted_FF / comb_UR
    
    # Merge all unique-read tables and randomly shuffle
    # Creates combined FASTQ file C as described in the paper
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    # Downsample to a fixed ratio to control synthetic sample size
    sampling_ratio = 1 / len(filenames)
    base_sample_df = merged_df[:int(len(merged_df) * sampling_ratio)].copy()
    
    # Step 2: Set up chromosome references using last sample in combination
    # Instead of using external Y_ref/X_ref files, use the last sample as reference source
    # This ensures consistency and eliminates dependency on external reference files
    ref_sample = filenames[-1]  # Use last sample in combination as reference
    ref_sample_df = fastq_dfs[ref_sample]['DATA']
    
    # Extract Y and X chromosome reads from reference sample
    #! REFERENCE SAMPLE IS NOT USED IN THE CODE
    # y_ref_df = ref_sample_df[ref_sample_df['Chr'] == 'chr24']  # Y chromosome reads
    # x_ref_df = ref_sample_df[ref_sample_df['Chr'] == 'chr23']  # X chromosome reads
    
    y_ref_df = base_sample_df[base_sample_df['Chr'] == 'chr24']
    x_ref_df = base_sample_df[base_sample_df['Chr'] == 'chr23']
    
    # Step 3: Apply SCA-specific transformations
    # Calculate GC content for quality control
    base_sample_df["GC"] = base_sample_df["Seq"].apply(lambda x: x.count("G") + x.count("C"))
    base_sample_df["Total"] = base_sample_df["Seq"].apply(len)
    Sample_GC = base_sample_df["GC"].sum() / base_sample_df["Total"].sum()
    
    UR = base_sample_df.shape[0]  # Total Unique Read Count (TRC in the paper)
    
    # Regression coefficients from paper's Equations (3.1, 3.2)
    # Used for modeling misaligned Y chromosome reads (Cy_RC)
    beta0 = 8.73854394  # Intercept
    beta1 = 3.76076E-05  # Coefficient between TRC and Cy_RC
    sigma = 14           # Standard deviation for normal distribution

    # Apply SCA-specific transformations based on paper's equations
    # IMPORTANT: Gender-specific sample requirements for SCA generation:
    # - XXY (Klinefelter) and XXX (Triple X): Use FEMALE fetus samples only
    # - XO (Turner) and XYY (Jacob's): Use MALE fetus samples only
    
    if sca_type == 'XO':
        # Generate XO (Turner syndrome) data using paper's Equation (3)
        # CX_RC = CXY_RC - CY_RC + Cy_RC
        # Uses regression model to estimate misaligned Y chromosome reads (Cy_RC)
        # NOTE: Should use MALE fetus samples for XO generation
        
        # Calculate predicted misaligned Y reads using Equation (3.2): Cy_RC′ = β0 + β1 · TRC(UR)
        mean = beta1 * UR + beta0  # Predicted value
        # Add random residual from normal distribution (Equation 3.1): Cy_RC = Cy_RC′ + ε
        y_count = int(np.random.normal(mean, sigma))
        
        # Separate non-Y and Y chromosome data
        except_y = base_sample_df[base_sample_df['Chr'] != 'chr24']  # All except Y
        y_reads_df = base_sample_df[base_sample_df['Chr'] == 'chr24']  # Y chromosome only
        
        # For XO: remove most Y reads, keep only misaligned ones (Cy_RC)
        if len(y_reads_df) >= y_count and y_count > 0:
            sampled_y = y_reads_df.sample(y_count)
            S_SCA_df = pd.concat([except_y, sampled_y], axis=0, ignore_index=True)
        else:
            # If y_count is 0 or negative, completely remove Y chromosome
            S_SCA_df = except_y.copy()
            
    elif sca_type == 'XYY':
        # Generate XYY (Jacob's syndrome) data using paper's Equation (4)
        # CXYY_RC = CXY_RC + CY_RC - Cy_RC
        # Uses Y chromosome data from the last sample in combination as reference
        # NOTE: Should use MALE fetus samples for XYY generation
        # Get existing Y chromosome data
        y_reads_df = base_sample_df[base_sample_df['Chr'] == 'chr24']
        
        # Calculate misaligned Y reads using regression model
        mean = beta1 * UR + beta0
        remove_y_count = int(np.random.normal(mean, sigma))  # Cy_RC (misaligned Y)
        
        # Calculate additional normal Y reads needed: CY_RC - Cy_RC
        add_y_count = len(y_reads_df) - remove_y_count
        add_y_count = max(0, add_y_count)  # Prevent negative values
        
        if add_y_count > 0:
            # Sample additional Y reads from reference data
            y_add_count = min(add_y_count, len(y_ref_df))
            add_Y_df = y_ref_df.sample(y_add_count)
            S_SCA_df = pd.concat([base_sample_df, add_Y_df], axis=0, ignore_index=True)
        else:
            S_SCA_df = base_sample_df.copy()
            
    elif sca_type == 'XXY':
        # Generate XXY (Klinefelter syndrome) data using paper's Equation (5)
        # CXXY_RC = CXX_RC + CY_RC
        # Uses Y chromosome data from the last sample in combination as reference
        # NOTE: Should use FEMALE fetus samples for XXY generation
        # XXY regression formula implementing paper's Equations (5.1, 5.2)
        # Complete regression model: Cy_RC = β0 + β1 · TRC + β2 · FF_snp + ε
        # Coefficients: β0 = -575.88, β1 = 0.000183, β2 = 72.46617, σ = 82
        # This accounts for both read count and fetal fraction effects on Y chromosome reads
        mean = 0.000183 * UR + 72.46617 * FF_new - 575.88
        std = 82
        
        # Generate Y chromosome read count using normal distribution
        y_count = int(np.random.normal(mean, std))
        y_count = max(0, y_count)  # Prevent negative values
        
        if y_count > 0:
            y_add_count = min(y_count, len(y_ref_df))
            add_Y_df = y_ref_df.sample(y_add_count)
            S_SCA_df = pd.concat([base_sample_df, add_Y_df], axis=0, ignore_index=True)
        else:
            S_SCA_df = base_sample_df.copy()
            
    elif sca_type == 'XXX':
        # Generate XXX (Triple X syndrome) data using paper's Equation (2)
        # CXXX_RC = CXX_RC + (CXX_RC × FF) / 2
        # NOTE: Should use FEMALE fetus samples for XXX generation
        X_reads_df = base_sample_df[base_sample_df['Chr'] == 'chr23']
        CXX_RC = len(X_reads_df)  # Normal X chromosome read count
        
        # Calculate additional X reads needed using refined formula
        # Formula: add_x_count = len(chrX) * (FF * 0.01) * 0.5
        # This accounts for fetal fraction percentage and dosage compensation
        add_x_count = int(CXX_RC * (FF_new * 0.01) * 0.5)
        
        # Sample additional X chromosome reads from reference sample X data
        # This follows the same approach as make_Total_female.py using X_ref
        if add_x_count > 0:
            x_add_count = min(add_x_count, len(x_ref_df))
            add_X_df = x_ref_df.sample(x_add_count)
            S_SCA_df = pd.concat([base_sample_df, add_X_df], axis=0, ignore_index=True)
        else:
            S_SCA_df = base_sample_df.copy()
    else:
        raise ValueError(f"Unsupported SCA type: {sca_type}")

    # Step 3: Calculate final chromosome read counts after SCA transformation
    chr_read_counts = []
    for i in range(1, 25):
        chr_count = len(S_SCA_df[S_SCA_df['Chr'] == f'chr{i}'])
        chr_read_counts.append(chr_count)

    # Step 4: Generate output dataframe and filename
    chr_col_names = [f'chr{i}' for i in range(1, 25)]  # Chromosome read counts after SCA transformation
    
    # Create descriptive filename including SCA type, sample combination, and weighted FF
    combined_names = '_'.join(filenames)
    filename_new = f'SynSCA_{sca_type}_{combined_names}_{FF_new:.2f}.csv'

    # Output format consistent with make_Total_male.py (includes del_range column)
    final_col_names = ['sample_id', 'result', 'GC', 'UR', 'snp_FF', 'del_range'] + chr_col_names
    final_df = pd.DataFrame(
        np.array([filename_new, sca_type, Sample_GC, UR, FF_new, 0] + chr_read_counts)
    ).T
    final_df.columns = final_col_names
    
    return final_df, filename_new


if __name__ == "__main__":
    # Define SCA types to generate synthetic data for
    sca_types = ['XO', 'XYY', 'XXY', 'XXX']
    
    # Set output directory for synthetic SCA data
    output_dir = './synthetic_positive_SCA'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Load per-sample FF values from CSV file
    sample_data = pd.read_csv('Sample_FF_info.csv')
    file_paths = sample_data['sample_id'].tolist()  # All file paths
    filenames = [get_filename_from_path(path) for path in file_paths]  # Extract filenames only
    
    all_combinations = []
    
    # Generate combinations of 2..N files for SCA generation
    for comb_number in range(2, min(4, len(filenames)+1)):  # Maximum 3-file combinations
        all_combinations += list(combinations(filenames, comb_number))

    # Create dictionary mapping filenames to their full paths
    filename_to_path = {get_filename_from_path(path): path for path in file_paths}
    
    # Y chromosome reference will be dynamically set from combination samples
    # Each combination will use its last sample as Y chromosome reference source
    # This approach eliminates dependency on external Y_ref files and ensures consistency
    
    # Process each SCA type separately
    for sca_type in sca_types:
        print(f"\n=== Generating {sca_type} synthetic data ===")
        
        # Generate SCA samples for each combination
        combination_desc = f"Processing {sca_type} combinations"
        for fastq_combination in tqdm(all_combinations, desc=combination_desc):
            fastq_dfs = {}
            
            # Load data for each file in the combination
            for filename in fastq_combination:
                # Get original file path
                original_file_path = filename_to_path[filename]
                
                # Create unique FASTQ file path in the same directory as original file
                original_dir = os.path.dirname(original_file_path)
                unique_fastq_path = os.path.join(original_dir, f"{filename}.unique.Fastq")
                
                # Ensure unique-read FASTQ exists; if not, extract unique reads
                if not os.path.exists(unique_fastq_path):
                    unique_file = extract_unique_reads(original_file_path, './UCSC_hg19/index', threads=12, keep_temp=False)

                # Load essential fields: chromosome, position, sequence from unique-read FASTQ
                unique_df = pd.read_table(
                    unique_fastq_path,
                    header=None, usecols=[2, 3, 9],
                    names=['Chr', 'Pos', 'Seq'],
                    dtype={'Chr': 'str', 'Pos': 'int32', 'Seq': 'str'}
                )

                # Normalize X and Y chromosome labels to numeric format (23, 24)
                unique_df['Chr'] = unique_df['Chr'].replace(['chrX', 'chrY'], ['chr23', 'chr24'])

                # Retrieve fetal fraction for this sample (search by original path)
                FF = sample_data[sample_data['sample_id'] == original_file_path]['FF'].iloc[0]
                
                fastq_dfs[filename] = {'DATA': unique_df, 'FF': FF}

            # Generate synthetic SCA data using combination approach
            # Y reference will be automatically set from the last sample in combination
            try:
                final_df, filename_new = make_combined_SCA(fastq_dfs, list(fastq_combination), sca_type)
                
                # Save results to output directory
                output_path = os.path.join(output_dir, filename_new)
                final_df.to_csv(output_path, index=False)
                
                # Progress logging with weighted FF calculation
                combination_str = ' + '.join(fastq_combination)
                # Calculate weighted FF using same method as negative generation
                weighted_ff = sum(fastq_dfs[f]['FF'] * fastq_dfs[f]['DATA'].shape[0] for f in fastq_combination) / sum(fastq_dfs[f]['DATA'].shape[0] for f in fastq_combination)
                print(f"Processing combination: {combination_str} -> {sca_type}")
                print(f"FF: {weighted_ff:.4f}")
                print(f"Saved file: {output_path}")
                
            except Exception as e:
                print(f"Error occurred - Combination: {fastq_combination}, SCA: {sca_type}, Error: {e}")
                continue

    print(f"\nAll SCA synthetic data generation completed!")
