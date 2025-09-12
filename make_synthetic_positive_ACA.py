#!/home/albatross/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Synthetic Positive ACA (Autosomal Chromosome Aneuploidies) Data Generator

This module implements the synthetic positive data generation methodology described in:
"Synthetic data-drive AI approach for fetal chromosomal aneuploidies detection"

Key Features:
- Implements Equation (1) from the paper: T_iRC = C_iRC + (C_iRC × FF) / 2
- Applies systematic sample combination approach from synthetic negative generation
- Uses weighted average FF calculation based on unique read counts as weights
- Generates synthetic trisomy data for chromosomes 1-22 (autosomes)
- Uses internal ACA reference from last sample in each combination

Variables:
- TRC: Total Read Count across all chromosomes
- FF_snp: SNP-based Fetal Fraction (as percentage)
- T_iRC: Read count of target chromosome i in presence of trisomy
- C_iRC: Read count of chromosome i in normal fetus
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


def make_combined_ACA(fastq_dfs, filenames):
    """
    Generate synthetic positive ACA data by combining multiple samples
    
    This function implements the paper's methodology for generating synthetic positive data:
    1. Combines multiple samples using weighted averaging (identical to negative generation)
    2. Weights each sample's contribution by its unique read count (UR)
    3. Sets up ACA reference using the last sample in the combination
    4. Applies ACA transformation using Equation (1): T_iRC = C_iRC + (C_iRC × FF) / 2
    
    The weighted FF calculation follows the paper's formula:
    Cff = (M1 × M1ff + ... + Mi × Miff + ... + Mn × Mnff) / (M1 + ... + Mi + ... + Mn)
    
    ACA reference strategy:
    - Uses the last sample in the combination as ACA reference source
    - Eliminates dependency on external reference files or FF-based selection
    - Ensures consistency and reproducibility across different environments
    
    Parameters
    ----------
    fastq_dfs : dict
        Dictionary containing sample data and FF values
        { filename: {"DATA": DataFrame, "FF": float} }
        where DATA contains unique mapped reads with columns: Chr, Pos, Seq
    filenames : list
        List of sample filenames to combine for synthetic generation
        
    Returns
    -------
    tuple
        (final_df, filename_new) - processed dataframe and output filename
        final_df contains both theoretical T values and actual chr counts
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
    
    # Step 2: Set up ACA reference using last sample in combination
    # Instead of using FF-based selection, use the last sample as ACA reference source
    # This ensures consistency and eliminates dependency on FF ranking
    aca_ref_sample = filenames[-1]  # Use last sample in combination as ACA reference
    ACA_ref_df = fastq_dfs[aca_ref_sample]['DATA'].copy()
    
    # Step 3: Apply ACA transformation using paper's Equation (1)
    # Calculate GC content for quality control
    base_sample_df["GC"] = base_sample_df["Seq"].apply(lambda x: x.count("G") + x.count("C"))
    base_sample_df["Total"] = base_sample_df["Seq"].apply(len)
    Sample_GC = base_sample_df["GC"].sum() / base_sample_df["Total"].sum()
    
    UR = base_sample_df.shape[0]  # Total Unique Read Count (TRC in the paper)
    ATRC = base_sample_df[(base_sample_df['Chr'] != 'chr23') & (base_sample_df['Chr'] != 'chr24')].shape[0]

    T_iRC_list = []  # Store theoretical trisomy read counts
    S_ACA_df = base_sample_df.copy()  # Initialize synthetic ACA dataframe

    # Apply trisomy transformation to each autosome (chromosomes 1-22)
    # Implements paper's Equation (1): T_iRC = C_iRC + (C_iRC × FF) / 2
    for i in range(1, 23):  # autosomes only
        C_iRC = len(base_sample_df[base_sample_df['Chr'] == f'chr{i}'])  # Normal chromosome read count
        
        # Calculate additional reads needed for trisomy using Equation (1)
        # delta_T_iRC represents the additional fetal reads: (C_iRC × FF) / 2
        delta_T_iRC = int(C_iRC * (FF_new * 0.01) * 0.5)
        
        # Calculate theoretical trisomy read count (T_iRC) for output
        T_iRC = C_iRC + delta_T_iRC
        T_iRC_list.append(T_iRC)
        
        # Sample additional trisomy reads from ACA reference data
        aca_chr_reads = ACA_ref_df[ACA_ref_df['Chr'] == f'chr{i}']
        if len(aca_chr_reads) >= delta_T_iRC:
            add_df = aca_chr_reads.sample(delta_T_iRC)
            S_ACA_df = pd.concat([S_ACA_df, add_df], axis=0, ignore_index=True)
        else:
            # Handle cases where reference data is insufficient
            if len(aca_chr_reads) > 0:
                S_ACA_df = pd.concat([S_ACA_df, aca_chr_reads], axis=0, ignore_index=True)

    # Step 5: Calculate final chromosome read counts (original counts before trisomy addition)
    chr_read_counts = []
    for i in range(1, 25):
        # Use original base sample counts (before trisomy addition) for chr columns
        # This maintains distinction between theoretical T values and actual chr counts
        chr_count = len(base_sample_df[base_sample_df['Chr'] == f'chr{i}'])
        chr_read_counts.append(chr_count)

    # Step 6: Generate output dataframe and filename
    chr_col_names = [f'chr{i}' for i in range(1, 25)]  # Actual chromosome read counts
    ACA_UR_col = [f"T{i}_UR" for i in range(1, 23)]    # Theoretical trisomy read counts
    
    # Create descriptive filename including sample combination and weighted FF
    combined_names = '_'.join(filenames)
    filename_new = f'SynACA_{combined_names}_{FF_new:.2f}.csv'

    # Final dataframe contains both theoretical T values and actual chromosome counts
    # T columns: theoretical values from Equation (1)
    # chr columns: actual read counts from combined base sample
    final_col_names = ['sample_id', 'result', 'GC', 'UR', 'snp_FF'] + ACA_UR_col + chr_col_names
    final_df = pd.DataFrame(
        np.array([filename_new, 'ACA', Sample_GC, UR, FF_new] + T_iRC_list + chr_read_counts)
    ).T
    final_df.columns = final_col_names
    
    return final_df, filename_new




if __name__ == "__main__":
    # Set output directory for synthetic ACA data
    output_dir = './synthetic_positive_ACA'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Load per-sample FF values from CSV file
    sample_data = pd.read_csv('Sample_FF_info.csv')
    file_paths = sample_data['sample_id'].tolist()  # All file paths
    filenames = [get_filename_from_path(path) for path in file_paths]  # Extract filenames only
    
    all_combinations = []
    
    # Generate all combinations of 2..N files for ACA generation
    # This creates comprehensive synthetic dataset covering all possible sample combinations
    for comb_number in range(2, len(filenames)+1):
        all_combinations += list(combinations(filenames, comb_number))

    # Create dictionary mapping filenames to their full paths
    filename_to_path = {get_filename_from_path(path): path for path in file_paths}
    
    # Process each combination to produce synthetic ACA samples
    combination_desc = f"Processing ACA combinations"
    for fastq_combination in tqdm(all_combinations, desc=combination_desc):
        fastq_dfs = {}
        
        # Load data for each file in the combination
        for filename in fastq_combination:
            # Get original file path
            original_file_path = filename_to_path[filename]
            
            # Create unique FASTQ file path in the same directory as original file
            original_dir = os.path.dirname(original_file_path)
            unique_fastq_path = os.path.join(original_dir, f"{filename}.Fastq.mapped")
            
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

        # Generate synthetic ACA data using combination approach
        final_df, filename_new = make_combined_ACA(fastq_dfs, list(fastq_combination))
        
        # Save results to output directory
        output_path = os.path.join(output_dir, filename_new)
        final_df.to_csv(output_path, index=False)
        
        # Progress logging with weighted FF calculation
        combination_str = ' + '.join(fastq_combination)
        # Calculate weighted FF using same method as negative generation
        weighted_ff = sum(fastq_dfs[f]['FF'] * fastq_dfs[f]['DATA'].shape[0] for f in fastq_combination) / sum(fastq_dfs[f]['DATA'].shape[0] for f in fastq_combination)
        print(f"Processing combination: {combination_str}")
        print(f"FF: {weighted_ff:.4f}")
        print(f"Saved file: {output_path}")
