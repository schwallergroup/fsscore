#!/bin/bash

# Script for:
# 1. processing both datasets
# 2. combine it and filter
# 3. split it into training and validation sets and format for model input

# Set up directories
ROOT_DIR=$(dirname $(pwd))
CWD=$(pwd)
# print out root directory
DATA_DIR=$ROOT_DIR/data

# make directories if they don't exist
mkdir -p $DATA_DIR

# 1. Process USPTO data (takes from URL)
python $CWD/uspto_processing.py --data_dir $DATA_DIR

# 2. Process data extracted from chemical journals
# comment out if don't want to include this data
CHEMPAPERS_FILE="path/to/utf8_file"
# will just exit if file not found
python $CWD/chempapers_processing.py --data_dir $DATA_DIR --input_file $CHEMPAPERS_FILE

# 3. Combine and filter data
USPTO_PROC=$DATA_DIR/uspto_raw/uspto_split_combo_fil_deloop.csv
CP_PROC=$DATA_DIR/chempapers_raw/chempapers_preproc3_deloop.csv
python $CWD/combine_datasets.py --uspto_file $USPTO_PROC --chempapers_file $CP_PROC --output_dir $DATA_DIR
