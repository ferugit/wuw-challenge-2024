#!/bin/bash

###############################################################################################
# This script runs the evaluation of the system on the extended test set.
# It assumes that the model has been trained and the test set is available.
# In addition, it generates the evaluation results and saves them in the output directory.
# Finally, the script also evaluates the system performance and prints the results.
#
# The provided code and the following cofiguration paraters are just an example.
# You can change the code and the parameters according to your needs.
###############################################################################################

# Audio configuration
sampling_rate=16000 # Hz
time_window=1.5 # seconds
hop_size=0.256 # seconds

# WuW detection criteria
threshold=0.5 # Detection threshold
n_positives=1 # Minimum number of windows to detect a WuW event

# Output directory
output_dir="output/"
SYSID="YOUR_SYSTEM_ID"
SITE="YOUR_SITE_ID"

# Test file path
dataset_path="/home/fer/Projects/okey-aura-v1.1.0"
test_file="$dataset_path/extended_test/test-extended.tsv"

# Model path
model_path="baseline/lambda-resnet18.jit"

generate_evaluation_results=false
evaluate_system=true

# Run the test
if [ "$generate_evaluation_results" = true ] ; then
    python src/generate_evaluation_results.py \
        --sampling_rate $sampling_rate \
        --time_window $time_window \
        --hop $hop_size \
        --threshold $threshold \
        --dataset_path $dataset_path \
        --test_tsv $test_file \
        --output_dir $output_dir \
        --n_positives $n_positives \
        --sysid $SYSID \
        --model_path $model_path
fi

if [ "$evaluate_system" = true ] ; then
    python src/evaluate_system.py \
        --results_tsv $output_dir/$SYSID.tsv \
        --ref_tsv $test_file \
        --output_dir $output_dir
fi

