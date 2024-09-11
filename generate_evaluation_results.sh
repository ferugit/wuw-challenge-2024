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
threshold=0.8 # Detection threshold
n_positives=2 # Minimum number of windows to detect a WuW event

# Output directory
output_dir="output_eval/"
SYSID="baseline"
SITE="baseline"

# Make the output directory if it does not exist
mkdir -p $output_dir

# Test file path
dataset_path="/home/fer/Projects/ok-aura-dataset/wuwdc2024/extended/wuwdc2024-eval"
test_file="$dataset_path/wuwdc2024-eval.tsv"
ref_file="/home/fer/Projects/ok-aura-dataset/wuwdc2024/extended/wuwdc2024-eval_w_anonymization.tsv"

# Model path
model_path="baseline/lambda-resnet18.jit"

generate_evaluation_results=true

# Run the test
if [ "$generate_evaluation_results" = true ] ; then
    echo "\nGenerating evaluation results..."
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

