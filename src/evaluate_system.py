# W. Fernando López Gavilánez
# Wake-up Word Detection Challenge, Iberspeech 2024

import os
import json
import argparse
import pandas as pd

def calculate_metrics(results_df, ref_df):
    """
    Calculate the Detect Cost Function (DCF) for the system.
    """
    cost_model = {
        'p_wuw' : 0.1,
        'c_miss' : 1,
        'c_fa' : 10
    }

    # Calculate misses
    predicted_labels = results_df['Label'].tolist()
    ref_labels = ref_df.replace({'WuW': 1, 'unknown': 0})['Label']
    n_misses = 0

    for i in range(len(ref_labels)):
        if ref_labels[i] == 1 and predicted_labels[i] == 0:
            n_misses += 1
    p_miss = n_misses / sum(ref_labels)

    print(f"Misses: {n_misses}")
    print(f"Miss Rate: {p_miss}")

    # calculate the false alarms
    n_false_positives = 0
    for i in range(len(ref_labels)):
        if ref_labels[i] == 0 and predicted_labels[i] == 1:
            n_false_positives += 1
    p_fa = n_false_positives / (len(predicted_labels) - sum(ref_labels))

    print(f"False Positives: {n_false_positives}")
    print(f"False Alarm Rate: {p_fa}")

    print(f'Number of samples: {len(predicted_labels)}')
    
    # calculate the cost of false positives and false negatives
    dcf = cost_model['c_miss'] * p_miss + cost_model['c_fa'] * p_fa

    # build a json with the results
    results = {
        'misses': n_misses,
        'false_positives': n_false_positives,
        'n_samples': len(predicted_labels),
        'p_miss': p_miss,
        'p_fa': p_fa,
        'dcf': dcf
    }

    return results


def main(args):

    # check if the results file exists
    if not os.path.exists(args.results_tsv):
        raise Exception("Results file not found")
    
    # check if the reference file exists
    if not os.path.exists(args.ref_tsv):
        raise Exception("Reference file not found")
    
    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        raise Exception("Output directory not found")
    
    # read the results and refence files
    results_df = pd.read_csv(args.results_tsv, header=0, sep='\t')
    ref_df = pd.read_csv(args.ref_tsv, header=0, sep='\t')
    
    # check if the results file has the expected columns
    if not all(col in results_df.columns for col in ["Filename", "Probability", "Label", "Start_Time", "End_Time"]):
        raise Exception("Results file has incorrect format")
    
    # TODO: check if the results file has the expected number of rows

    # Calculate the Detect Cost Funtion (DCF) for the system
    results = calculate_metrics(results_df, ref_df)

    print(f"DCF: {results['dcf']}")

    # Save the results to a file: json parsing
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a system for the wake-up word detection challenge')
    parser.add_argument('--results_tsv', required=True, help='tsv file containing the results of the system')
    parser.add_argument('--ref_tsv', required=True, help='tsv file containing the reference labels')
    parser.add_argument('--output_dir', required=True, help='output path')
    args = parser.parse_args()

    main(args)

