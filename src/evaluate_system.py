# W. Fernando López Gavilánez
# Wake-up Word Detection Challenge, Iberspeech 2024

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from tem import calculate_tem


def calculate_metrics(results_df, ref_df):
    """
    Calculate the Detect Cost Function (DCF) for the system.
    """
    cost_model = {
        'p_wuw' : 0.5,
        'c_miss' : 1,
        'c_fa' : 1.5
    }

    # Calculate misses
    predicted_labels = results_df['Label'].tolist()
    ref_labels = ref_df.replace({'WuW': 1, 'unknown': 0, 'WuW+Command': 1})['Label']
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
    p_fa = n_false_positives / (len(ref_labels) - sum(ref_labels))

    print(f"False Positives: {n_false_positives}")
    print(f"False Alarm Rate: {p_fa}")

    print(f'Number of samples: {len(predicted_labels)}')
    
    # Calculate the Detect Cost Function (DCF)
    dcf = cost_model['c_miss'] * p_miss * cost_model['p_wuw'] + cost_model['c_fa'] * p_fa * (1 - cost_model['p_wuw'])

    # Build a json with the results
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
    
    # Check if the results and reference files have the same number of rows
    if len(results_df) != len(ref_df):
        raise Exception("Results and reference files have different number of rows")

    # Check if the reference tsv has the "Filename" column else use "Sample_Path"
    if 'Filename' not in ref_df.columns:
        if 'Sample_Path' in ref_df.columns:
            ref_df = ref_df.rename(columns={'Sample_Path': 'Filename'})
            ref_df['Filename'] = ref_df['Filename'].apply(lambda x: x.split('/')[-1])
        else:
            raise Exception("Reference file has incorrect format")
    
    if not all(results_df['Filename'] == ref_df['Filename']):
        print("The filenames in the results and reference files do not match")
        print("Reordering the reference file to match the filenames in the results file")
        # Reorder them to match the filenames
        ref_df = ref_df.set_index('Filename')
        ref_df = ref_df.loc[results_df['Filename']]
        ref_df = ref_df.reset_index()

    # Calculate the Detect Cost Funtion (DCF) for the system
    results = calculate_metrics(results_df, ref_df)
    print(f"DCF: {results['dcf']}")

    # Extract DCF per 100 thresholds
    thresholds = [i/100 for i in range(101)]
    dcf_values = []
    for threshold in thresholds:
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("Evaluating threshold: ", threshold)
        results_df['Label'] = results_df['Probability'].apply(lambda x: 1 if x >= threshold else 0)
        _results = calculate_metrics(results_df, ref_df)
        print("DCF: ", _results['dcf'])
        dcf_values.append(_results['dcf'])
        print("++++++++++++++++++++++++++++++++++++++++++++")
    
    # Store DCF vs Threshold in a tsv file
    dcf_df = pd.DataFrame({'Threshold': thresholds, 'DCF': dcf_values})
    dcf_df.to_csv(os.path.join(args.output_dir, 'DCF_vs_Threshold.tsv'), sep='\t', index=False)

    # Calculate the Time Error Metric (TEM)
    tem_results = calculate_tem(results_df, ref_df)
    print(f"Time Error Metric (TEM): {tem_results}")

    # Get the minimum DCF value
    min_dcf = min(dcf_values)

    # Save the results to a file: json parsing
    results['tem'] = tem_results
    results['min_dcf'] = min_dcf

    # Save the results to a file: json parsing
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot the DCF values
    plt.plot(thresholds, dcf_values)
    plt.xlabel('Threshold')
    plt.ylabel('DCF')
    plt.title('DCF vs Threshold')
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'DCF_vs_Threshold.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a system for the wake-up word detection challenge')
    parser.add_argument('--results_tsv', required=True, help='tsv file containing the results of the system')
    parser.add_argument('--ref_tsv', required=True, help='tsv file containing the reference labels')
    parser.add_argument('--output_dir', required=True, help='output path')
    args = parser.parse_args()

    main(args)

