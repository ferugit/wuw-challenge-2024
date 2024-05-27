# 2024 Wake-Up Word Detection Challenge (WUWDC)

Organized by Telefónica Innovación Digital, this challenge aims to assess the performance of State-of-The-Art Keyword Spotting systems in addressing various industrial needs such as accuracy, inference delay, computational load, and energy efficiency. Further information of the challenge can be found [here](https://catedrartve.unizar.es/wuwdc2024.html)


This repository contains the baseline system for the challenge and a simple script to evaluate models.


## Environment setup

First of all, install all dependences. Open a command line and run the next lines:

``` bash
# create a virtual environment
python3 -m venv .venv
# activate the virtual environment
source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
```

## Streaming Test Script Explanation

The `run_streaming_test.sh` script is used to evaluate the system on an extended test set. It assumes that the model has been trained and the test set is available. The script generates evaluation results and saves them in the output directory. It also evaluates the system performance and prints the results.

Here's a breakdown of the script:

### Configuration Parameters

- **Audio Configuration**
  - `sampling_rate`: The sampling rate of the audio in Hz. Set to 16000 Hz by default.
  - `time_window`: The time window for the audio in seconds. Set to 1.5 seconds by default.
  - `hop_size`: The hop size for the audio in seconds. Set to 0.256 seconds by default.

- **Wake-Up-Word (WuW) Detection Criteria**
  - `threshold`: The threshold for detection. Set to 0.5 by default.
  - `n_positives`: The minimum number of windows to detect a WuW event. Set to 2 by default.

- **Output Directory**
  - `output_dir`: The directory where the evaluation results will be saved. Set to "output/evaluation_results" by default.
  - `SYSID`: The system identifier. Set to "YOUR_SYSTEM_ID" by default.
  - `SITE`: The site identifier. Set to "YOUR_ORGANIZATION_ID" by default.

- **Test File Path**
  - `dataset_path`: The path to the dataset. Set to "../okey-aura-v1.1.0" by default.
  - `test_tsv`: The path to the test TSV file. Set to "$dataset_path/extended_test/test-extended.tsv" by default.

- **Model Path**
- `model_path`: The path to the trained model. Set to "output/models/your_model.pth" by default.

### Running the Test

The script generates the results by executing the `src/generate_evaluation_results.py` Python script with the above configuration parameters.
