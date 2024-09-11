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

## Results genaration

The `generate_system_results.sh` script is used to generate the system's reulsts on an extended test set. It assumes that the model has been trained and the test set is available. The script generates evaluation results and saves them in the output directory.

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

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2101.12732,
  doi = {10.48550/ARXIV.2101.12732},
  url = {https://arxiv.org/abs/2101.12732},
  author = {Bonet, David and Cámbara, Guillermo and López, Fernando and Gómez, Pablo and Segura, Carlos and Luque, Jordi},
  title = {Speech Enhancement for Wake-Up-Word detection in Voice Assistants},
  publisher = {arXiv},
  year = {2021},
}

@article{cambara2022tase,
  title={TASE: Task-Aware Speech Enhancement for Wake-Up Word Detection in Voice Assistants},
  author={C{\'a}mbara, Guillermo and L{\'o}pez, Fernando and Bonet, David and G{\'o}mez, Pablo and Segura, Carlos and Farr{\'u}s, Mireia and Luque, Jordi},
  journal={Applied Sciences},
  volume={12},
  number={4},
  pages={1974},
  year={2022},
  publisher={MDPI}
}

```




