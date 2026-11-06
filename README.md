# Few-Shot Learning for Time Series Classification Using a Tabular Foundation Model

Slides (PT-BR):

## Description

Labeling time series demands effort and often requires expert annotation, which makes classification with few labeled examples (few-shot) a difficult and realistic scenario. Tabular foundation models, such as TabPFN, perform well in this setting on tabular data, but their use with time series remains little explored. This work investigates TS2TabPFN, a framework that extracts features from time series and classifies them with TabPFN, updated to version 3.5. First, versions 2.5 and 3.5 are compared with the full training set and three feature extractors. Then, the framework is evaluated in few-shot, varying the number of examples per class and the strategy used to select them, and is compared with simpler classifiers under the same conditions, with methods from the literature, and in terms of computational cost. With ten examples per class, TS2TabPFN retains about 90% of the accuracy obtained with the full training set, but performance drops quickly with fewer examples. Selection by k-medoids, which requires no labels, improves performance in one of the two collections evaluated. The work provides a systematic evaluation of TabPFN in few-shot time series classification, with public code that allows it to be reproduced.

This work builds on [TS2TabPFN](https://github.com/gabrielcmerlin/TS2TabPFN).

## Structure

```bash
FewShot-TS2TabPFN/
│
├── data/                         # Dataset download scripts
│   ├── download_mv_core.py       # Script for downloading Multivariate collection
│   └── download_redux.py         # Script for downloading Univariate collection
│
├── exp/                          # Experiments
│   ├── few/                      # TabPFN 3.5 used on the selected support sets
│   ├── few_baseline/             # Ridge and 1NN used on the same support sets
│   └── full/                     # TabPFN 2.5 / 3.5 used on the full training set
│
├── features/                     # Feature extraction
│   ├── logs/                     # Extraction time and energy logs
│   ├── config.yaml               # Benchmark and datasets to extract
│   └── extract_*.py              # Scripts for extracting features from time series
│
├── instance_selection/           # Few-shot support set selection
│   ├── config.yaml               # Feature, benchmark, datasets and K values
│   └── main.py                   # Script for selecting few-shot support sets
│
├── outputs/                      # Raw experiment outputs
│   ├── few/                      # Few-shot outputs, by selection strategy and K
│   ├── few_baseline/             # Baseline outputs, by model, selection strategy and K
│   ├── full/                     # Full training outputs (accuracy, time, energy, GPU)
│   ├── convert_few.py            # Script for converting few-shot outputs into results/few
│   └── convert_full.py           # Script for converting full training outputs into results/full
│
├── results/                      # Results formatted for comparison
│   ├── few/                      # One CSV per method, per selection strategy and K
│   ├── full/                     # One CSV per method, per collection
│   └── thesis/
│       ├── figs_thesis/          # Figures, by thesis section
│       └── plot_thesis.ipynb     # Generates every figure in the thesis
│
├── utils/                        # Helpers (config parsing, data loading, training, selection)
│
├── .gitignore
├── env.yaml                      # Conda environment definition for reproducibility
└── README.md                     # This file
```

## How to start

### Requirements

- Linux with an NVIDIA GPU and CUDA (needed by TabPFN)
- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda

### Downloading code

```bash
git clone https://github.com/gabrielcmerlin/FewShot-TS2TabPFN.git
cd FewShot-TS2TabPFN
```

### Installing requisites

```bash
conda env create -f env.yaml
conda activate thesis
```

### Setting up TabPFN

TabPFN 3.5 requires a Prior Labs API key:

1. Log in (or register) at [ux.priorlabs.ai](https://ux.priorlabs.ai).
2. Accept the license on the **Licenses** tab.
3. Copy your API key from [ux.priorlabs.ai/account](https://ux.priorlabs.ai/account).
4. Create a `.env` file in the repository root with:

```bash
TABPFN_TOKEN="<your-api-key>"
```

The experiment scripts load this file automatically.

## Run

Each step reads its own `config.yaml`, where you choose the benchmark and the datasets. Update it before running the step.

### 1. Download data

```bash
cd data
python3 download_redux.py
python3 download_mv_core.py
cd ..
```

### 2. Extract features

Features are extracted once and cached in `features/feat/`, so every experiment reuses them.

```bash
# Change * for {c22, tsfresh, mr}
cd features
python3 extract_*.py -c config.yaml
cd ..
```

### 3. Select few-shot support sets

Selects K examples per class (K ∈ {1, 3, 5, 10}) with each selection strategy and saves them in `instance_selection/instances/`.

```bash
cd instance_selection
python3 main.py -c config.yaml
cd ..
```

### 4. Run experiments

```bash
# TabPFN with the full training set
cd exp/full
python3 main.py -c config.yaml
cd ../..

# TabPFN in few-shot
cd exp/few
python3 main.py -c config.yaml
cd ../..

# Ridge and 1NN in few-shot (set `model` in the config)
cd exp/few_baseline
python3 main.py -c config.yaml
cd ../..
```

### 5. Prepare outputs

The experiments generate raw outputs that must be formatted before the analysis:

```bash
cd outputs
python3 convert_full.py
python3 convert_few.py
cd ..
```

### 6. Result analysis

Run the notebook `results/thesis/plot_thesis.ipynb`. It generates every figure in the thesis in `results/thesis/figs_thesis/`.