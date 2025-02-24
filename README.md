# Genetic Algorithm for Data Compression Parameters

## Overview

This repository implements a genetic algorithm (GA) designed to optimize parameters for multiple data compression algorithms, including:

- Zstandard (ZSTD)
- Brotli
- LZMA
- AC2

The algorithm identifies the optimal parameter settings to achieve maximum compression efficiency for any file provided by the user.

---

## Features & Components

### Genetic Algorithm

Automatically optimizes compression parameters.

### Supported Compressors

- **Zstandard (ZSTD)**
- **Brotli**
- **LZMA**
- **AC2**

### Compressor Classes

- **BaseCompressor**: Provides common functionality for all compressors.
- Individual compressor classes (`zstd_compressor.py`, `brotli_compressor.py`, `lzma_compressor.py`, `ac2_compressor.py`): Specific implementations for each compression method.

### Parameter Configuration

- Use the provided `params.json` file to specify parameter ranges for optimization.

### Analysis and Visualization

- **plot\_maker.py**: Generates plots to visualize and analyze the results of the genetic algorithm.

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone git@github.com:TextDate/Genetical-Algorithm.git
cd Genetical-Algorithm
```

### Step 2: Create and Activate a Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux or macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Genetic Algorithm

### Basic Command

```bash
python main.py --compressor zstd --param_file params.json --input your_file.txt
```

### Available Parameters

- `--compressor`: Compression algorithm (`zstd`, `lzma`, `brotli`, `ac2`).
- `--param_file`: JSON file with parameter ranges.
- `--input`: Path to the text file to compress.
- `--generations`: Number of generations (default: 100).
- `--population_size`: Population size for each generation (default: 100).
- `--mutation_rate`: Mutation rate (default: 0.01).
- `--crossover_rate`: Crossover rate (default: 0.5).
- `--output_dir`: Directory to store results (default: `ga_results_all_files`).

### Example Usage

Optimizing Zstandard parameters for an `example.txt` file:

```bash
python main.py --compressor zstd --param_file params.json --input example.txt --generations 50 --population_size 200 --mutation_rate 0.02 --crossover_rate 0.6
```

In this example, we use the ZSTD compressor, pass the params.json file as the file with the parameters, pass the path to the file to compress, and decide to set the number of generations to 50 with 200 individuals and a 60% crossover rate and 2% mutation rate.

### Expected Output

```
Best parameters found: {'level': 22, 'window_log': 27, 'chain_log': 25, 'hash_log': 19, 'search_log': 9, 'min_match': 4, 'target_length': 16384, 'strategy': 3} with fitness 3.8765
```

This indicates the best parameter settings found and the achieved compression efficiency.

---

## Repository Structure

```
Genetical-Algorithm-main/
├── Compressors/
│   ├── base_compressor.py
│   ├── zstd_compressor.py
│   ├── brotli_compressor.py
│   ├── lzma_compressor.py
│   └── ac2_compressor.py
├── Extractors/
│   └── zstd_extractor.py
├── Plots/
│   └── plot_maker.py
├── Results/ (results on some files I had)
│   └── [decade]_ga_results/
│       └── [Compressor]_ga_results/
├── main.py
├── evaluator.py (under development; independent of GA)
├── params.json
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Contributions

This work was done under **Investigator Diogo Pratas** and **Professor Armando Pinho** as part of an *Initiation to Investigation Scholarship* on **IEETA**.
