# KOShapleyValueForCRNs

Command-line toolkit for SBML model simulation and perturbation analysis:

- knockout and knockin model edits (species/reactions)
- Shapley-style importance assessment
- Sobol-based sensitivity analysis

The main CLI entry point is in `src/mainV2.py`.

## Table of contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Output structure](#output-structure)
- [Examples](#examples)
- [Tips and troubleshooting](#tips-and-troubleshooting)

## Overview

This project lets you load an SBML model and run one of several workflows:

- `simulate`: run time-course simulation and export CSV/plots
- `importance_assessment`: quantify species importance with knockout/knockin and optional perturbations
- `sensitivity_analysis`: run Sobol sensitivity analysis
- `knockout_species`: save a new model with one species knocked out
- `knockout_reaction`: save a new model with one reaction knocked out
- `knockin_species`: save a new model with one species knocked in
- `knockin_reaction`: save a new model with one reaction knocked in

## Requirements

Python 3.9+ is recommended.

Install dependencies:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `python-libsbml`
- `libroadrunner`
- `SALib`
- `matplotlib`
- `seaborn`
- `plotly`

Example install:

```bash
python -m pip install \
  numpy pandas scipy scikit-learn python-libsbml libroadrunner \
  SALib matplotlib seaborn plotly
```

## Quickstart

From the repository root:

```bash
python -m src.mainV2 -h
```

Run a simulation:

```bash
python -m src.mainV2 simulate models/BIOMD0000000623_url.xml -t 120 -o results
```

Inspect command-specific options:

```bash
python -m src.mainV2 simulate -h
python -m src.mainV2 importance_assessment -h
python -m src.mainV2 sensitivity_analysis -h
```

## Commands

General form:

```bash
python -m src.mainV2 <command> [options]
```

Available commands:

- `simulate`
- `importance_assessment`
- `sensitivity_analysis`
- `knockout_species`
- `knockout_reaction`
- `knockin_species`
- `knockin_reaction`

## Output structure

By default, outputs are written under `./results` in a model-specific folder:

```text
<output>/<model_name>/
├── csv/
├── images/
└── reports/
```

## Examples

### 1) Simulate a model (static plot + CSV)

```bash
python -m src.mainV2 simulate models/BIOMD0000000623_url.xml \
  -t 120 \
  -i cvode \
  -o results
```

### 2) Simulate until steady state (interactive HTML plot)

```bash
python -m src.mainV2 simulate models/BIOMD0000000623_url.xml \
  --steady-state \
  --max-time 2000 \
  --sim-step 10 \
  --threshold 1e-7 \
  --interactive \
  -o results
```

### 3) Importance assessment (knockout, no perturbations)

```bash
python -m src.mainV2 importance_assessment models/BIOMD0000000623_url.xml \
  --operation knockout \
  --payoff-function last \
  -t 120 \
  -o results
```

### 4) Importance assessment with random perturbations

```bash
python -m src.mainV2 importance_assessment models/BIOMD0000000623_url.xml \
  --operation knockout \
  --input-species S1 S2 \
  --use-perturbations \
  --num-samples 10 \
  --max-combinations 2000 \
  --variation 20 \
  --payoff-function max \
  -t 120 \
  -o results
```

### 5) Importance assessment with fixed perturbations

```bash
python -m src.mainV2 importance_assessment models/BIOMD0000000623_url.xml \
  --operation knockin \
  --input-species S1 S2 \
  --use-perturbations \
  --use-fixed-perturbations \
  --fixed-perturbations -20 20 \
  --payoff-function min \
  -t 120 \
  -o results
```

### 6) Sensitivity analysis (Sobol)

```bash
python -m src.mainV2 sensitivity_analysis models/BIOMD0000000623_url.xml \
  --input-species S1 S2 \
  --base-samples 1024 \
  --perturbation-range 20 \
  --operation knockout \
  -o results
```

### 7) Sensitivity analysis with convergence check

```bash
python -m src.mainV2 sensitivity_analysis models/BIOMD0000000623_url.xml \
  --input-species S1 S2 \
  --check-convergence \
  -o results
```

### 8) Knock out one species and save edited model

```bash
python -m src.mainV2 knockout_species models/BIOMD0000000623_url.xml S1 \
  --model-dir models \
  -o results
```

### 9) Knock out one reaction and save edited model

```bash
python -m src.mainV2 knockout_reaction models/BIOMD0000000623_url.xml R1 \
  --model-dir models \
  -o results
```

### 10) Knock in one species and save edited model

```bash
python -m src.mainV2 knockin_species models/BIOMD0000000623_url.xml S1 \
  --model-dir models \
  -o results
```

### 11) Knock in one reaction and save edited model

```bash
python -m src.mainV2 knockin_reaction models/BIOMD0000000623_url.xml R1 \
  --model-dir models \
  -o results
```

## Tips and troubleshooting

- Replace placeholder IDs (`S1`, `S2`, `R1`) with IDs that actually exist in your SBML file.
- Start with models in `models/` to validate your setup.
- Add logging to any command with `-l <log_file_path>`.
- For large perturbation spaces, use `--max-combinations` to cap Cartesian-product runs and avoid RAM saturation.
- If simulation fails immediately, first run `simulate -h` and verify required options and valid integrator/model values.

## Project layout

- `src/mainV2.py`: CLI entry point and pipeline orchestration
- `models/`: sample SBML models
- `results/`: default output directory
