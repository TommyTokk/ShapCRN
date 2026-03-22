# KOShapleyValueForCRNs

KOShapleyValueForCRNs is a command-line application for studying SBML biochemical reaction-network models through simulation and controlled perturbations. It is designed to support both exploratory analysis and reproducible experiments: you can run model dynamics over time, inspect behavior near steady state, generate publication-friendly outputs (CSV and plots), and compare how system behavior changes when species or reactions are altered.

Beyond plain simulation, the project provides analysis pipelines to quantify influence and robustness at network level. In practice, this includes knockout/knockin workflows, Shapley-style importance assessment (with optional random or fixed perturbation scenarios), and Sobol-based global sensitivity analysis for selected targets. The file `src/mainV2.py` serves as the reference CLI example that orchestrates these capabilities end to end.

## Table of contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Output structure](#output-structure)
- [Examples](#examples)
- [Tips and troubleshooting](#tips-and-troubleshooting)
- [Project layout](#project-layout)

## Overview

This project lets you load an SBML model and run one of several workflows:

- `simulate`: run time-course simulation with selectable integrators, optional steady-state mode, and CSV/plot exports.
- `importance_assessment`: estimate influence of species/reactions with knockout or knockin scenarios using a Shapley-style workflow.
- `importance_assessment` with perturbations: compare behavior under random or fixed perturbations of selected input species.
- `sensitivity_analysis`: compute Sobol indices for selected targets and optionally run convergence checks.
- `knockout_species`: create and save a modified SBML model where one species is disabled.
- `knockout_reaction`: create and save a modified SBML model where one reaction is disabled.
- `knockin_species`: create and save a modified SBML model where one species is reinforced/activated.
- `knockin_reaction`: create and save a modified SBML model where one reaction is reinforced/activated.

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
python -m src.mainV2 simulate models/test.xml -t 120 -o results
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
- `create_network` (currently a placeholder in `mainV2.py`)

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
python -m src.mainV2 simulate models/KnockinModelV2.xml \
  -t 120 \
  -i cvode \
  -o results
```

### 2) Simulate until steady state (interactive HTML plot)

```bash
python -m src.mainV2 simulate models/KnockinModelV2.xml \
  --steady-state \
  --max-time 2000 \
  --sim-step 10 \
  --threshold 1e-7 \
  --interactive \
  -o results
```

### 3) Importance assessment (knockout, no perturbations)

```bash
python -m src.mainV2 importance_assessment models/KnockinModelV2.xml \
  --operation knockout \
  --payoff-function last \
  -t 120 \
  -o results
```

### 4) Importance assessment with random perturbations

```bash
python -m src.mainV2 importance_assessment models/KnockinModelV2.xml \
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
python -m src.mainV2 importance_assessment models/KnockinModelV2.xml \
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
python -m src.mainV2 sensitivity_analysis models/KnockinModelV2.xml \
  --input-species S1 S2 \
  --base-samples 1024 \
  --perturbation-range 20 \
  --operation knockout \
  -o results
```

### 7) Sensitivity analysis with convergence check

```bash
python -m src.mainV2 sensitivity_analysis models/KnockinModelV2.xml \
  --input-species S1 S2 \
  --check-convergence \
  -o results
```

### 8) Knock out one species and save edited model

```bash
python -m src.mainV2 knockout_species models/KnockinModelV2.xml S1 \
  --model-dir models \
  -o results
```

### 9) Knock out one reaction and save edited model

```bash
python -m src.mainV2 knockout_reaction models/KnockinModelV2.xml R1_MassAction_Explicit \
  --model-dir models \
  -o results
```

### 10) Knock in one species and save edited model

```bash
python -m src.mainV2 knockin_species models/KnockinModelV2.xml S1 \
  --model-dir models \
  -o results
```

### 11) Knock in one reaction and save edited model

```bash
python -m src.mainV2 knockin_reaction models/KnockinModelV2.xml R1_MassAction_Explicit \
  --model-dir models \
  -o results
```

## Tips and troubleshooting

- `models/KnockinModelV2.xml` works with the IDs used in these examples (`S1`, `S2`, `R1_MassAction_Explicit`).
- Start with models in `models/` to validate your setup.
- Add logging to any command with `-l <log_file_path>`.
- For large perturbation spaces, use `--max-combinations` to cap Cartesian-product runs and avoid RAM saturation.
- If simulation fails immediately, first run `simulate -h` and verify required options and valid integrator/model values.

## Project layout

- `src/mainV2.py`: reference CLI example and pipeline orchestration
- `models/`: sample SBML models
- `results/`: default output directory
