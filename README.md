# KOShapleyValueForCRNs

<p align="right">
  <br><br>
  <br><br>
  There is no learning without having to pose a question.<br>
  And a question requires doubt.<br>
  — <b>Richard Feynman</b>
  <br><br>
  <br><br>
</p>


## Introduction

KOShapleyValueForCRNs is a command-line application for studying SBML biochemical reaction-network models through simulation and controlled perturbations. It is designed to support both exploratory analysis and reproducible experiments: you can run model dynamics over time, inspect behavior near steady state, generate publication-friendly outputs (CSV and plots), and compare how system behavior changes when species or reactions are altered.

Beyond plain simulation, the project provides analysis pipelines to quantify influence and robustness at network level. In practice, this includes knockout/knockin workflows, Shapley-style importance assessment (with optional random or fixed perturbation scenarios), and Sobol-based global sensitivity analysis for selected targets. The file `src/examples/mainV2.py` is an example runner that shows how to orchestrate these capabilities end to end; it is not the project entry point.

## Table of contents

- [Overview](#overview)
- [Architecture and code map](#architecture-and-code-map)
- [Main functionalities](#main-functionalities)
- [Additional functionalities](#additional-functionalities)
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

In a typical run, the flow is:

1. Parse command-line arguments (`src/utils/utils.py`).
2. Load and prepare SBML (`src/utils/sbml/io.py` + `src/utils/sbml/reactions.py`).
3. Dispatch to a pipeline (`src/pipelines/...`).
4. Run simulations/analysis (`src/utils/simulation.py`, `src/utils/sensitivity.py`).
5. Save artifacts (CSV, plots, reports, edited SBML).
6. Return logs and outputs under the selected output folder.

## Architecture and code map

The codebase follows a layered structure:

- Example runner layer:
  `src/examples/mainV2.py` demonstrates how to wire commands to pipelines.
  It is intentionally an example launcher, not a canonical package entry point.
- Pipeline layer:
  `src/pipelines/*` contains command-oriented orchestration.
  Each pipeline parses command-specific arguments, coordinates utilities, and writes outputs.
- Utility layer:
  `src/utils/*` contains reusable logic split by domain:
  - `src/utils/sbml/`: SBML I/O, reaction preprocessing, and knock operations.
  - `src/utils/simulation.py`: RoadRunner setup, simulation, perturbation sampling and aggregation helpers.
  - `src/utils/sensitivity.py`: Sobol setup/execution, convergence checks, and statistics.
  - `src/utils/plot.py`: static and interactive plotting utilities.
  - `src/utils/graph.py`: model-to-network conversion and graph rendering.
  - `src/utils/utils.py`: CLI parser construction, normalization helpers, logging, and shared helpers.

Design intent:

- Keep pipelines thin and scenario-focused.
- Keep model/math/plot logic reusable in `utils`.
- Keep output organization consistent across commands (`images/`, `csv/`, `reports/`, `dot/`).

## Main functionalities

This section describes the core capabilities and maps them to the main functions in the codebase.

### 1) Simulate

Main execution path:

- Example runner: `src/examples/mainV2.py` (command `simulate`)
- Model loading and normalization: `src/utils/sbml/io.py::load_and_prepare_model`
- RoadRunner setup: `src/utils/simulation.py::load_roadrunner_model`
- Simulation engine: `src/utils/simulation.py::simulate`
- Steady-state mode (if enabled): `src/utils/simulation.py::simulate_with_steady_state`
- Output plots: `src/utils/plot.py::plot_results` and `plot_results_interactive`

How it works:

- `load_roadrunner_model(...)` converts the SBML model to a RoadRunner instance and configures integrator/tolerances.
- `simulate(...)` supports two modes:
  - Standard mode: one run from `start_time` to `end_time`.
  - Steady-state mode: adaptive block simulation until variation is below threshold.
- `simulate_with_steady_state(...)` compares the last point of consecutive blocks and tracks relative/absolute variation per monitored species. When all monitored species stay below threshold for consecutive checks, it flags steady state.
- Results are exported to CSV and plotted (static PNG or interactive HTML).

### 2) Species/Reactions Knockout

Main execution path:

- Species KO pipeline: `src/pipelines/knockout/knockout_species.py::knockout_species`
- Reaction KO pipeline: `src/pipelines/knockout/knockout_reaction.py::knockout_reaction`
- Core KO logic: `src/utils/sbml/knock.py::knockout_species` and `knockout_reaction`

How species knockout works (`knockout_species` in `knock.py`):

- Forces assignment/initial rules for the target species to `0` when present.
- Scans events and initial assignments and sets the target species update math to `0`.
- If the species is a reactant in a reaction, that reaction is marked for knockout.
- If the species is only a product, the product entry is removed from that reaction; if no products remain, the reaction is also knocked out.
- Finalizes by setting target species initial concentration to `0.0` and boundary condition `True` (fixed species).

How reaction knockout works (`knockout_reaction` in `knock.py`):

- Looks up the target reaction.
- Replaces the kinetic law AST with constant `0`, disabling flux while preserving the reaction object in the SBML structure.

### 3) Species/Reactions Knockin

Main execution path:

- Species KI pipeline: `src/pipelines/knockin/knockin_species.py::knockin_species`
- Reaction KI pipeline: `src/pipelines/knockin/knockin_reaction.py::knockin_reaction`
- Core KI logic: `src/utils/sbml/knock.py::knockin_species` and `knockin_reaction`

How species knockin works:

- `get_species_peak_value(...)` runs a short simulation (`end_time=60`) and uses the target species maximum simulated value as the knock-in value.
- `knockin_species(...)` then sets that value as initial concentration/amount (depending on species representation) and marks the species as fixed (`boundaryCondition=True`, `constant=True`).

How reaction knockin works:

- `get_reactants_peak_values(...)` collects max simulated values for each reactant of the target reaction.
- `knockin_reaction(...)` creates constant reactant copies with `_KI` suffix, one per original reactant.
- The target reaction is cloned, its reactants are replaced with these new constant species, and the kinetic law expression is rewritten to reference `_KI` species.
- The original reaction is removed and the modified cloned reaction is added back to the model.

## Additional functionalities

### Importance assessment

Main path:

- Pipeline: `src/pipelines/importance.py::importance_assessment`
- Model setup: `model_preparation(...)`
- Sampling setup: `generate_samples(...)`
- Baseline simulation: `simulate_original_model(...)`
- KO/KI simulation batch: `simulate_knocked_data(...)`
- Payoff/Shapley: `run_shap_analysis(...)`
- Perturbation diagnostics: `assess_perturbation_importance(...)` and report generation

What it produces:

- Variation matrices (log-ratio based)
- Shapley matrices (raw and normalized for plotting)
- Heatmaps and optional text reports
- Optional fixed-vs-random perturbation comparison outputs

### Sensitivity analysis

Main path:

- Pipeline: `src/pipelines/sensitivity_analysis.py::sensitivity_analysis`
- Problem specification: `src/utils/sensitivity.py::get_problem_parameters`
- Sobol sampling/analysis: SALib (`saltelli.sample`, `sobol.analyze`)
- Batch simulation backend: `src/utils/sensitivity.py::run_simulation_with_params`
- Optional convergence workflow: `run_convergence_analysis(...)` + convergence plots

What it produces:

- Sobol indices per target species
- Optional convergence diagnostics and plots
- Fixed-vs-sampled perturbation comparison CSV (when requested)

### Network generation

Main path:

- Pipeline: `src/pipelines/network.py::create_model_netwrok`
- Graph extraction: `src/utils/graph.py::get_network_from_sbml`
- Rendering: `src/utils/graph.py::plot_network`

What it produces:

- Network image (PNG)
- DOT graph source (`.gv`) for external graph tooling

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

From the repository root (example runner):

```bash
python -m src.examples.mainV2 -h
```

Run a simulation:

```bash
python -m src.examples.mainV2 simulate models/test.xml -t 120 -o results
```

Inspect command-specific options:

```bash
python -m src.examples.mainV2 simulate -h
python -m src.examples.mainV2 importance_assessment -h
python -m src.examples.mainV2 sensitivity_analysis -h
```

## Commands

These commands use the example runner in `src/examples/mainV2.py`.

General form:

```bash
python -m src.examples.mainV2 <command> [options]
```

Available commands:

- `simulate`
  Time-course model simulation with optional steady-state termination and plotting.
- `importance_assessment`
  Shapley-style influence analysis with knockout/knockin scenarios, optionally with perturbations.
- `sensitivity_analysis`
  Global Sobol sensitivity analysis for selected species/targets.
- `knockout_species`
  Build and save a model where one species is disabled.
- `knockout_reaction`
  Build and save a model where one reaction is disabled.
- `knockin_species`
  Build and save a model where one species is fixed to a computed reinforced value.
- `knockin_reaction`
  Build and save a model where one reaction is reinforced via reactant replacement strategy.
- `create_network`
  Build and save a reaction-network graph from the SBML model.

## Output structure

By default, outputs are written under `./results` in a model-specific folder:

```text
<output>/<model_name>/
├── csv/
├── dot/
├── images/
└── reports/
```

## Examples

### 1) Simulate a model (static plot + CSV)

```bash
python -m src.examples.mainV2 simulate models/KnockinModelV2.xml \
  -t 120 \
  -i cvode \
  -o results
```

### 2) Simulate until steady state (interactive HTML plot)

```bash
python -m src.examples.mainV2 simulate models/KnockinModelV2.xml \
  --steady-state \
  --max-time 2000 \
  --sim-step 10 \
  --threshold 1e-7 \
  --interactive \
  -o results
```

### 3) Importance assessment (knockout, no perturbations)

```bash
python -m src.examples.mainV2 importance_assessment models/KnockinModelV2.xml \
  --operation knockout \
  --payoff-function last \
  -t 120 \
  -o results
```

### 4) Importance assessment with random perturbations

```bash
python -m src.examples.mainV2 importance_assessment models/KnockinModelV2.xml \
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
python -m src.examples.mainV2 importance_assessment models/KnockinModelV2.xml \
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
python -m src.examples.mainV2 sensitivity_analysis models/KnockinModelV2.xml \
  --input-species S1 S2 \
  --base-samples 1024 \
  --perturbation-range 20 \
  --operation knockout \
  -o results
```

### 7) Sensitivity analysis with convergence check

```bash
python -m src.examples.mainV2 sensitivity_analysis models/KnockinModelV2.xml \
  --input-species S1 S2 \
  --check-convergence \
  -o results
```

### 8) Knock out one species and save edited model

```bash
python -m src.examples.mainV2 knockout_species models/KnockinModelV2.xml S1 \
  --model-dir models \
  -o results
```

### 9) Knock out one reaction and save edited model

```bash
python -m src.examples.mainV2 knockout_reaction models/KnockinModelV2.xml R1_MassAction_Explicit \
  --model-dir models \
  -o results
```

### 10) Knock in one species and save edited model

```bash
python -m src.examples.mainV2 knockin_species models/KnockinModelV2.xml S1 \
  --model-dir models \
  -o results
```

### 11) Knock in one reaction and save edited model

```bash
python -m src.examples.mainV2 knockin_reaction models/KnockinModelV2.xml R1_MassAction_Explicit \
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

- `src/examples/mainV2.py`: example CLI runner and pipeline orchestration (not an entry point)
- `models/`: sample SBML models
- `results/`: default output directory

## Acknowledgments
<p align="right">
  <br><br>
  <br><br>
  To Aurora<br>
  Thank you for your constant support, encouragement, and guidance.<br>
  This work wouldn't be what it is without you.
  <br><br>
  <br><br>
</p>
<hr>
