# Repository Audit Report

## Scope

This audit reviewed the main library components of `shapcrn`, with emphasis on:

- `src/shapcrn/utils/utils.py`
- `src/shapcrn/utils/simulation.py`
- `src/shapcrn/utils/sensitivity.py`
- `src/shapcrn/utils/sbml/io.py`
- `src/shapcrn/utils/sbml/utils.py`
- `src/shapcrn/utils/sbml/knock.py`
- Public pipeline entry paths under `src/shapcrn/pipelines/` where needed to confirm reachability

The review was static only. No code fixes were applied, and no build or mutation-oriented commands were run as part of the audit itself.

## Executive Summary

The repository is a focused scientific-analysis package rather than a security-sensitive service. I did not find any obvious high-severity security issues such as shell injection, arbitrary code execution, or unsafe deserialization in the reviewed code paths.

I did find several confirmed correctness and reliability defects that affect the advertised CLI and multiple public workflows:

- The published CLI entry point is broken because the package metadata and README reference a module that does not exist.
- `importance_assessment` contains a reachable runtime failure in `simulate_original_model()`.
- `sensitivity_analysis` mixes non-interactive CLI packaging with interactive `input()` prompts and also raises an exception class that is not defined.
- `utils.z_score_normalize()` does not compute z-scores correctly.
- `usage_example.py` references a missing exception class and over-wraps unrelated failures as model-loading errors.
- `knockin_species` dereferences a possibly missing species before validating that it exists.
- Steady-state simulation silently overrides the user-selected integrator and always switches to `cvode`.

## Confirmed Findings

### 1. Broken published CLI entry point

- Severity: High
- Type: Reliability
- Affected files:
  - `pyproject.toml:25`
  - `README.md:18`
  - `README.md:62`
  - `README.md:90`
  - `README.md:240`
  - `src/shapcrn/examples/usage_example.py:24`

Why it breaks:

The package metadata exposes the console script as `shapcrn = "shapcrn.examples.mainV2:main"`, and the README repeatedly documents `src/shapcrn/examples/mainV2.py` as the CLI runner. That module is not present in the repository. The only visible runner is `src/shapcrn/examples/usage_example.py`.

Practical impact:

- Installing the package and invoking `shapcrn` will fail at import time.
- The documented `python -m shapcrn.examples.mainV2 ...` commands are also invalid.
- This blocks the primary public entry point advertised by the project.

Recommended fix direction:

- Point `project.scripts.shapcrn` to the real module, or restore the missing `mainV2.py`.
- Update the README so all documented commands reference the actual CLI module.

### 2. Reachable runtime failure in `importance_assessment`

- Severity: High
- Type: Bug
- Affected files:
  - `src/shapcrn/pipelines/importance.py:205`
  - `src/shapcrn/pipelines/importance.py:213`
  - `src/shapcrn/examples/usage_example.py:115`

Why it breaks:

`simulate_original_model()` attempts to detect whether a knocked target is a reaction with:

`if ts in sbml_model.getListOfReactions().getId():`

`getListOfReactions()` returns a reaction list, not a single reaction object, so `.getId()` is not a valid call on that list. This path is reached from the public `importance_assessment` pipeline before simulation results are finalized.

Practical impact:

- The importance workflow can fail with an `AttributeError` during normal execution.
- This is a hard failure in a documented top-level command.

Recommended fix direction:

- Replace the invalid membership check with a real reaction-ID collection, for example by precomputing `[r.getId() for r in sbml_model.getListOfReactions()]`.

### 3. `sensitivity_analysis` is not consistently usable as a packaged CLI

- Severity: High
- Type: Reliability
- Affected files:
  - `src/shapcrn/pipelines/sensitivity_analysis.py:145`
  - `src/shapcrn/pipelines/sensitivity_analysis.py:154`
  - `src/shapcrn/pipelines/sensitivity_analysis.py:161`
  - `src/shapcrn/pipelines/sensitivity_analysis.py:173`
  - `src/shapcrn/pipelines/sensitivity_analysis.py:262`
  - `src/shapcrn/pipelines/sensitivity_analysis.py:264`
  - `src/shapcrn/exceptions.py:1`

Why it breaks:

Two separate issues are confirmed here:

- The non-convergence branch calls `prompt_user_for_N()`, which uses `input()` repeatedly. That makes the workflow interactive even though the package is presented as a command-line tool suitable for scripted runs.
- The same branch raises `ex.InvalidArgumentError`, but `InvalidArgumentError` is not defined in `src/shapcrn/exceptions.py`.

Practical impact:

- Automated or non-interactive use can block waiting for stdin.
- If fixed perturbations are missing, the intended validation error becomes an `AttributeError` on the missing exception class instead.

Recommended fix direction:

- Replace interactive prompts with explicit CLI arguments and deterministic defaults.
- Add a real exception type for invalid user arguments, or reuse an existing validation-oriented exception consistently.

### 4. `usage_example.py` references an undefined exception and obscures root causes (RESUME HERE)

- Severity: High
- Type: Reliability
- Affected files:
  - `src/shapcrn/examples/usage_example.py:35`
  - `src/shapcrn/examples/usage_example.py:48`
  - `src/shapcrn/examples/usage_example.py:134`
  - `src/shapcrn/exceptions.py:33`
  - `src/shapcrn/exceptions.py:361`

Why it breaks:

- The file existence check raises `ex.FileNotFoundError`, but that exception class is not defined in `src/shapcrn/exceptions.py`.
- The broad `except Exception as e:` wrapper then re-raises everything as `ModelNotFoundError`, even when the underlying issue is not a missing model file.

Practical impact:

- Missing input files do not produce the intended custom error type.
- Unrelated failures become misleading "model not found" errors, which makes debugging much harder.

Recommended fix direction:

- Replace `ex.FileNotFoundError` with a defined repository exception or the built-in `FileNotFoundError`.
- Narrow exception handling so model-loading failures are not conflated with downstream pipeline bugs.

### 5. `knockin_species` dereferences a missing species before validating it

- Severity: Medium
- Type: Bug
- Affected files:
  - `src/shapcrn/pipelines/knockin/knockin_species.py:27`
  - `src/shapcrn/pipelines/knockin/knockin_species.py:31`
  - `src/shapcrn/utils/sbml/knock.py:256`

Why it breaks:

The pipeline only checks whether the argument itself is `None`, not whether the species exists in the model. It then does:

`sbml_model.getSpecies(parsed_args["target_species"]).getId()`

If the user provides an unknown species ID, `getSpecies(...)` returns `None` and `.getId()` raises immediately.

Practical impact:

- A simple invalid species identifier causes an unhandled `AttributeError`.
- The downstream, better-defined validation in `utils/sbml/knock.py` is never reached.

Recommended fix direction:

- Validate the resolved species object before calling `.getId()`.
- Raise `InvalidSpeciesError` from the pipeline when the target is absent.

### 6. `z_score_normalize()` does not compute z-scores correctly

- Severity: Medium
- Type: Bug
- Affected files:
  - `src/shapcrn/utils/utils.py:548`
  - `src/shapcrn/utils/utils.py:554`

Why it breaks:

The implementation uses:

`z_scores = (heatmap_data - means) / 3 * stds_safe`

Because multiplication and division have the same precedence and are evaluated left-to-right, this computes `((x - mean) / 3) * std`, not `(x - mean) / std`. The result scales with the standard deviation instead of normalizing by it.

Practical impact:

- Any downstream logic or visualization relying on this helper gets numerically incorrect values.
- The function name and output semantics are misleading.

Recommended fix direction:

- Implement the standard formula `(heatmap_data - means) / stds_safe`.
- Keep the zero-variance protection already present via `stds_safe`.

### 7. Steady-state simulation overrides the user-selected integrator

- Severity: Medium
- Type: Reliability
- Affected files:
  - `src/shapcrn/utils/simulation.py:278`
  - `src/shapcrn/utils/simulation.py:287`
  - `src/shapcrn/utils/simulation.py:371`
  - `src/shapcrn/utils/utils.py:95`

Why it breaks:

The CLI allows users to choose `cvode`, `gillespie`, or `rk4`, and `simulate()` accepts that configured model. But the steady-state path unconditionally calls:

`rr_model.setIntegrator("cvode")`

inside `simulate_with_steady_state()`.

Practical impact:

- User intent is silently ignored whenever steady-state mode is used.
- Results may differ materially from what the caller requested, especially for stochastic workflows.

Recommended fix direction:

- Preserve the caller-selected integrator, or reject unsupported integrators explicitly when steady-state mode requires a specific solver.

## Security Assessment

No obvious high-severity security issues were found in the reviewed code paths.

Specifically, the audit did not identify:

- use of `eval()` or `exec()`
- shell execution via `subprocess` or `os.system`
- unsafe deserialization patterns such as unrestricted pickle loading
- network-facing request handling logic

The main security-relevant concern is lower severity: the tool accepts user-controlled output and log paths and writes to them directly. That is typical for a local CLI, but it means the process can overwrite any file path the invoking user can access.

Relevant examples:

- `src/shapcrn/utils/utils.py:425`
- `src/shapcrn/utils/sbml/io.py:97`
- `src/shapcrn/utils/sbml/io.py:105`
- `src/shapcrn/utils/graph.py:173`
- `src/shapcrn/utils/plot.py:399`

For a local scientific CLI this is usually an acceptable trust model, but it should be documented rather than treated as a hardened boundary.

## Lower-Risk Observations

### Mixed log-file conventions

Some functions treat `log_file` as a string path and pass it through `print_log()`, while others call `log_file.write(...)` directly.

Relevant examples:

- Path-based logging: `src/shapcrn/utils/utils.py:408`
- Direct handle-style writes: `src/shapcrn/utils/utils.py:557`, `src/shapcrn/utils/graph.py:234`

This inconsistency can create secondary failures on error paths if a string path is passed where a file-like object is assumed.

### Debug output left in library code

There are several unconditional `print()` statements and debug-oriented traces in library paths, for example:

- `src/shapcrn/utils/sbml/io.py:94`
- `src/shapcrn/utils/sbml/io.py:95`
- `src/shapcrn/utils/sensitivity.py:466`
- `src/shapcrn/utils/sensitivity.py:479`

These do not appear security-critical, but they make CLI behavior noisier and reduce polish for scripted use.

### User-controlled write destinations are broad but expected

The package creates directories and writes reports, plots, CSVs, DOT files, and logs to caller-specified locations. This is normal for a local analysis CLI, but it would be risky if the code were ever repurposed for a multi-user or service context.

## Suggested Remediations

1. Restore a valid CLI entry point and make the package metadata, README, and actual module layout consistent.
2. Fix the confirmed hard runtime failures first:
   - reaction-ID detection in `importance_assessment`
   - missing exception classes in `usage_example.py` and `sensitivity_analysis.py`
   - missing-species validation in `knockin_species`
3. Correct `z_score_normalize()` and add a small unit test that checks the formula against a known array.
4. Decide whether steady-state mode should preserve the chosen integrator or reject unsupported ones explicitly.
5. Normalize logging conventions so every helper either accepts a path or a file handle, but not both implicitly.
6. Document the trust model for output and log paths, especially if the project may later be wrapped by a service or notebook environment.
