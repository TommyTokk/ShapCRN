import libsbml
import pandas as pd
import numpy as np
import os

from src.utils import utils as ut
from src.utils.sbml import io as sbml_io
from src.utils.sbml import utils as sbml_ut
from src.utils.sbml import reactions as sbml_react
from src.utils import simulation as sim_ut
from src.utils import plot as plt_ut


payoff_functions = {
    'max': ut.payoff_max,
    'min': ut.payoff_min,
    'last': ut.payoff_last,
}



def parse_args(args):
    """
    Parses the command line arguments for the importance assessment pipeline.
    """

    # Model path
    input_path = args.input_path
    operation = args.operation

    # Species selection args
    input_species_ids = args.input_species
    knocked_species_ids = args.knocked
    target_ids = args.target_nodes
    preserve_inputs = args.preserve_inputs

    # Sampling and perturbations
    use_perturbations = args.use_perturbations
    use_fixed_perturbations = args.use_fixed_perturbations

    fixed_perturbations = None

    if use_fixed_perturbations:
        if args.fixed_perturbations is None:
            raise ValueError(
                "You need to specify the variations to use setting the --fixed-perturbations parameter"
            )
        fixed_perturbations = args.fixed_perturbations

    num_samples = args.num_samples
    variation_percentage = args.variation

    perturbations_importance = args.perturbations_importance
    random_perturbations_importance = args.random_perturbations_importance

    # Shapley arguments
    payoff_function = args.payoff_function

    # Simulation arguments
    sim_time = args.time
    sim_integrator = args.integrator
    use_steady_state = args.steady_state
    ss_max_time = args.max_time
    ss_sim_steps = args.sim_step
    ss_sim_points = args.points
    ss_threshold = args.threshold

    # Output arguments
    output_dir = args.output
    log_file = args.log

    #Pack args and return
    parsed_args = {
        "input_path": input_path,
        "operation": operation,
        "input_species_ids": input_species_ids,
        "knocked_species_ids": knocked_species_ids,
        "target_ids": target_ids,
        "preserve_inputs": preserve_inputs,
        "use_perturbations": use_perturbations,
        "use_fixed_perturbations": use_fixed_perturbations,
        "fixed_perturbations": fixed_perturbations,
        "num_samples": num_samples,
        "variation_percentage": variation_percentage,
        "perturbations_importance": perturbations_importance,
        "random_perturbations_importance": random_perturbations_importance,
        "payoff_function": payoff_function,
        "sim_time": sim_time,
        "sim_integrator": sim_integrator,
        "use_steady_state": use_steady_state,
        "ss_max_time": ss_max_time,
        "ss_sim_steps": ss_sim_steps,
        "ss_sim_points": ss_sim_points,
        "ss_threshold": ss_threshold,
        "output_dir": output_dir,
        "log_file": log_file
    }

    return parsed_args


def model_preparation(args):
    """
    Prepares the model for importance assessment by loading it and selecting the species to analyze.
    """


    log_file = args["log_file"]

    sbml_doc = sbml_io.load_model(args["input_path"])
    sbml_model = sbml_react.split_all_reversible_reactions(sbml_doc.getModel(), args['log_file'])

    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Validate the knocked list
    if args["knocked_species_ids"] is None:# If no knocked species are provided, all will be used
        knocked_ids = species_list
    else:# Otherwise only the ones passed
        knocked_ids = args["knocked_species_ids"]

    # Check for the input preservation
    if args["preserve_inputs"]:
        knocked_ids = list(set(knocked_ids) - set(args["input_species_ids"]))

    # Sort the species
    id_to_idx = {}
    for indx, s in enumerate(species_list):
        id_to_idx[s] = indx

    knocked_ids.sort(key=lambda x: id_to_idx[x])

    return {
        'sbml_model': sbml_model,
        'knocked_ids': knocked_ids
    }

def generate_samples(sbml_model, args):

    samples = None

    use_perturbations = args["use_perturbations"]
    use_fixed_perturbations = args["use_fixed_perturbations"]

    if use_perturbations:
        if use_fixed_perturbations:
            samples = sbml_ut.get_fixed_combinations(
                sbml_model,
                args["input_species_ids"],
                args["fixed_perturbations"],
                args["log_file"]
            )
        else:
            samples = sbml_ut.generate_species_random_combinations(
                sbml_model,
                target_species=args["input_species_ids"],
                n_samples=args["num_samples"],
                variation = args["variation_percentage"]
            )
    return samples

def simulate_original_model(sbml_model:libsbml.Model, knocked_ids, samples,  args):

    species_list = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Load the roadrunner model
    rr = sim_ut.load_roadrunner_model(
        sbml_model=sbml_model,
        integrator=args["sim_integrator"],
        log_file=args["log_file"]
    )

    # fix the selections
    selections = rr.timeCourseSelections
    for s in species_list:
        if f"[{s}]" not in selections:
            selections.append(f"[{s}]")

    # Add also the reactions in the selections if target
    for ts in knocked_ids:
        if ts in sbml_model.getListOfReactions().getId():
            selections.append(f"{ts}")

    rr.timeCourseSelections = selections

    # Simulate 

    original_results, ss_time, colnames = sim_ut.simulate(
        rr,
        end_time=args["sim_time"],
        start_time=0,
        steady_state=args["use_steady_state"],
        max_end_time=args["ss_max_time"],
        log_file=args["log_file"]
    )

    original_df = pd.DataFrame(original_results[:, 1:], columns=colnames[1:])

    colnames_to_index = {}
    for i, el in enumerate(colnames):
        if el == "time":
            continue
        colnames_to_index[el] = i

    min_ss_time = (
        ss_time
        if ss_time is not None and ss_time <= args["ss_max_time"]
        else args["ss_max_time"]
    )

    original_data = None

    # Simulating the perturbations if required

    if args["use_perturbations"]:
        if args["use_fixed_perturbations"]:
            ut.print_log(args["log_file"], "[INFO] Simulating with fixed perturbations")
        else:
            ut.print_log(args["log_file"], "[INFO] Simulating with random perturbations")

        samples_simulations_results, _ = sim_ut.simulate_combinations(
            rr,
            sbml_ut.create_combinations(samples),
            args["input_species_ids"],
            min_ss_time,
            args["sim_time"],
            args["ss_max_time"],
            args["use_steady_state"],
            args["log_file"]
        )

    # Prepare the final data
    original_data = [original_df]

    if args["use_perturbations"]:
        for i in range(len(samples_simulations_results)):
            sim_res_i = samples_simulations_results[i]
            original_data.append(
                pd.DataFrame(sim_res_i[:, 1:], columns=colnames[1:])
            )

    return original_data, selections, min_ss_time

def simulate_knocked_data(sbml_model: libsbml.Model, knocked_ids, samples, selections, ss_min_time, args, new_values: list=None):
    
    operation = args["operation"]

    ut.print_log(args["log_file"], f"Operation: {operation}")

    # Create the models

    knocked_data = []

    sbml_str = libsbml.writeSBMLToString(sbml_model.getSBMLDocument())

    if operation == "knockin":
        models_dict = sbml_ut.create_ki_models(knocked_ids, sbml_model, sbml_str, new_values, args["log_file"])
    elif operation == "knockout":
        models_dict = sbml_ut.create_ko_models(knocked_ids, sbml_model, sbml_str, args["log_file"])

    # TODO: Complete the knockout 
    knocked_data = sim_ut.process_species_multiprocessing(
        knocked_ids,
        models_dict,
        samples,
        args["input_species_ids"],
        selections,
        args["sim_integrator"],
        start_time=0,
        end_time = args["sim_time"],
        steady_state=args["use_steady_state"],
        max_end_time=args["ss_max_time"],
        min_ss_time=ss_min_time,
        use_perturbations=args["use_perturbations"],
        preserve_input=args["preserve_inputs"],
        log_file=args["log_file"]
    )

    return knocked_data


def run_shap_analysis(original_data, knocked_data, n_combinations, n_input_ids, payoff = "last", log_file=None):

    payoff = payoff_functions[payoff]
    
    #Get the dictionary of payoffs
    payoff_dict = sim_ut.get_payoff_vals(
        original_data,
        knocked_data,
        payoff,
        log_file=log_file
    )

    shap_values = sim_ut.get_shapley_values(
        payoff_dict,
        n_combinations,
        n_input_ids,
        log_file=log_file
    )

    return shap_values


def _lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC).

    Measures agreement between two sets of measurements of the same quantity,
    combining precision (Pearson ρ) and accuracy (bias correction).

        CCC = 2 · Cov(x, y) / (Var(x) + Var(y) + (μ_x − μ_y)²)

    Returns NaN when either vector has fewer than 2 valid observations.

    References
    ----------
    Lin, L. I-K. (1989). "A Concordance Correlation Coefficient to Evaluate
    Reproducibility." *Biometrics*, 45(1), 255–268.
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    if sx2 == 0.0 and sy2 == 0.0:
        return 1.0 if np.isclose(mx, my) else 0.0
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return float(2.0 * sxy / (sx2 + sy2 + (mx - my) ** 2))


def _benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    """
    Benjamini–Hochberg procedure for controlling the False Discovery Rate.

    Parameters
    ----------
    pvals : np.ndarray
        1-D array of raw p-values.
    alpha : float
        Target FDR level (default 0.05).

    Returns
    -------
    rejected : np.ndarray[bool]
        Boolean mask — True where the null hypothesis is rejected after
        BH correction.
    adjusted : np.ndarray[float]
        BH-adjusted p-values (capped at 1.0).

    References
    ----------
    Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
    Rate: A Practical and Powerful Approach to Multiple Testing."
    *Journal of the Royal Statistical Society B*, 57(1), 289–300.
    """
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    # BH-adjusted p-values: p_adj_i = min(p_i * m / rank_i, 1.0)
    # enforced to be monotonically non-decreasing from the bottom up
    adjusted = np.minimum(sorted_p * m / np.arange(1, m + 1), 1.0)
    # enforce monotonicity (cumulative minimum from the right)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # un-sort
    adjusted_unsorted = np.empty(m)
    adjusted_unsorted[order] = adjusted

    rejected = adjusted_unsorted <= alpha
    return rejected, adjusted_unsorted


def assess_perturbation_importance(
    original_data, knocked_data, alpha=0.05, min_wilcoxon_n=6, log_file=None
):
    """
    Statistically assess whether input perturbations materially change
    knockout-effect estimates.

    The function compares, for every (knocked-species, target-species) pair,
    the knockout effect obtained from the **unperturbed baseline** with the
    distribution of effects obtained across input **perturbations**.  It
    returns four complementary, well-established statistical indicators.

    Parameters
    ----------
    original_data : list[pd.DataFrame]
        Index 0 is the baseline (unperturbed) simulation of the original model;
        indices 1 … N are perturbation simulations.
    knocked_data : list[tuple[str, list[pd.DataFrame]]]
        Each entry is ``(species_id, [baseline_ko_df, perturbed_ko_df_1, …])``.
    alpha : float, optional
        Significance level for the per-entry Wilcoxon signed-rank tests
        **after** Benjamini–Hochberg FDR correction (default 0.05).
    min_wilcoxon_n : int, optional
        Minimum number of perturbation samples required to run a Wilcoxon test
        for a given matrix entry (default 6).  Entries with fewer samples are
        excluded from the count.
    log_file : file, optional
        Log file handle.

    Returns
    -------
    dict
        lins_ccc : float
            **Lin's Concordance Correlation Coefficient** between the flattened
            baseline-only effect vector and the perturbation-median effect vector
            (signed log₂-ratios).

            ======  ============================================
            CCC     Interpretation (after Lin, 1989)
            ======  ============================================
            ≥ 0.95  Almost perfect agreement → *negligible* need
            ≥ 0.80  Substantial agreement    → *low* need
            ≥ 0.50  Moderate agreement       → *moderate* need
            < 0.50  Poor agreement           → *high* need
            ======  ============================================

        median_cv : float
            Median **coefficient of variation** of the signed log₂-ratio across
            perturbations, taken over all (knocked, target) entries.
            CV = std(d_pert) / |mean(d_pert)|.
            Low CV → effects are stable across perturbations.

        fraction_significant_bh : float
            Fraction of tested (knocked, target) entries where the Wilcoxon
            signed-rank test rejects H₀ after **Benjamini–Hochberg** FDR
            correction at level ``alpha``.

        fraction_significant_raw : float
            Same fraction but **without** multiple-testing correction
            (provided for reference; should not be used for decision-making).

        n_tests : int
            Total number of Wilcoxon tests performed.

        necessity_level : str
            Human-readable label derived from ``lins_ccc``:
            ``"negligible"`` / ``"low"`` / ``"moderate"`` / ``"high"``.

        ranking_spearman : float
            **Spearman ρ** between knocked-species importance rankings (median
            absolute effect per species) with and without perturbations.

        ranking_spearman_pvalue : float
            Two-sided p-value for the Spearman test.

        cv_matrix : pd.DataFrame
            Per-entry CV matrix (n_knocked × n_species).

        per_species_cv : pd.Series
            Median CV per knocked species (index = species id).

        baseline_effects : pd.DataFrame
            Signed log₂-ratio effect matrix from the baseline simulation.

        perturbation_median_effects : pd.DataFrame
            Signed log₂-ratio effect matrix aggregated (median) across
            perturbation simulations only (baseline excluded).

        wilcoxon_pvals_raw : pd.DataFrame
            Raw Wilcoxon p-values matrix (n_knocked × n_species).  NaN where
            no test was run.

        wilcoxon_pvals_bh : pd.DataFrame
            BH-adjusted p-values matrix (same shape).

    References
    ----------
    - Lin, L. I-K. (1989). "A Concordance Correlation Coefficient to Evaluate
      Reproducibility." *Biometrics*, 45(1), 255–268.
    - Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods."
      *Biometrics Bulletin*, 1(6), 80–83.
    - Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
      Rate: A Practical and Powerful Approach to Multiple Testing."
      *J. R. Stat. Soc. B*, 57(1), 289–300.
    """
    from scipy.stats import spearmanr, wilcoxon

    # ── input validation ────────────────────────────────────────────────
    if original_data is None or len(original_data) == 0:
        raise ValueError("original_data must contain at least one simulation DataFrame.")
    if knocked_data is None or len(knocked_data) == 0:
        raise ValueError("knocked_data must contain at least one knocked simulation.")
    if len(original_data) < 2:
        raise ValueError(
            "original_data must contain at least the baseline [0] and one "
            "perturbation [1] to assess perturbation importance."
        )

    epsilon = 1e-20
    n_perturbations = len(original_data) - 1
    columns = original_data[0].columns

    # ── 1. Baseline signed log₂-ratio matrix ───────────────────────────
    baseline_orig_ss = original_data[0].tail(1).clip(lower=epsilon)
    baseline_rows = []

    for knocked_species, knocked_species_data in knocked_data:
        ko_list = (
            knocked_species_data
            if isinstance(knocked_species_data, list)
            else [knocked_species_data]
        )
        if len(ko_list) == 0:
            raise ValueError(f"Knocked data for species '{knocked_species}' is empty.")

        ko_ss = ko_list[0].tail(1).clip(lower=epsilon)
        lr = np.log2(ko_ss.to_numpy() / baseline_orig_ss.to_numpy()).flatten()
        series = pd.Series(lr, index=columns, name=knocked_species)
        baseline_rows.append(series)

    baseline_effects = pd.DataFrame(baseline_rows)
    _mask_diagonal(baseline_effects)

    # ── 2. Per-perturbation signed log₂-ratios ─────────────────────────
    perturbation_log_ratios = {}

    for knocked_species, knocked_species_data in knocked_data:
        ko_list = (
            knocked_species_data
            if isinstance(knocked_species_data, list)
            else [knocked_species_data]
        )
        ko_perturbations = ko_list[1:]  # skip baseline
        lrs = []
        for i, ko_pert_df in enumerate(ko_perturbations):
            orig_pert_ss = original_data[i + 1].tail(1).clip(lower=epsilon)
            ko_pert_ss = ko_pert_df.tail(1).clip(lower=epsilon)
            lr = np.log2(ko_pert_ss.to_numpy() / orig_pert_ss.to_numpy()).flatten()
            lrs.append(lr)

        perturbation_log_ratios[knocked_species] = pd.DataFrame(
            lrs, columns=columns
        )

    # ── 3. Perturbation-median signed effect matrix ─────────────────────
    pert_median_rows = []
    for sp in baseline_effects.index:
        med = perturbation_log_ratios[sp].median()
        med.name = sp
        pert_median_rows.append(med)

    perturbation_median_effects = pd.DataFrame(pert_median_rows)
    _mask_diagonal(perturbation_median_effects)

    # ── 4. Lin's CCC ───────────────────────────────────────────────────
    b_flat = baseline_effects.to_numpy().flatten()
    p_flat = perturbation_median_effects.to_numpy().flatten()
    valid = ~(np.isnan(b_flat) | np.isnan(p_flat))
    lins_ccc = _lins_ccc(b_flat[valid], p_flat[valid])

    # ── 5. CV across perturbations ──────────────────────────────────────
    cv_rows = []
    for sp in baseline_effects.index:
        df_pert = perturbation_log_ratios[sp]
        means = df_pert.mean()
        stds = df_pert.std(ddof=1)
        cv = stds / (means.abs() + epsilon)
        cv.name = sp
        cv_rows.append(cv)

    cv_matrix = pd.DataFrame(cv_rows)
    _mask_diagonal(cv_matrix)
    median_cv = float(np.nanmedian(cv_matrix.to_numpy()))
    per_species_cv = cv_matrix.median(axis=1, skipna=True)

    # ── 6. Wilcoxon signed-rank tests + BH correction ──────────────────
    #    First pass: collect all raw p-values
    raw_pval_matrix = pd.DataFrame(
        np.nan, index=baseline_effects.index, columns=columns
    )
    pval_coords = []   # list of (row_label, col_label) for tests that ran
    raw_pvals_list = []

    for sp in baseline_effects.index:
        bl_row = baseline_effects.loc[sp]
        df_pert = perturbation_log_ratios[sp]

        for col in columns:
            bl_val = bl_row[col]
            if np.isnan(bl_val):
                continue
            pert_vals = df_pert[col].dropna().to_numpy()
            if len(pert_vals) < min_wilcoxon_n:
                continue
            diffs = pert_vals - bl_val
            if np.all(diffs == 0):
                # No variation at all → p = 1.0 (cannot reject H₀)
                raw_pval_matrix.loc[sp, col] = 1.0
                pval_coords.append((sp, col))
                raw_pvals_list.append(1.0)
                continue
            try:
                _, p = wilcoxon(diffs, alternative="two-sided")
                raw_pval_matrix.loc[sp, col] = p
                pval_coords.append((sp, col))
                raw_pvals_list.append(p)
            except ValueError:
                continue

    n_tests = len(raw_pvals_list)

    ut.print_log(log_file, f"[INFO] Wilcoxon tests performed: {n_tests}")

    # Second pass: BH correction across all tests
    bh_pval_matrix = pd.DataFrame(
        np.nan, index=baseline_effects.index, columns=columns
    )

    if n_tests > 0:
        raw_pvals_arr = np.array(raw_pvals_list)
        bh_rejected, bh_adjusted = _benjamini_hochberg(raw_pvals_arr, alpha)

        for idx, (sp, col) in enumerate(pval_coords):
            bh_pval_matrix.loc[sp, col] = bh_adjusted[idx]

        n_sig_raw = int(np.sum(raw_pvals_arr < alpha))
        n_sig_bh = int(np.sum(bh_rejected))
    else:
        n_sig_raw = 0
        n_sig_bh = 0

    fraction_significant_raw = n_sig_raw / n_tests if n_tests > 0 else np.nan
    fraction_significant_bh = n_sig_bh / n_tests if n_tests > 0 else np.nan

    # ── 7. Ranking stability (Spearman ρ) ───────────────────────────────
    no_pert_importance = baseline_effects.abs().median(axis=1, skipna=True)
    with_pert_importance = perturbation_median_effects.abs().median(axis=1, skipna=True)

    common_ids = no_pert_importance.dropna().index.intersection(
        with_pert_importance.dropna().index
    )

    ranking_spearman = np.nan
    ranking_spearman_pvalue = np.nan
    if len(common_ids) >= 3:
        rho, pval = spearmanr(
            no_pert_importance.loc[common_ids].to_numpy(),
            with_pert_importance.loc[common_ids].to_numpy(),
        )
        ranking_spearman = float(rho)
        ranking_spearman_pvalue = float(pval)

    # ── 8. Necessity level (based on CCC thresholds from Lin, 1989) ────
    if lins_ccc >= 0.95:
        necessity_level = "negligible"
    elif lins_ccc >= 0.80:
        necessity_level = "low"
    elif lins_ccc >= 0.50:
        necessity_level = "moderate"
    else:
        necessity_level = "high"

    return {
        "lins_ccc": lins_ccc,
        "median_cv": median_cv,
        "fraction_significant_bh": fraction_significant_bh,
        "fraction_significant_raw": fraction_significant_raw,
        "n_tests": n_tests,
        "necessity_level": necessity_level,
        "ranking_spearman": ranking_spearman,
        "ranking_spearman_pvalue": ranking_spearman_pvalue,
        "cv_matrix": cv_matrix,
        "per_species_cv": per_species_cv,
        "baseline_effects": baseline_effects,
        "perturbation_median_effects": perturbation_median_effects,
        "wilcoxon_pvals_raw": raw_pval_matrix,
        "wilcoxon_pvals_bh": bh_pval_matrix,
    }


def _mask_diagonal(df: pd.DataFrame) -> None:
    """Set self-comparison entries (knocked species vs itself) to NaN in-place."""
    col_names = df.columns.astype(str).str.strip("[]")
    idx = df.index.get_indexer(col_names)
    valid = idx != -1
    df.values[idx[valid], np.arange(len(df.columns))[valid]] = np.nan


def generate_importance_report(
        importance_results: dict, 
        variations_df: pd.DataFrame, 
        shapley_values_df: pd.DataFrame, 
        out_dirs: dict, 
        log_file=None):
    """
    Generate a plain-text report summarising the perturbation-importance
    assessment, Shapley values and variation statistics.

    The report is written to ``<out_dirs['reports']>/importance_report.txt``.

    Parameters
    ----------
    importance_results : dict
        Output of :func:`assess_perturbation_importance`.
    variations_df : pd.DataFrame
        Log-ratio variation matrix (knocked × species).
    shapley_values_df : pd.DataFrame
        Shapley-value matrix (knocked × species).
    out_dirs : dict
        Directory map with at least a ``'reports'`` key.
    log_file : file-like or None
        Optional log handle.
    """

    report_path = os.path.join(out_dirs["reports"], "importance_report.txt")

    sep = "=" * 60
    sub_sep = "-" * 60

    lines: list[str] = []
    w = lines.append  # shorthand

    # ── Header ──────────────────────────────────────────────────────
    w(sep)
    w("  PERTURBATION IMPORTANCE ASSESSMENT REPORT")
    w(sep)
    w("")

    # ── 1. Overall agreement ────────────────────────────────────────
    w("1. Overall agreement (baseline vs perturbation-median effects)")
    w(sub_sep)
    ccc = importance_results["lins_ccc"]
    w(f"   Lin's CCC            : {ccc:.4f}")
    w(f"   Necessity level      : {importance_results['necessity_level']}")
    w(f"   Median CV            : {importance_results['median_cv']:.4f}")
    w("")

    # ── 2. Wilcoxon signed-rank tests ───────────────────────────────
    w("2. Wilcoxon signed-rank tests")
    w(sub_sep)
    n_tests = importance_results["n_tests"]
    w(f"   Tests performed      : {n_tests}")
    frac_bh = importance_results["fraction_significant_bh"]
    frac_raw = importance_results["fraction_significant_raw"]
    if np.isnan(frac_bh):
        w("   Significant (BH)     : N/A (no tests run)")
        w("   Significant (raw)    : N/A")
    else:
        w(f"   Significant (BH)     : {frac_bh:.2%}  ({int(round(frac_bh * n_tests))}/{n_tests})")
        w(f"   Significant (raw)    : {frac_raw:.2%}  ({int(round(frac_raw * n_tests))}/{n_tests})")
    w("")

    # ── 3. Ranking stability ────────────────────────────────────────
    w("3. Ranking stability")
    w(sub_sep)
    rho = importance_results["ranking_spearman"]
    rho_p = importance_results["ranking_spearman_pvalue"]
    if np.isnan(rho):
        w("   Spearman rho         : N/A (too few species)")
    else:
        w(f"   Spearman rho         : {rho:.4f}")
        w(f"   Spearman p-value     : {rho_p:.4e}")
    w("")

    # ── 4. Top-5 most variable knocked species (by median CV) ──────
    w("4. Per-species variability (top-5 by median CV)")
    w(sub_sep)
    per_sp_cv = importance_results["per_species_cv"].dropna().sort_values(ascending=False)
    for i, (sp, cv_val) in enumerate(per_sp_cv.head(5).items()):
        w(f"   {i+1}. {sp:30s}  CV = {cv_val:.4f}")
    w("")

    # ── 5. Shapley-value summary ────────────────────────────────────
    w("5. Shapley-value summary")
    w(sub_sep)
    abs_shap = shapley_values_df.abs()
    median_shap = abs_shap.median(axis=1, skipna=True).sort_values(ascending=False)
    w("   Top-5 knocked species by median |Shapley|:")
    for i, (sp, val) in enumerate(median_shap.head(5).items()):
        w(f"   {i+1}. {sp:30s}  median |SV| = {val:.6f}")
    w("")

    # ── 6. Variation summary ────────────────────────────────────────
    w("6. Variation summary (log-ratio)")
    w(sub_sep)
    abs_var = variations_df.abs()
    median_var = abs_var.median(axis=1, skipna=True).sort_values(ascending=False)
    w("   Top-5 knocked species by median |variation|:")
    for i, (sp, val) in enumerate(median_var.head(5).items()):
        w(f"   {i+1}. {sp:30s}  median |var| = {val:.6f}")
    w("")

    w(sep)
    w("  END OF REPORT")
    w(sep)

    report_text = "\n".join(lines) + "\n"

    os.makedirs(out_dirs["reports"], exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)

    ut.print_log(log_file, f"[INFO] Importance report saved to {report_path}")




def importance_assessment(args, out_dirs):
    """
    Pipeline for importance assessment of reactions in a CRN.
    """

    # Arguent parsing
    parsed_args = parse_args(args)

    prep_res = model_preparation(parsed_args)

    sbml_model = prep_res["sbml_model"]
    knocked_ids = prep_res["knocked_ids"]

    ut.print_log(parsed_args["log_file"], f"Model loaded and prepared. Knocked species: {knocked_ids}")

    # Handle the samples
    samples = generate_samples(sbml_model, parsed_args)

    ut.print_log(parsed_args["log_file"], f"{samples}")

    # Simulate original model
    original_simulation_data, selections, min_ss_time = simulate_original_model(sbml_model, knocked_ids, samples, parsed_args)

    if parsed_args["operation"] == "knockin":
        # Calculating the new values (max_values), avoid first column with time
        max_values = list(original_simulation_data[0][selections[1:]].max())
        knocked_data = simulate_knocked_data(sbml_model, knocked_ids, samples, selections, min_ss_time, parsed_args, new_values=max_values)
    else:
        knocked_data = simulate_knocked_data(sbml_model, knocked_ids, samples, selections, min_ss_time, parsed_args)

    

    # Analyse the results
    if parsed_args["use_perturbations"]:
        if parsed_args["random_perturbations_importance"]:
            ut.print_log(parsed_args["log_file"], "[WARNING] Sorry random perturbations importance analysis is still in development :(")


        # Calculate the Shapley value
        n_combinations = np.power(parsed_args["num_samples"], len(parsed_args["input_species_ids"])) + 1
        shapley_values = run_shap_analysis(
            original_simulation_data, 
            knocked_data, 
            n_combinations, 
            len(parsed_args["input_species_ids"]),
            payoff=parsed_args["payoff_function"], 
            log_file=parsed_args["log_file"]
            )
        
        # Calculate the variations
        samples_relative_vars = sim_ut.get_relative_variations_log_ratio(
            original_simulation_data,
            knocked_data,
            aggregation="median",
            return_signed=False
        )

        
        # Plot the variations heatmap
        colnames_to_index = {}
        for i, el in enumerate(original_simulation_data[0].columns):
            if el == "time":
                continue
            colnames_to_index[el] = i

        variations_df = pd.DataFrame(samples_relative_vars, columns=selections[1:], index=knocked_ids)
        

        # Plot the Shapley values heatmap
        shapley_df = pd.DataFrame(shapley_values, columns=selections[1:], index=knocked_ids)

        if parsed_args["target_ids"] is not None:
            target_ids = parsed_args["target_ids"]

            # Resolve target IDs to actual column names (may be bracketed)
            resolved_cols = []
            for tid in target_ids:
                if tid in variations_df.columns:
                    resolved_cols.append(tid)
                elif f"[{tid}]" in variations_df.columns:
                    resolved_cols.append(f"[{tid}]")

            # focus only on the target ids columns
            variations_df = variations_df[resolved_cols]
            shapley_df = shapley_df[resolved_cols]


        # TODO: Complete the importance analysis with perturbations
        if parsed_args["perturbations_importance"]:
            importance_assessment_results = assess_perturbation_importance(
                original_simulation_data,
                knocked_data,                
                alpha=0.05,
                log_file=parsed_args["log_file"]
            )

            # TODO: Add the random perturbations importance analysis

            generate_importance_report(
                importance_assessment_results,
                variations_df,
                shapley_df,
                out_dirs,
                log_file=parsed_args["log_file"]
            )



        # Normalize with asinh to better visualize the differences
        shapley_df_normal, s = ut.normalize_asinh(shapley_df)

        

        # Save the CSVs
        variations_df.to_csv(os.path.join(out_dirs['csv'], "variations_across_perturbations.csv"))
        shapley_df.to_csv(os.path.join(out_dirs['csv'], "shapley_values.csv"))

        # Plots the heamaps
        plt_ut.plot_heatmap(
            variations_df,
            colnames_to_index=colnames_to_index,
            x_labels=variations_df.columns,
            y_labels=variations_df.index,
            title="Relative variations (log ratio) across perturbations",
            img_name="log_ratio_variations_heatmap.png",
            save_path=out_dirs['images'],
            log_file=parsed_args["log_file"]
        )

        plt_ut.plot_heatmap(
            shapley_df_normal,
            colnames_to_index=colnames_to_index,
            x_labels=shapley_df.columns,
            y_labels=shapley_df.index,
            title="Shapley values across perturbations (asinh normalized)",
            img_name="shapley_values_heatmap_normal.png",
            save_path=out_dirs['images'],
            log_file=parsed_args["log_file"]
        )


        pass
    else:# No perturbations required
        # Analyse just original and knocked data
        # Calculate the Shapley value
        ut.print_log(parsed_args["log_file"], "[INFO] Calculating without perturbations")
        ut.print_log(parsed_args["log_file"], f"Original data: \n{original_simulation_data}")
        ut.print_log(parsed_args["log_file"], f"Knocked data: \n{knocked_data}")

        variations = sim_ut.get_relative_variations_log_ratio_no_samples(
            original_simulation_data[0],
            knocked_data,
            return_signed=True
        )

        ut.print_log(parsed_args["log_file"], f"Variations: \n{variations}")

