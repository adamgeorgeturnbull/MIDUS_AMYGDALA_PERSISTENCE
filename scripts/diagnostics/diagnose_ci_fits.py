"""User-run, two-model diagnostic. Never overwrites analysis result tables.

Import is data-free. main() is for the restricted environment only.
Constructs models using the existing exporters' actual preparation functions,
then tries each optimizer independently on copies of the same model.
"""
import ast
import copy
import hashlib
import io
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from confidence_intervals import coefficient_ci


class ModelCaptured(BaseException):
    """Escape the exporter's Exception handlers before any fit is attempted."""
    def __init__(self, model):
        self.model = model


def capture_model(function, *args):
    def construct(*a, **kw):
        raise ModelCaptured(smf.mixedlm(*a, **kw))
    namespace = dict(function.__globals__)
    namespace['smf'] = SimpleNamespace(mixedlm=construct)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    try:
        clone(*args)
    except ModelCaptured as captured:
        return captured.model
    raise ValueError('Exporter did not construct a model; check sample size locally.')


def load_moderation_functions(path, shared):
    tree = ast.parse(path.read_text())
    namespace = dict(shared)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(definitions, str(path), 'exec'), namespace)
    return namespace


def fit_diagnostics(model, label, terms, methods=('lbfgs', 'powell', 'nm', 'bfgs'), maxiter=2000):
    rows = []
    for method in methods:
        common = dict(model=label, optimizer=method, maxiter=maxiter, n=int(model.nobs),
                      n_groups=int(model.n_groups), fixed_columns=model.exog.shape[1],
                      fixed_rank=int(np.linalg.matrix_rank(model.exog)),
                      design_condition_number=float(np.linalg.cond(model.exog)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = copy.deepcopy(model).fit(reml=True, method=method,
                                                   maxiter=maxiter, full_output=True, disp=False)
                covariance = result.cov_params().loc[result.fe_params.index, result.fe_params.index].to_numpy()
                minimum = float(np.linalg.eigvalsh((covariance + covariance.T)/2).min()) if np.isfinite(covariance).all() else np.nan
                common.update(fit_status='returned', converged=bool(result.converged),
                              log_likelihood=float(result.llf), re_var=float(np.asarray(result.cov_re)[0, 0]),
                              residual_variance=float(result.scale), fixed_cov_min_eigenvalue=minimum)
                for term in terms:
                    ci = coefficient_ci(result, term)
                    beta, se = float(result.fe_params[term]), float(result.bse_fe[term])
                    p = float(2 * stats.norm.sf(abs(beta/se))) if ci['ci_status'] == 'ok' else np.nan
                    rows.append(dict(common, term=term, beta=beta, se=se, p_two_tailed=p, **ci))
            except Exception as error:
                rows.append(dict(common, fit_status='error', error_type=type(error).__name__))
            categories = '|'.join(sorted({w.category.__name__ for w in caught}))
        for row in rows:
            if row['optimizer'] == method:
                row['warning_categories'] = categories
    return pd.DataFrame(rows)


def validate_sample(model, table, predictor, outcome, moderator=None):
    frame = pd.read_csv(table)
    mask = frame.predictor.eq(predictor) & frame.outcome.eq(outcome)
    if moderator is not None:
        mask &= frame.moderator.eq(moderator)
    selected = frame.loc[mask]
    if len(selected) != 1 or int(selected.iloc[0]['n']) != int(model.nobs):
        raise ValueError('Diagnostic sample size does not match the current aggregate result; stop.')


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Run from the authoritative project root.')
    # Only this user-run entry point imports the data-loading module.
    import analysis_utils as u
    source = ROOT / 'scripts/analysis/06_sensitivity.py'
    moderation = load_moderation_functions(source, u.__dict__)
    with contextlib.redirect_stdout(io.StringIO()):
        fc = u.load_master(fc=True)
        u.prepare_persistence_vars(fc)
        _, sample = u.get_samples(fc, check_fc_col='l_amyg-ant_vmPFC_neg_vs_neu', require_diary=False)
        pred, outcome = 'l_amyg-ant_vmPFC_neg', 'neg_persist_crossrun_mean_z_L'
        direct = capture_model(u.run_mlm, sample, pred, outcome, u.get_covariates(sample))
        validate_sample(direct, ROOT/'results/tables/05_fc_persistence/sensitivity_neg_condition/mlm.csv', pred, outcome)
        master = u.load_master(fc=False)
        u.prepare_persistence_vars(master)
        panas = master[(master['has_neg_persistence'] == 1) & (master['qc_conservative'] == 1)].copy()
        mod = capture_model(moderation['run_moderation_mlm'], panas,
                            'neg_persist_crossrun_mean_z_L', 'C5SPGN', 'C5SER', u.get_covariates(panas))
        validate_sample(mod, ROOT/'results/tables/06_persistence_affect_moderation/sensitivity_panas/reappraisal/moderation_mlm.csv',
                        'neg_persist_crossrun_mean_z_L', 'C5SPGN', 'C5SER')
    parent = ROOT/'results/diagnostics'
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='ci_fit_review_', dir=parent))
    jobs = [('negative_condition_fc', direct, ['l_amyg_ant_vmPFC_neg']),
            ('panas_na_reappraisal', mod, ['neg_persist_crossrun_mean_z_L_c', 'C5SER_c',
                                         'neg_persist_crossrun_mean_z_L_c_x_C5SER_c'])]
    for label, model, terms in jobs:
        print('Checking:', label, flush=True)
        frame = fit_diagnostics(model, label, terms)
        frame.to_csv(destination/(label+'.csv'), index=False)
        print('Saved aggregate diagnostics:', label+'.csv', flush=True)
    sources = ['scripts/analysis/analysis_utils.py', 'scripts/analysis/06_sensitivity.py',
               'scripts/analysis/05_sensitivity.py', 'scripts/analysis/confidence_intervals.py',
               'scripts/diagnostics/diagnose_ci_fits.py']
    metadata = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, statsmodels=statsmodels.__version__, reml=True,
                    maxiter=2000, methods=['lbfgs', 'powell', 'nm', 'bfgs'],
                    starting_values='independent default starts; no warm starts',
                    selection='No optimizer selected; diagnostics only',
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                    reference='https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html')
    (destination/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Finished. Review for disclosure before sharing:', destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
