#!/usr/bin/env python3
"""User-run in restricted project only: three primary diary MLM interactions.

Reuses analysis 06/07 fit functions; requires agreement with saved primary rows.
Plots population fixed-effect predictions averaged over the model sample's
covariate distribution, with pointwise 95% mean-response CIs (not prediction
intervals). No participant points, IDs, or individual predictions are exported.
Review aggregate outputs for disclosure before transfer. No automatic transfer.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

CASES = [
    ('06_persistence_affect_moderation', 'neg_persist_crossrun_mean_z_L', 'C5SER', 'reappraisal', 'PA_score', 'Persistence × reappraisal', 'Amygdala persistence (Fisher z)'),
    ('06_persistence_affect_moderation', 'neg_persist_crossrun_mean_z_L', 'C5SES', 'suppression', 'PA_score', 'Persistence × suppression', 'Amygdala persistence (Fisher z)'),
    ('07_fc_affect_moderation', 'l_amyg-post_vmPFC_neg_vs_neu', 'C5SES', 'suppression', 'NA_score', 'Posterior FC × suppression', 'Posterior FC (negative − neutral Fisher z)'),
]


def linear_summary(design, beta, covariance):
    """Full covariance propagation for any fixed-effect linear contrast."""
    design = np.atleast_2d(design)
    covariance = np.asarray(covariance)
    if not all(np.isfinite(x).all() for x in (design, beta, covariance)):
        raise ValueError('Nonfinite fixed-effect inputs')
    if not np.allclose(covariance, covariance.T):
        raise ValueError('Asymmetric fixed-effect covariance')
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Invalid fixed-effect covariance')
    variance = np.einsum('ij,jk,ik->i', design, covariance, design)
    if (variance < -1e-10).any():
        raise ValueError('Negative contrast variance')
    estimate = design @ np.asarray(beta)
    se = np.sqrt(np.maximum(variance, 0))
    return estimate, se, estimate - norm.ppf(.975)*se, estimate + norm.ppf(.975)*se


def capture_fit(module, sample, predictor, outcome, moderator):
    """Capture the result returned by the unchanged primary fitting routine."""
    original = module.smf.mixedlm
    fits = []

    def factory(*args, **kwargs):
        model = original(*args, **kwargs)
        original_fit = model.fit

        def fit(*a, **kw):
            result = original_fit(*a, **kw)
            fits.append(result)
            return result
        model.fit = fit
        return model

    with patch.object(module.smf, 'mixedlm', factory):
        row = module.run_moderation_mlm(sample, predictor, outcome, moderator,
                                       module.get_covariates(sample))
    if row is None or not fits or not fits[-1].converged:
        raise ValueError('Primary refit did not converge; outputs not promoted')
    return row, fits[-1]


def check_anchor(row, path, predictor, outcome, moderator):
    saved = pd.read_csv(path)
    saved = saved[(saved.predictor == predictor) & (saved.outcome == outcome)
                  & (saved.moderator == moderator)]
    if len(saved) != 1:
        raise ValueError('Expected exactly one saved primary result')
    anchor = saved.iloc[0]
    for key in ['n', 'beta_predictor', 'beta_moderator', 'beta_interaction',
                'se_interaction', 'p_interaction', 're_var', 'log_likelihood']:
        if not np.isclose(float(row[key]), float(anchor[key]), rtol=1e-6, atol=1e-8):
            raise ValueError(f'Saved primary result mismatch: {key}; stop for review')
    if row['interaction_ci_status'] != 'ok' or anchor['interaction_ci_status'] != 'ok':
        raise ValueError('Interaction inference requires review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='New restricted review directory; must not exist')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Choose a new output directory; existing outputs are preserved.')
    root = Path.cwd()
    analysis = root / 'scripts/analysis'
    sys.path.insert(0, str(analysis))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modules = {}
    prepared = []
    slope_rows, checks = [], []
    # Validate all fits before creating any output files.
    for block, pred, mod, mod_label, outcome, title, xlabel in CASES:
        if block not in modules:
            spec = importlib.util.spec_from_file_location(block, analysis / (block+'.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[block] = module
        module = modules[block]
        df = module.load_master(fc=block.startswith('07'))
        module.prepare_persistence_vars(df)
        anchor_fc = module.PREDICTORS[0] if block.startswith('07') else None
        _, sample = module.get_samples(df, check_fc_col=anchor_fc)
        required = ['C5PAGE','sex','time_P2_P5','n_days_complete','SAMPLMAJ','M2FAMNUM','M2ID'] + ['race_'+str(i) for i in range(2,7)]
        if any(c not in sample for c in required):
            raise ValueError('Required covariate/grouping columns missing')
        row, result = capture_fit(module, sample, pred, outcome, mod)
        check_anchor(row, root/'results/tables'/block/mod_label/'moderation_mlm.csv', pred, outcome, mod)
        names = list(result.fe_params.index)
        beta = result.fe_params.to_numpy()
        covariance = result.cov_params().loc[names,names].to_numpy()
        frame = result.model.data.frame
        pred_c = pred.replace('-', '_').replace('.', '_')+'_c'
        mod_c = mod+'_c'
        interaction = pred_c+'_x_'+mod_c
        ip, im, ii = [names.index(x) for x in [pred_c, mod_c, interaction]]
        mod_mean, mod_sd = frame[mod].mean(), frame[mod].std()
        if not np.isfinite(mod_sd) or mod_sd <= 0:
            raise ValueError('Moderator has no usable variation')
        values = [-mod_sd, 0., mod_sd]
        if min(values)+mod_mean < frame[mod].min() or max(values)+mod_mean > frame[mod].max():
            raise ValueError('Mean ± SD falls outside observed moderator range; review before plotting')
        x = np.linspace(*frame[pred_c].quantile([.05,.95]).to_numpy(), 150)
        curves = []
        for label, m in zip(['−1 SD','Mean','+1 SD'], values):
            # Average design rows retains all covariate contributions, including
            # dummy proportions. Set the focal terms to each plot location.
            design = np.tile(result.model.exog.mean(axis=0), (len(x),1))
            design[:,ip], design[:,im], design[:,ii] = x, m, x*m
            est,se,lo,hi = linear_summary(design,beta,covariance)
            contrast = np.zeros(len(names));contrast[ip]=1;contrast[ii]=m
            b,s,l,h = [v[0] for v in linear_summary(contrast,beta,covariance)]
            if s <= 0:
                raise ValueError('Invalid simple-slope standard error')
            slope_rows.append(dict(model=block,predictor=pred,moderator=mod,outcome=outcome,n=int(row['n']),level=label,moderator_value=mod_mean+m,slope=b,se=s,ci_low=l,ci_high=h,p_two_sided=2*norm.sf(abs(b/s))))
            curves.append((label,est,lo,hi))
        prepared.append((title,xlabel,outcome,x,curves,row))
        checks.append(dict(model=block,predictor=pred,moderator=mod,outcome=outcome,n=int(row['n']),optimizer=row['optimizer'],converged=True,saved_result_match=True,interaction_beta=row['beta_interaction'],interaction_p=row['p_interaction']))
    fig,axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax,(title,xlabel,outcome,x,curves,row) in zip(axes,prepared):
        for (label,y,lo,hi),color,style in zip(curves,['#0072B2','#666666','#D55E00'],['--','-',':']):
            ax.plot(x,y,label=label,color=color,ls=style,lw=2)
            ax.fill_between(x,lo,hi,color=color,alpha=.12)
        ax.set(title=title,xlabel=xlabel+'\n(mean centered)',ylabel='Predicted diary '+('positive' if outcome=='PA_score' else 'negative')+' affect')
        ax.legend(title='Strategy score',frameon=False)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    fig.savefig(args.output_dir/'primary_moderation_simple_slopes.png',dpi=250)
    fig.savefig(args.output_dir/'primary_moderation_simple_slopes.pdf')
    plt.close(fig)
    pd.DataFrame(slope_rows).to_csv(args.output_dir/'simple_slopes.csv',index=False)
    pd.DataFrame(checks).to_csv(args.output_dir/'model_checks.csv',index=False)
    (args.output_dir/'README.txt').write_text('Three primary diary interactions; no sensitivity models.\nFits checked against saved primary MLM results.\nLines: fixed-effect predictions averaged over complete-case covariate distribution; random intercept zero.\nShading: pointwise two-sided 95% confidence intervals for mean response, not individual prediction intervals.\nSlopes: two-sided normal-Wald tests; nominal, not multiplicity-adjusted.\nModerator: model-specific mean and ±1 sample SD. Predictor plotted over central 90% of complete-case values. No joint-support guarantee; lines are model-based.\nNo participant points or IDs exported. Review all outputs for disclosure before sharing.\n')
    print('All three primary refits match saved results. Review outputs for disclosure:',args.output_dir)


if __name__ == '__main__':
    main()
