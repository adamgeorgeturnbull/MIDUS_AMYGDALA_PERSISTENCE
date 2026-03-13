"""
tier_utils.py

Shared tier assignment utilities for analysis scripts 02–07.

Tier definitions:
- primary: Left amygdala, cross-run negative persistence, negative condition (FC),
           PA_score/NA_score (daily diary), conservative sample
- secondary: Right amygdala (same otherwise)
- sensitivity: Bilateral, positive/neutral conditions, concatenated persistence,
               PANAS outcomes, log-transforms, safety_vs_threat (derived),
               neg_vs_neu/neg_vs_pos contrasts, full sample

Row-level tier = worst tier among its components.
"""

# ============================================================================
# Tier order (lower = more primary)
# ============================================================================
_TIER_ORDER = {"primary": 0, "secondary": 1, "sensitivity": 2}
_TIER_FROM_ORDER = {v: k for k, v in _TIER_ORDER.items()}


def combine_tiers(*tiers):
    """Return worst (least primary) tier among inputs."""
    worst = max(_TIER_ORDER[t] for t in tiers)
    return _TIER_FROM_ORDER[worst]


# ============================================================================
# Persistence variable tier
# ============================================================================
def get_persistence_tier(var_name):
    """Assign tier based on persistence variable name.

    Primary: left amygdala, cross-run negative (amygdala replication)
    Secondary: right amygdala, cross-run negative; vmPFC seeds (comparison ROI)
    Sensitivity: positive persistence, concatenated persistence
    """
    if var_name == "neg_persist_crossrun_mean_z_L":
        return "primary"
    if var_name == "neg_persist_crossrun_mean_z_R":
        return "secondary"
    # vmPFC persistence: same construct, comparison ROI → secondary
    if "vmPFC" in var_name and "neg_image" in var_name:
        return "secondary"
    return "sensitivity"


# ============================================================================
# Affect variable tier
# ============================================================================
def get_affect_tier(var_name):
    """Assign tier based on affect variable name.

    Primary: daily diary PA and NA (raw)
    Sensitivity: log-transforms, PANAS outcomes
    """
    if var_name in ("PA_score", "NA_score"):
        return "primary"
    return "sensitivity"


# ============================================================================
# FC variable tier (for condition-level scripts 04, 05, 07)
# ============================================================================
def get_fc_tier(fc_var, condition=None, seed_target=None):
    """Assign tier based on FC variable properties.

    Primary: negative condition, left amygdala, ant_vmPFC or post_vmPFC
    Secondary: negative condition, right amygdala, ant_vmPFC or post_vmPFC
    Sensitivity: other conditions, contrasts, safety_vs_threat derived

    Parameters
    ----------
    fc_var : str
        Full FC variable name (e.g., "l_amyg-ant_vmPFC_neg")
    condition : str, optional
        Condition label if already parsed (e.g., "neg", "neu", "neg_vs_neu")
    seed_target : str, optional
        Seed-target label if already parsed (e.g., "l_amyg-ant_vmPFC")
    """
    # Parse condition from fc_var if not provided
    if condition is None:
        if "safety_vs_threat" in fc_var:
            condition = fc_var.rsplit("_", 1)[-1]
        elif "_vs_" in fc_var:
            # e.g., l_amyg-ant_vmPFC_neg_vs_neu → neg_vs_neu
            condition = "_".join(fc_var.rsplit("_", 3)[-3:])
        else:
            condition = fc_var.rsplit("_", 1)[-1]

    # Non-negative conditions are always sensitivity
    if condition != "neg":
        return "sensitivity"

    # Safety_vs_threat derived variables are sensitivity
    if "safety_vs_threat" in fc_var:
        return "sensitivity"

    # Parse seed from fc_var if not provided
    if seed_target is None:
        if "safety_vs_threat" in fc_var:
            seed_target = fc_var.rsplit("_", 1)[0]
        elif "_vs_" in fc_var:
            seed_target = fc_var.rsplit("_", 3)[0]
        else:
            seed_target = fc_var.rsplit("_", 1)[0]

    # Left amygdala = primary, right = secondary
    if seed_target.startswith("l_amyg"):
        return "primary"
    if seed_target.startswith("r_amyg"):
        return "secondary"

    return "sensitivity"


# ============================================================================
# Convenience: print results grouped by tier
# ============================================================================
def print_by_tier(results, format_row_fn, p_col="p"):
    """Print results grouped by tier (primary → secondary → sensitivity).

    Parameters
    ----------
    results : DataFrame
        Must have a 'tier' column.
    format_row_fn : callable
        Function(row) → str for each result row.
    p_col : str
        Column name for p-value (used for significance counting).
    """
    for tier in ["primary", "secondary", "sensitivity"]:
        tier_results = results[results["tier"] == tier]
        if len(tier_results) == 0:
            continue

        n_sig = (tier_results[p_col] < 0.05).sum()
        print(f"\n  === {tier.upper()} ({n_sig}/{len(tier_results)} significant) ===")

        if tier == "sensitivity":
            # Condensed: just list significant ones
            sig = tier_results[tier_results[p_col] < 0.05]
            if len(sig) > 0:
                for _, row in sig.iterrows():
                    print(f"    {format_row_fn(row)}")
            else:
                print("    (none significant)")
        else:
            for _, row in tier_results.iterrows():
                print(f"    {format_row_fn(row)}")
