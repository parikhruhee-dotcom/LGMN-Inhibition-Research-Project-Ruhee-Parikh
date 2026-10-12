"""Drug-likeness, CNS penetration and structural-alert filters.

For an Alzheimer's target, brain penetration is the filter that decides the
project. A compound that inhibits legumain beautifully in a cuvette and cannot
cross the blood-brain barrier is not a lead for a CNS indication, and most
otherwise-attractive hits fail here. So the CNS MPO score is treated as a
primary criterion rather than an afterthought appended to Lipinski.

Everything in this module is a computed prediction from 2D structure. These
models are trained on drug-like chemical space and extrapolate poorly; they
rank and triage, they do not measure.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# --- PAINS and other structural alerts -------------------------------------
# Pan-assay interference compounds hit many targets by mechanisms such as
# redox cycling, aggregation or covalent protein modification. For a cysteine
# protease these alerts need care: the reactivity that makes a compound a PAINS
# liability is also the mechanism of a legitimate covalent inhibitor, so an
# alert here is a reason to look, not an automatic rejection.
_PAINS_PARAMS = FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
_PAINS_CATALOG = FilterCatalog(_PAINS_PARAMS)

_BRENK_PARAMS = FilterCatalogParams()
_BRENK_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK_CATALOG = FilterCatalog(_BRENK_PARAMS)


def structural_alerts(mol: Chem.Mol) -> dict:
    """PAINS and Brenk alert matches."""
    pains = [entry.GetDescription() for entry in _PAINS_CATALOG.GetMatches(mol)]
    brenk = [entry.GetDescription() for entry in _BRENK_CATALOG.GetMatches(mol)]
    return {
        "pains_alerts": len(pains),
        "pains_names": "|".join(sorted(set(pains)))[:200],
        "brenk_alerts": len(brenk),
        "brenk_names": "|".join(sorted(set(brenk)))[:200],
    }


# --- Oral drug-likeness ----------------------------------------------------


def lipinski(mol: Chem.Mol) -> dict:
    """Lipinski rule-of-five violations.

    The rule describes oral absorption, not potency, and one violation is
    routinely tolerated in marketed drugs, so the violation count is reported
    rather than a pass/fail verdict.
    """
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum(
        [mw > 500, clogp > 5, hbd > 5, hba > 10]
    )
    return {
        "mw": mw,
        "clogp": clogp,
        "hbd": hbd,
        "hba": hba,
        "lipinski_violations": int(violations),
    }


def veber(mol: Chem.Mol) -> dict:
    """Veber criteria for oral bioavailability: flexibility and polarity."""
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    return {
        "rotatable_bonds": rotb,
        "tpsa": tpsa,
        "veber_pass": bool(rotb <= 10 and tpsa <= 140),
    }


# --- CNS penetration -------------------------------------------------------
# CNS MPO (Wager et al., ACS Chem Neurosci 2010) scores six properties, each
# mapped to 0-1 by a piecewise-linear desirability function, and sums them to
# 0-6. A score >= 4 is the usual marker of a CNS-suitable profile.
#
# Each entry is (good_threshold, bad_threshold): the property scores 1.0 at or
# beyond `good`, 0.0 at or beyond `bad`, and interpolates between. Where `bad`
# is below `good` the desirability decreases with the property value.
_MPO_RAMPS = {
    # Lower is better for all four of these.
    "clogp": (3.0, 5.0),
    "clogd": (2.0, 4.0),
    "mw": (360.0, 500.0),
    "pka_most_basic": (8.0, 10.0),
    # TPSA is doubly bounded: too polar does not cross, too greasy is
    # non-specific, so it is handled separately below.
}


def _ramp(value: float, good: float, bad: float) -> float:
    """Piecewise-linear desirability, clamped to [0, 1]."""
    if np.isnan(value):
        return np.nan
    if good == bad:
        return 1.0 if value <= good else 0.0
    if good < bad:  # lower is better
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        return float((bad - value) / (bad - good))
    # higher is better
    if value >= good:
        return 1.0
    if value <= bad:
        return 0.0
    return float((value - bad) / (good - bad))


def _tpsa_desirability(tpsa: float) -> float:
    """TPSA desirability: best between 40 and 90 A^2.

    Below ~20 the compound is too lipophilic to be selective; above ~120 it
    will not cross the barrier passively.
    """
    if tpsa <= 20.0:
        return 0.0
    if tpsa < 40.0:
        return float((tpsa - 20.0) / 20.0)
    if tpsa <= 90.0:
        return 1.0
    if tpsa < 120.0:
        return float((120.0 - tpsa) / 30.0)
    return 0.0


def _most_basic_pka(mol: Chem.Mol) -> float:
    """Crude estimate of the most basic pKa.

    A proper calculation needs a trained pKa model, which is not available
    offline. This approximates by basic-centre class: an aliphatic amine is
    strongly basic, an aniline or heteroaromatic nitrogen much less so. It is
    the weakest term in the MPO score and is reported separately so its
    influence stays visible.
    """
    patterns = [
        ("[NX3;H2;!$(NC=O);!$(Nc)]", 10.5),   # primary aliphatic amine
        ("[NX3;H1;!$(NC=O);!$(Nc)]", 10.0),   # secondary aliphatic amine
        ("[NX3;H0;!$(NC=O);!$(Nc);!$(N=*)]", 9.5),  # tertiary aliphatic amine
        ("[nX2]1ccccc1", 5.2),                # pyridine-like
        ("[NX3;H2]c", 4.6),                   # aniline
        ("[NX3][CX3]=[NX2]", 11.5),           # amidine/guanidine
    ]
    best = np.nan
    for smarts, pka in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            best = pka if np.isnan(best) else max(best, pka)
    return best if not np.isnan(best) else 0.0


def cns_mpo(mol: Chem.Mol) -> dict:
    """CNS MPO score (0-6) and its components."""
    clogp = Crippen.MolLogP(mol)
    # clogD (distribution coefficient at pH 7.4) is approximated as clogP
    # reduced for ionisable centres, since a proper clogD needs a pKa model.
    pka = _most_basic_pka(mol)
    clogd = clogp - (1.0 if pka >= 8.0 else 0.0)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)

    components = {
        "mpo_clogp": _ramp(clogp, *_MPO_RAMPS["clogp"]),
        "mpo_clogd": _ramp(clogd, *_MPO_RAMPS["clogd"]),
        "mpo_mw": _ramp(mw, *_MPO_RAMPS["mw"]),
        "mpo_tpsa": _tpsa_desirability(tpsa),
        "mpo_hbd": _ramp(float(hbd), 0.5, 3.5),
        "mpo_pka": _ramp(pka, *_MPO_RAMPS["pka_most_basic"]),
    }
    score = float(np.nansum(list(components.values())))
    return {
        **components,
        "cns_mpo": score,
        "cns_mpo_pass": bool(score >= 4.0),
        "estimated_pka": pka,
        "clogd": clogd,
    }


def bbb_rules(mol: Chem.Mol) -> dict:
    """Simple published cutoffs for CNS penetration, as a cross-check.

    These agree with MPO often enough to be useful and disagree often enough
    to be worth printing: a compound passing one and failing the other is a
    genuinely borderline case rather than a confident call.
    """
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    clogp = Crippen.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    charge = Chem.GetFormalCharge(mol)

    return {
        # Commonly cited CNS property space.
        "bbb_mw_ok": bool(mw <= 450),
        "bbb_tpsa_ok": bool(tpsa <= 90),
        "bbb_hbd_ok": bool(hbd <= 3),
        "bbb_clogp_ok": bool(1.0 <= clogp <= 4.0),
        "bbb_rotb_ok": bool(rotb <= 8),
        # A permanently charged species will not cross passively.
        "bbb_neutral": bool(charge == 0),
    }


# A compound must clear both the MPO score and most of the rule checks to be
# called a CNS candidate.
#
# Neither test alone is sufficient, and the failure mode is concrete: sucrose
# scores exactly 4.0 on CNS MPO and so "passes". That is not a bug in the
# implementation -- the clogP, clogD and pKa desirability functions all reward
# low lipophilicity monotonically, so an extremely polar sugar collects three
# free points and needs only one more to reach the threshold. MPO was designed
# to rank drug-like candidates against each other, not to reject molecules
# nothing would mistake for drugs. The rule checks catch exactly that case:
# sucrose passes 3 of 6, diazepam all 6.
MIN_BBB_RULES = 5


def cns_candidate(row: dict) -> bool:
    """Combined CNS verdict: MPO score and rule checks must both hold."""
    return bool(
        row.get("cns_mpo_pass")
        and row.get("bbb_rules_passed", 0) >= MIN_BBB_RULES
    )


def profile(smiles: str) -> dict | None:
    """Full ADMET profile for one SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    row: dict = {"smiles": smiles}
    row.update(lipinski(mol))
    row.update(veber(mol))
    row.update(cns_mpo(mol))
    row.update(bbb_rules(mol))
    row.update(structural_alerts(mol))

    bbb_checks = [v for k, v in row.items() if k.startswith("bbb_")]
    row["bbb_rules_passed"] = int(sum(bool(v) for v in bbb_checks))
    row["bbb_rules_total"] = len(bbb_checks)
    row["cns_candidate"] = cns_candidate(row)
    row["qed"] = Descriptors.qed(mol)
    return row
