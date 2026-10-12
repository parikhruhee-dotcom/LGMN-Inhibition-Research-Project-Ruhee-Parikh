"""Molecular featurisation and applicability domain.

Two feature blocks are offered because they fail differently on a dataset of
this size. Morgan fingerprints are high-dimensional and sparse, which suits
nearest-neighbour similarity but gives a 180-row regression far more columns
than rows. The physicochemical descriptor block is small enough to fit but
cannot distinguish two compounds that differ only in substituent position.
Both are evaluated rather than one being assumed better.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ECFP4 equivalent. 2048 bits keeps bit collisions rare without making the
# matrix unmanageable at this sample size.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_MORGAN = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

# Descriptors chosen to span size, lipophilicity, polarity, flexibility and
# shape without including the dozens of near-duplicate RDKit descriptors,
# which would add collinearity rather than information.
DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "MolMR",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Kappa2",
    "LabuteASA",
    "HallKierAlpha",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "qed",
]

_DESCRIPTOR_FUNCS = [
    (name, getattr(Descriptors, name))
    for name in DESCRIPTOR_NAMES
    if hasattr(Descriptors, name)
]


def morgan_fingerprint(mol: Chem.Mol):
    """Return the RDKit bit-vector fingerprint (needed for similarity)."""
    return _MORGAN.GetFingerprint(mol)


def morgan_array(mol: Chem.Mol) -> np.ndarray:
    """Return the fingerprint as a float array for scikit-learn."""
    array = np.zeros((MORGAN_BITS,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), array)
    return array


def descriptor_array(mol: Chem.Mol) -> np.ndarray:
    """Return the physicochemical descriptor vector.

    Individual descriptors can fail on unusual valences; a failure yields NaN
    for that column rather than discarding the whole molecule, and the
    modelling pipeline imputes it.
    """
    values = []
    for _name, func in _DESCRIPTOR_FUNCS:
        try:
            values.append(float(func(mol)))
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


def featurize(smiles: list[str], kind: str) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from SMILES.

    `kind` is one of 'morgan', 'descriptors' or 'both'. Returns the matrix and
    the feature names, so that feature importances stay interpretable.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(mol is None for mol in mols):
        bad = [s for s, mol in zip(smiles, mols) if mol is None]
        raise ValueError(f"unparseable SMILES reached featurisation: {bad[:3]}")

    blocks = []
    names: list[str] = []
    if kind in ("morgan", "both"):
        blocks.append(np.vstack([morgan_array(mol) for mol in mols]))
        names += [f"morgan_{i}" for i in range(MORGAN_BITS)]
    if kind in ("descriptors", "both"):
        blocks.append(np.vstack([descriptor_array(mol) for mol in mols]))
        names += [name for name, _ in _DESCRIPTOR_FUNCS]
    if not blocks:
        raise ValueError(f"unknown feature kind {kind!r}")

    return np.hstack(blocks), names


def tanimoto_to_training_set(
    query_smiles: list[str], training_smiles: list[str]
) -> np.ndarray:
    """Max Tanimoto similarity of each query to any training compound.

    This is the applicability domain measure. A model trained on 180 compounds
    across 55 scaffolds knows almost nothing about chemistry unlike what it has
    seen, and a prediction for a dissimilar compound is extrapolation dressed
    up as a number. Reporting the nearest-neighbour similarity alongside every
    prediction makes that visible instead of hiding it.
    """
    training_fps = [
        morgan_fingerprint(mol)
        for mol in (Chem.MolFromSmiles(s) for s in training_smiles)
        if mol is not None
    ]
    if not training_fps:
        raise ValueError("no valid training structures for similarity")

    out = np.zeros(len(query_smiles), dtype=np.float64)
    for index, smiles in enumerate(query_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out[index] = np.nan
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(
            morgan_fingerprint(mol), training_fps
        )
        out[index] = max(similarities) if similarities else 0.0
    return out
