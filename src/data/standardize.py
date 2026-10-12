"""Structure standardisation and covalent-warhead detection.

Shared by the dataset-preparation and screening-library code so that every
molecule in the project passes through exactly one normalisation path. If two
scripts standardised differently, a compound could appear twice under two
canonical SMILES and silently leak between train and test splits.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Electrophilic groups that react with the catalytic cysteine of a cysteine
# protease. Legumain inhibitors are overwhelmingly covalent, and non-covalent
# docking scores them meaninglessly, so they must be identifiable throughout.
#
# Each entry is (label, SMARTS). Order matters only for reporting: the first
# match is used as the primary warhead label.
COVALENT_WARHEADS: list[tuple[str, str]] = [
    # Halomethyl ketones (YVAD-CMK, AAN-CMK in the legumain crystal structures)
    ("halomethyl_ketone", "[CX3](=O)[CH2][F,Cl,Br,I]"),
    # Acyloxymethyl / aryloxymethyl ketone
    ("acyloxymethyl_ketone", "[CX3](=O)[CH2]O[CX3]=O"),
    # Diazomethyl ketone
    ("diazomethyl_ketone", "[CX3](=O)[CH]=[N+]=[N-]"),
    # Electrophilic nitrile (the 7FQI thioimidate-forming warhead class)
    ("nitrile", "[NX1]#[CX2]"),
    # Michael acceptors: vinyl ketone/amide/ester, vinyl sulfone, vinyl azole
    ("vinyl_sulfone", "[CX3]=[CX3][SX4](=O)(=O)"),
    ("acrylamide_or_enone", "[CX3]=[CX3][CX3](=O)[#6,#7,#8]"),
    ("vinyl_azole", "[CX3]=[CX3]c1ncco1"),
    ("vinyl_heteroaryl", "[CX3]=[CX3]c1[n;R][c;R][c;R][o,s,n;R]1"),
    # Aza-peptidyl Michael acceptor / aspartyl vinyl systems
    ("alpha_beta_unsaturated_ester", "[CX3]=[CX3][CX3](=O)O[#6]"),
    # Epoxide and aziridine
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine", "[NX3r3]1[#6r3][#6r3]1"),
    # Reversible-covalent carbonyl and boron warheads
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("boronic_acid", "[BX3]([OX2H1])[OX2H1]"),
    # Acyl-transfer electrophiles
    ("isocyanate_or_isothiocyanate", "[NX2]=[CX2]=[O,S]"),
    ("sulfonyl_fluoride", "[SX4](=O)(=O)[F]"),
    # Maleimide
    ("maleimide", "O=C1[CX3]=[CX3]C(=O)N1"),
]

_COVALENT_PATTERNS = [
    (label, Chem.MolFromSmarts(smarts)) for label, smarts in COVALENT_WARHEADS
]
for _label, _patt in _COVALENT_PATTERNS:
    if _patt is None:  # pragma: no cover - guards against a typo in the table
        raise ValueError(f"invalid SMARTS for warhead {_label!r}")

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(smiles: str) -> Chem.Mol | None:
    """Normalise a molecule to one canonical form.

    Cleans up valences and representations, keeps only the largest covalent
    fragment (strips counter-ions and solvates), and neutralises charges so
    that a hydrochloride salt and its free base collapse to one record.

    Returns None if the SMILES cannot be parsed.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None

    if mol.GetNumHeavyAtoms() == 0:
        return None
    return mol


def standardize_smiles(smiles: str) -> str | None:
    """Return the canonical standardised SMILES, or None if unparseable."""
    mol = standardize_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def find_warheads(mol: Chem.Mol) -> list[str]:
    """Return labels of every electrophilic warhead class present."""
    if mol is None:
        return []
    return [label for label, patt in _COVALENT_PATTERNS if mol.HasSubstructMatch(patt)]


def is_covalent(mol: Chem.Mol) -> bool:
    """True if the molecule carries any known cysteine-reactive warhead.

    This is a structural alert, not proof of a covalent mechanism: an aldehyde
    buried in a scaffold may never reach the catalytic cysteine. It is used to
    separate compounds whose docking scores cannot be trusted, where a false
    positive is cheap and a false negative is not.
    """
    return bool(find_warheads(mol))


def basic_properties(mol: Chem.Mol) -> dict[str, float]:
    """Physicochemical properties used by both modelling and ADMET filters."""
    return {
        "mw": Descriptors.MolWt(mol),
        "clogp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotb": Descriptors.NumRotatableBonds(mol),
        "arom_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "fsp3": Descriptors.FractionCSP3(mol),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
    }
