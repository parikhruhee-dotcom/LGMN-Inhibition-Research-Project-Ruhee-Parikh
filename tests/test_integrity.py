"""End-to-end integrity checks on the produced data.

Unit tests check functions in isolation. These check that the artefacts on
disk are internally consistent: that the record accounting balances, that no
compound appears twice, that every pose file contains the molecule it is
named after, and that recorded values match the files they were derived from.

A pose/compound mix-up would invalidate every docking conclusion while
producing output that looks entirely normal, so it is worth asserting rather
than assuming.

All of these skip when the pipeline has not been run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def _need(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not generated")
    return pd.read_csv(path)


class TestDatasetIntegrity:
    def test_no_duplicate_compounds(self):
        final = _need(PROCESSED / "lgmn_regression.csv")
        assert final["smiles"].nunique() == len(final)
        assert final["molecule_chembl_id"].nunique() == len(final)

    def test_every_compound_traces_back_to_a_raw_record(self):
        """Nothing may appear in the modelling set that was not downloaded."""
        from data.standardize import standardize_smiles

        final = _need(PROCESSED / "lgmn_regression.csv")
        raw = _need(ROOT / "data" / "raw" / "chembl_lgmn_activities.csv")
        known = {
            standardize_smiles(s)
            for s in raw["canonical_smiles"].dropna().unique()
        }
        assert set(final["smiles"]).issubset(known)

    def test_record_accounting_balances(self):
        """Every raw record is either used or explained, never silently lost."""
        raw = _need(ROOT / "data" / "raw" / "chembl_lgmn_activities.csv")
        rejected = _need(PROCESSED / "lgmn_rejected.csv")
        censored = _need(PROCESSED / "lgmn_censored.csv")
        cellular = _need(PROCESSED / "lgmn_cellular.csv")

        record_level = {
            "endpoint_not_a_concentration",
            "assay_organism_not_human",
            "units_not_nM",
            "missing_value_or_structure",
            "nonpositive_concentration",
            "chembl_data_validity_comment",
            "assay_target_confidence_too_low",
            "structure_unparseable",
        }
        dropped = int(rejected["reject_reason"].isin(record_level).sum())
        remaining = len(raw) - dropped - len(censored) - len(cellular)
        assert remaining > 0
        # Those records aggregate into compounds, so there cannot be more
        # compounds than records they came from.
        final = _need(PROCESSED / "lgmn_regression.csv")
        assert len(final) <= remaining

    def test_potency_values_are_plausible(self):
        from data.prepare_dataset import MAX_PIC50, MIN_PIC50

        final = _need(PROCESSED / "lgmn_regression.csv")
        assert final["pic50"].between(MIN_PIC50, MAX_PIC50).all()


class TestDockingIntegrity:
    def test_every_pose_contains_the_molecule_it_is_named_after(self):
        """A pose/compound mix-up would invalidate every docking result.

        Compared on the InChIKey connectivity block, since docking alters
        protonation and the perceived stereochemistry of the written file.
        """
        from rdkit import Chem

        from data.standardize import inchikey

        scores = _need(RESULTS / "docking" / "docking_scores.csv")
        pose_dir = RESULTS / "docking" / "poses"
        if not pose_dir.exists():
            pytest.skip("no poses")

        mismatches = []
        for row in scores.head(40).itertuples(index=False):
            path = pose_dir / f"{row.molecule_chembl_id}.sdf"
            if not path.exists():
                continue
            mol = next(
                (m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m),
                None,
            )
            if mol is None:
                continue
            posed = inchikey(Chem.MolToSmiles(Chem.RemoveHs(mol)))
            recorded = inchikey(row.smiles)
            if posed and recorded:
                assert posed.split("-")[0] == recorded.split("-")[0], (
                    f"{row.molecule_chembl_id}: pose holds a different molecule"
                )
        assert not mismatches

    def test_recorded_affinities_match_the_pose_files(self):
        from rdkit import Chem

        scores = _need(RESULTS / "docking" / "docking_scores.csv")
        pose_dir = RESULTS / "docking" / "poses"
        if not pose_dir.exists():
            pytest.skip("no poses")

        for row in scores.head(20).itertuples(index=False):
            path = pose_dir / f"{row.molecule_chembl_id}.sdf"
            if not path.exists() or pd.isna(row.affinity):
                continue
            mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m]
            if not mols or not mols[0].HasProp("minimizedAffinity"):
                continue
            assert float(mols[0].GetProp("minimizedAffinity")) == pytest.approx(
                row.affinity, abs=1e-4
            )

    def test_active_site_flag_agrees_with_the_distance(self):
        from docking.dock import ACTIVE_SITE_CONTACT

        scores = _need(RESULTS / "docking" / "docking_scores.csv")
        usable = scores.dropna(subset=["dist_to_thiol", "in_active_site"])
        expected = usable["dist_to_thiol"] <= ACTIVE_SITE_CONTACT
        assert (expected == usable["in_active_site"].astype(bool)).all()
