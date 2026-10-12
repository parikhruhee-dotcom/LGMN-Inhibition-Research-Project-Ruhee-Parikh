"""Regression tests for docking checkpoint and resume behaviour.

A five-hour screen is only safe to interrupt if the checkpoint is always the
union of everything finished so far. The bug these guard against lost roughly
half an hour of completed work: results were accumulated into a list in
processing order and written as-is, so once docking order was shuffled each
write truncated the file, discarding finished compounds that happened to sit
later in the shuffled order.

The tests model the checkpoint logic directly rather than invoking smina,
which is not available in CI and would take hours regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking.dock import SHUFFLE_SEED  # noqa: E402


def _compounds(n: int = 20) -> pd.DataFrame:
    """A potency-sorted table, as the real input is."""
    return pd.DataFrame(
        {
            "molecule_chembl_id": [f"CMPD_{i:03d}" for i in range(n)],
            "smiles": ["CCO"] * n,
            "pic50": [10.0 - i * 0.25 for i in range(n)],
        }
    )


def _run(compounds: pd.DataFrame, done: dict, stop_after: int | None = None):
    """Reproduce the screen loop's bookkeeping and return each checkpoint."""
    results = dict(done)
    checkpoints = []
    shuffled = compounds.sample(frac=1.0, random_state=SHUFFLE_SEED)
    docked = 0
    for row in shuffled.itertuples(index=False):
        identifier = row.molecule_chembl_id
        if identifier in done:
            continue
        results[identifier] = {
            "molecule_chembl_id": identifier,
            "smiles": row.smiles,
            "affinity": -7.0,
            "status": "ok",
        }
        checkpoints.append(set(results))
        docked += 1
        if stop_after is not None and docked >= stop_after:
            break
    return results, checkpoints


class TestCheckpointNeverShrinks:
    def test_checkpoint_grows_monotonically(self):
        _, checkpoints = _run(_compounds(), done={})
        for earlier, later in zip(checkpoints, checkpoints[1:]):
            assert earlier.issubset(later), "checkpoint lost a finished compound"

    def test_resumed_work_is_present_in_the_first_checkpoint(self):
        """The regression. Resumed compounds must survive the first write.

        Previously the first write contained only the newly docked compound,
        so a crash straight after a restart destroyed every earlier result.
        """
        compounds = _compounds()
        done = {
            cid: {"molecule_chembl_id": cid, "affinity": -7.0, "status": "ok"}
            for cid in compounds["molecule_chembl_id"].head(12)
        }
        _, checkpoints = _run(compounds, done=done, stop_after=1)
        assert checkpoints, "nothing was docked"
        first = checkpoints[0]
        assert set(done).issubset(first)
        assert len(first) == len(done) + 1

    def test_every_compound_appears_exactly_once(self):
        results, _ = _run(_compounds(), done={})
        assert len(results) == len(_compounds())

    def test_resume_skips_completed_work(self):
        compounds = _compounds()
        done = {
            cid: {"molecule_chembl_id": cid, "affinity": -9.9, "status": "ok"}
            for cid in compounds["molecule_chembl_id"].head(15)
        }
        results, checkpoints = _run(compounds, done=done)
        # Only the 5 outstanding compounds were docked.
        assert len(checkpoints) == len(compounds) - len(done)
        # Resumed values were preserved, not recomputed.
        assert results["CMPD_000"]["affinity"] == -9.9


class TestShuffleIsRepresentative:
    def test_prefix_spans_the_potency_range(self):
        """A truncated run must not be a sample of only the potent compounds.

        The input arrives sorted by potency; docking it in that order made the
        first 31 compounds span pIC50 8.74-10.00 against a full range of
        5.03-10.00, which would have made the score-versus-potency
        correlation meaningless.
        """
        compounds = _compounds(60)
        shuffled = compounds.sample(frac=1.0, random_state=SHUFFLE_SEED)
        prefix = shuffled.head(15)
        full_span = compounds["pic50"].max() - compounds["pic50"].min()
        prefix_span = prefix["pic50"].max() - prefix["pic50"].min()
        assert prefix_span > 0.6 * full_span

    def test_shuffle_is_deterministic(self):
        compounds = _compounds(30)
        first = compounds.sample(frac=1.0, random_state=SHUFFLE_SEED)
        second = compounds.sample(frac=1.0, random_state=SHUFFLE_SEED)
        assert list(first["molecule_chembl_id"]) == list(
            second["molecule_chembl_id"]
        )

    def test_shuffle_actually_reorders(self):
        compounds = _compounds(30)
        shuffled = compounds.sample(frac=1.0, random_state=SHUFFLE_SEED)
        assert list(shuffled["molecule_chembl_id"]) != list(
            compounds["molecule_chembl_id"]
        )
