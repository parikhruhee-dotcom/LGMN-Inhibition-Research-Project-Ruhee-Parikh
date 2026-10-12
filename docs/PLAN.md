# LGMN Inhibition — Project Plan and Status

## Goal
Identify candidate small-molecule inhibitors of **legumain (LGMN)** as a computational
Alzheimer's disease target, using ligand-based machine learning, molecular docking, and
ADMET filtering.

## Why LGMN
Legumain (asparaginyl endopeptidase, AEP) cleaves both tau and APP. Its cleavage of tau
at N368 produces aggregation-prone fragments, and it also acts on APP at N585, feeding
amyloidogenic processing. Inhibiting it is therefore a dual-pathway hypothesis rather
than a single-pathway one.

- UniProt: **Q99538** (human LGMN)
- Catalytic dyad, measured in all 18 structures: **Cys189 / His148**

## Pipeline status

### Stage 1 — Target characterisation — **done**
- [x] Survey every experimental structure (18 entries)
- [x] Locate the catalytic dyad geometrically rather than by published numbering
- [x] Distinguish mature enzyme from prolegumain
- [x] Choose a docking template on measured evidence

Written up in [TARGET.md](TARGET.md). The finding that changed the implementation:
dyad S···N distance spans 4.1–7.9 Å across activation and ligand states, so it cannot
be used to identify the site. Identification rests on nearest-pair identity plus
agreement across independent structures.

### Stage 2 — Dataset assembly — **done**
- [x] Query ChEMBL via the UniProt accession
- [x] Standardise structures (salt stripping, neutralisation, canonical form)
- [x] Convert to pIC50; aggregate replicates by median
- [x] Separate assay classes before aggregating
- [x] Flag covalent warheads, validated against crystallographic ground truth
- [x] Check class balance before modelling

**180 compounds, 55 scaffolds, 85% covalent.** Written up in [DATASET.md](DATASET.md).

The planned actives/inactives classifier was **abandoned on the evidence**. With 180
compounds over 55 scaffolds, 42 of them singletons, a scaffold-split test fold holds
~36 compounds and a couple of errors move the AUC by a lot. A regression on pIC50 with
honest error bars says more and claims less.

### Stage 3 — Machine learning — **done, with a negative result**
- [x] Morgan fingerprints and physicochemical descriptors, evaluated separately
- [x] Scaffold-grouped splits, with random splits reported alongside for contrast
- [x] Baselines: training mean and Tanimoto 1-NN
- [x] Nested cross-validation for hyperparameters
- [x] Applicability domain on every prediction

Best scaffold-split model: random forest, **MAE 0.772** vs mean-baseline 1.030, but
**R² = −0.19** and Spearman 0.47. It ranks compounds with modest skill and cannot
predict absolute potency on new chemistry. See [RESULTS.md](RESULTS.md).

### Stage 4 — Docking — **done, protocol does not validate**
- [x] Prepare receptor from the chosen template
- [x] **Redock a known ligand to validate the setup** — done for all five candidates
- [x] Screen the dataset and measure score-vs-potency directly

No template reproduced its crystal pose as the top-ranked result, though sampling found
it in four of five — a scoring failure, not a broken setup.

Screening the dataset then refined that conclusion rather than confirming it.
Controlling for ligand size, docking score tracks potency **within the non-covalent
subset** (partial ρ ≈ −0.6) and not within the covalent one, because the scoring
function has no term for bond formation. The whole-set correlation looks null only
because the covalent majority swamps a real relationship in the minority. Docking
therefore contributes to the ranking for non-covalent compounds only. See
[DOCKING.md](DOCKING.md).

### Stage 5 — ADMET — **done**
- [x] Lipinski and Veber filters
- [x] CNS MPO as a primary criterion, not an afterthought
- [x] PAINS and Brenk alerts, reported rather than used to reject
- [x] Shortlist with per-compound reasoning

**Only 6 of 180 compounds are both CNS-suitable and non-covalent.** See
[SHORTLIST.md](SHORTLIST.md).

### Stage 7 — Beyond the original plan — **done**
These were not in the first plan; each closed a limitation the project had been
carrying as an unanswered caveat.

- [x] **Permeability.** 35 compounds have paired enzyme/cell potency: median 34-fold
      loss crossing an ordinary cell membrane, and molecular weight predicts it.
      [PERMEABILITY.md](PERMEABILITY.md)
- [x] **Selectivity.** Counter-screened against nine caspases and cathepsins. Only
      4 of 180 compounds have any published data, and 12 of 14 points are lower
      bounds. [SELECTIVITY.md](SELECTIVITY.md)
- [x] **Dataset expansion.** PubChem adds 5 compounds and 3 scaffolds and changes
      nothing measurable. [DATA_EXPANSION.md](DATA_EXPANSION.md)
- [x] **The premise itself.** Cited, and the published case *against* targeting
      legumain documented. [REFERENCES.md](REFERENCES.md)

### Stage 6 — Write-up — **done**
- [x] Figures (dataset overview, model performance, ADMET cascade)
- [x] Methods and limitations, generated from artefacts on disk
- [x] Explicit statement that this is prediction, not validation

## Limitations, stated up front
- No wet-lab confirmation; every result is a hypothesis.
- Docking did not pass its own redocking test, and its scores carry signal only for
  the 27 non-covalent compounds — a subset too small to settle the question.
- ADMET predictors are trained on drug-like chemical space and extrapolate poorly.
- Legumain is a cysteine protease, so most known inhibitors are covalent, and standard
  non-covalent docking cannot score them meaningfully.
- Selectivity is unknown for 176 of 180 compounds. Where it was measured it looks
  good, but mostly as lower bounds rather than measurements.
- **The therapeutic premise is contested in the literature** (Behrendt 2019). Every
  method here assumes inhibition is desirable and none can test that assumption.

## What would move this forward

Everything answerable from public data has been answered. What remains needs an
experiment or a substantially larger dataset.

1. **Test the 6 non-covalent CNS-suitable compounds** against recombinant legumain,
   and run an Ames test on the series — all six carry an ortho-phenylenediamine.
2. **Counter-screen anything active** against caspase-3 and cathepsin B. The public
   record cannot answer this: 176 of 180 compounds have never been tested.
3. **Measure permeability directly** (PAMPA or Caco-2) on the non-covalent series, to
   see whether it escapes the 34-fold cellular penalty the rest of the set shows.
4. **Resolve the premise.** Whether AEP-mediated tau cleavage is pathogenic or part of
   normal microglial clearance is contested, and it sits upstream of everything here.

Not worth repeating: expanding beyond ChEMBL was tried and yielded 5 usable compounds
with no measurable effect. The gap to a usable dataset is hundreds of compounds.

## Environment
Python 3.12 conda environment at `.conda-env` (RDKit has no wheel for the image's
default 3.14, and `python3.12-venv` is unavailable).

```bash
make env
make test
make all
```

## Working rule
Commit **and push** at the end of every session. A Codespace rebuild re-clones from
GitHub, so a local-only commit is not safe either — this project already lost a day's
setup that way.
