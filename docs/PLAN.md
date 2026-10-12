# LGMN Inhibition — Project Plan

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
- Candidate structures to evaluate: see `docs/TARGET.md` (to be filled in)

## Pipeline

### Stage 1 — Target characterisation
- [ ] Pull the UniProt record and sequence
- [ ] Survey available PDB structures; pick a docking template on resolution,
      completeness of the active site, and activation state (legumain is a zymogen that
      self-activates at low pH — the mature form is what matters)
- [ ] Identify the catalytic dyad (Cys189 / His148) and define the binding site box

### Stage 2 — Dataset assembly
- [ ] Query ChEMBL for compounds with activity against LGMN
- [ ] Standardise structures (salt stripping, neutralisation, canonical tautomers)
- [ ] Convert IC50/Ki to pIC50; resolve duplicate measurements
- [ ] Decide the actives/inactives threshold and document the choice
- [ ] **Check the class balance before modelling.** LGMN has far less public bioactivity
      data than a kinase; if there are only tens of compounds, a classifier is not
      defensible and the project should lean on docking plus similarity search instead.

### Stage 3 — Machine learning (conditional on Stage 2 data volume)
- [ ] Descriptors + fingerprints (Morgan/ECFP4, RDKit physicochemical)
- [ ] Scaffold-based train/test split, not random — random splits inflate scores
- [ ] Baseline (logistic regression) before anything fancier
- [ ] Random forest / XGBoost with cross-validated hyperparameters
- [ ] Report ROC-AUC, PR-AUC, MCC; PR-AUC matters more under class imbalance
- [ ] Applicability domain: flag predictions outside the training chemical space

### Stage 4 — Virtual screening and docking
- [ ] Assemble a screening library (ChEMBL/ZINC/DrugBank subset)
- [ ] Prepare receptor and ligands
- [ ] Dock with AutoDock Vina or smina
- [ ] Re-dock a known ligand first to validate the setup (redocking RMSD < 2 Å)
- [ ] Rank hits, inspect poses for real contacts with the catalytic site

### Stage 5 — ADMET
- [ ] Lipinski / Veber filters
- [ ] **Blood–brain barrier permeability is the make-or-break filter here** — a CNS
      target means most otherwise-good hits will fail
- [ ] Toxicity alerts (PAINS, structural alerts)
- [ ] Shortlist with per-compound rationale

### Stage 6 — Write-up
- [ ] Figures, methods, limitations
- [ ] Be explicit that this is computational prediction, not validation

## Honest limitations to state up front
- No wet-lab confirmation; every result is a hypothesis.
- Docking scores correlate weakly with real binding affinity.
- ADMET predictors are trained on drug-like chemical space and extrapolate poorly.
- Legumain is a cysteine protease, so covalent inhibitors are plausible, and standard
  non-covalent docking scores them badly. Note which hits are covalent warheads.

## Environment
Python 3.12 virtualenv at `.venv` (RDKit does not reliably build on the image's
default 3.14).

```bash
source .venv/bin/activate
```

## Layout
```
data/raw/         # downloaded, never edited by hand
data/processed/   # cleaned datasets produced by scripts in src/data
src/data/         # acquisition + cleaning
src/models/       # ML
src/docking/      # receptor/ligand prep, docking runs
src/admet/        # filters
results/          # figures, trained models, docking output
notebooks/        # exploration only; anything load-bearing moves into src/
docs/             # plan, target notes, write-up
```

## Working rule
Commit at the end of every session. Uncommitted work in a Codespace does not survive a
container rebuild — this project already lost a day's setup that way.
