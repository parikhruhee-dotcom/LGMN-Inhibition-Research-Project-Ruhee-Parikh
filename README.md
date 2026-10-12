# LGMN-Inhibition-Research-Project-Ruhee-Parikh

Computational search for small-molecule inhibitors of **legumain (LGMN)** as an
Alzheimer's disease target, using ligand-based machine learning, molecular docking and
ADMET prediction.

Legumain (asparaginyl endopeptidase) cleaves tau at N368 and APP at N585, so inhibiting
it targets the tau and amyloid pathways at once rather than either alone.

> **Everything here is a computational prediction.** No compound has been tested
> experimentally by this project. Potency values are literature measurements from
> ChEMBL; ADMET columns are predictions from 2D structure.

## Findings so far

| | |
|---|---|
| Human legumain structures surveyed | 18 (17 mature, 1 zymogen) |
| Catalytic dyad, found geometrically in every chain | **Cys189 / His148** |
| Compounds with usable human biochemical potency | **180** across 55 scaffolds |
| Carrying a covalent warhead | **153 (85%)** |
| Best QSAR model, scaffold split | MAE **0.772** log units (mean baseline 1.030) |
| Same model, R² | **−0.19** — ranks compounds, cannot predict absolute potency |
| Compounds both CNS-suitable **and** non-covalent | **6** |

**The headline result is a negative one.** Only 6 of 180 published compounds are both
predicted to reach the brain and free of a covalent warhead. Legumain inhibitor
chemistry is overwhelmingly covalent, peptide-like and polar — close to the opposite of
the profile that crosses the blood-brain barrier. For a CNS indication those two
requirements pull against each other, and almost nothing in the published chemistry
satisfies both.

Full write-up in [docs/RESULTS.md](docs/RESULTS.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Findings, limitations, what would move this forward |
| [docs/TARGET.md](docs/TARGET.md) | Structure survey, dyad geometry, template choice |
| [docs/DATASET.md](docs/DATASET.md) | What the data can and cannot support |
| [docs/PERMEABILITY.md](docs/PERMEABILITY.md) | Enzyme-to-cell potency loss, the permeability evidence |
| [docs/DOCKING.md](docs/DOCKING.md) | Redocking validation and what it licenses |
| [docs/SHORTLIST.md](docs/SHORTLIST.md) | Ranked candidate table |
| [docs/PLAN.md](docs/PLAN.md) | Pipeline design |

## Three things that were nearly wrong

Recorded because each would have produced confident, publishable-looking, wrong numbers.

1. **A dyad distance cutoff selected the wrong structures.** The Cys189–His148 S···N
   distance spans 4.1 Å in the zymogen, ~5.4 Å where an inhibitor has covalently
   modified the thiol, and ~7.9 Å in protein-inhibitor complexes. A 4.5 Å cutoff
   therefore matched only the 3.9 Å *inactive* zymogen. Identification now rests on
   nearest-pair identity plus agreement across 18 independent structures.

2. **Pooling two assay types manufactured "replicate noise."** 30 compounds appeared to
   disagree by 1–3 log units between measurements. They came from one paper using two
   assays — recombinant enzyme and a HEK293-A cell-based assay that runs 1–2 logs
   weaker. ChEMBL labels both `assay_type: B`, so separating them needs `bao_format`.
   Fixing it cut unusable labels from 31 compounds to 1 and *grew* the dataset.

3. **Three "non-covalent" leads were covalent.** The compounds ranked as the best
   brain-penetrant non-covalent candidates were terminal alkynes on an asparagine-mimic
   scaffold — the alkyne analogues of WR9, which is covalently bound 1.73 Å from the
   catalytic thiol in 7FQH. The warhead detector is now validated against every ligand
   resolved in a legumain active site, labelled by measured bond distance.

## Setup

RDKit has no wheel for the Codespace image's default Python 3.14, and `python3.12-venv`
is unavailable, so the project uses a local conda environment:

```bash
make env          # or the two commands below
/opt/conda/bin/conda create -y -p ./.conda-env python=3.12
./.conda-env/bin/pip install -r requirements.txt
/opt/conda/bin/conda install -y -p ./.conda-env -c conda-forge smina openbabel
```

## Running

```bash
make test       # 90 tests, no network required
make data       # fetch ChEMBL, build the modelling set, plot it
make target     # survey structures, pick and prepare a docking template
make model      # train and evaluate QSAR models
make docking    # validate the protocol, then screen  (slow: hours on 2 cores)
make admet      # ADMET/CNS filters and shortlist
make report     # regenerate the write-up from whatever has been run
make all        # everything, in order
```

`make report` reads only files on disk, so a section that has not been run says so
rather than quoting a stale number.

## Layout

| Path | Contents |
|---|---|
| `src/data/` | ChEMBL fetch, standardisation, dataset build, EDA |
| `src/models/` | Featurisation, applicability domain, QSAR training |
| `src/docking/` | Structure survey, receptor prep, protocol validation, docking |
| `src/admet/` | Drug-likeness, CNS and structural-alert filters |
| `tests/` | 90 tests, including crystallographic ground truth for warheads |
| `data/raw/` | Downloads, never hand-edited (gitignored) |
| `data/processed/` | Built by scripts (gitignored) |
| `results/` | Figures, metrics, models, docking output |
| `docs/` | Generated write-ups |

## A note on saving work

Commit **and push** at the end of every session. A Codespace that rebuilds re-clones
from GitHub, so a local-only commit is not safe either — this project lost its first
day of setup that way.
