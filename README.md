# LGMN-Inhibition-Research-Project-Ruhee-Parikh

Computational search for small-molecule inhibitors of **legumain (LGMN)** as an
Alzheimer's disease target, using ligand-based machine learning, molecular docking and
ADMET prediction.

Legumain (asparaginyl endopeptidase, also called δ-secretase) cleaves tau at N368 and APP
at N585, so inhibiting it targets the tau and amyloid pathways at once rather than
either alone ([Wu 2019](https://doi.org/10.1016/j.pneurobio.2019.101730),
[Liao 2021](https://doi.org/10.1016/j.neuropharm.2021.108737); via PubMed).

> **The therapeutic premise is contested.** Behrendt *et al.* (2019) find that AEP
> cleaves tau inside microglia as part of normal protein degradation, and explicitly
> caution against blocking it as an Alzheimer's strategy
> ([DOI](https://doi.org/10.1016/j.nbd.2019.104518)). That sits upstream of everything
> in this repository: every method here assumes inhibition is desirable and only asks
> which molecule does it best. See [docs/REFERENCES.md](docs/REFERENCES.md).

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
| Compounds both CNS-suitable **and** non-covalent | **6** — but one series, not six leads |
| Median potency lost entering a cell (n=35, paired) | **1.53 log units, 34-fold** |
| Compounds with any published counter-screen | **4 of 180** |
| Compounds gained by searching PubChem too | **+5** — and no measurable model gain |
| Compounds docked | **180 of 180**, no failures, 172 poses in the active site |
| Docking score vs measured potency | **ρ = -0.50** non-covalent, **-0.22** covalent (size-controlled) |

**The headline result is a negative one.** Only 6 of 180 published compounds are both
predicted to reach the brain and free of a covalent warhead. Legumain inhibitor
chemistry is overwhelmingly covalent, peptide-like and polar — close to the opposite of
the profile that crosses the blood-brain barrier. For a CNS indication those two
requirements pull against each other, and almost nothing in the published chemistry
satisfies both.

Three things sharpen that conclusion rather than soften it:

- **Independent experimental support.** 35 compounds were measured on both purified
  enzyme and inside cells. They lose a median 34-fold potency crossing an ordinary cell
  membrane, and molecular weight predicts the loss (ρ=0.73, Holm p=4×10⁻⁶). The
  blood-brain barrier is a much harder membrane than that one.
- **The 6 survivors are a single chemical series**, not six independent leads — their
  common core covers 82% of the median compound. All six carry an
  ortho-phenylenediamine, a genotoxicity alert, and they are 742-fold weaker than the
  best covalent inhibitor. One of them *is* the fragment bound in the 5LUA crystal
  structure, which at least gives the series a solved binding mode.
- **Selectivity is untested for 176 of 180 compounds.** Where it was tested it looks
  good, but 12 of 14 data points are lower bounds ("no inhibition at 50 µM"), not
  measurements. An untested compound is not a selective one.
- **Docking works, but weakly, and better where its physics applies.** All 180 compounds
  were docked. After cancelling out the fact that bigger molecules score better simply
  for being bigger, the link between score and real potency survives for both groups —
  about twice as strong for non-covalent compounds (ρ = -0.50) as for covalent ones
  (ρ = -0.22). An earlier reading from half the data said covalent compounds showed no
  link at all; the full data overturned it ([docs/DOCKING.md](docs/DOCKING.md)).
- **The data ceiling is real, not an artefact of using one database.** Searching
  PubChem as well added 5 compounds and 3 scaffolds, and retraining on them changed
  nothing measurable (p = 0.98). An earlier version of that comparison appeared to show
  a real improvement; it was leakage from four tautomer duplicates, and it vanished
  once they were removed. The gap between this dataset and a usable one is hundreds of
  compounds, not tens ([docs/DATA_EXPANSION.md](docs/DATA_EXPANSION.md)).

Full write-up in [docs/RESULTS.md](docs/RESULTS.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Findings, limitations, what would move this forward |
| [docs/TARGET.md](docs/TARGET.md) | Structure survey, dyad geometry, template choice |
| [docs/DATASET.md](docs/DATASET.md) | What the data can and cannot support |
| [docs/PERMEABILITY.md](docs/PERMEABILITY.md) | Enzyme-to-cell potency loss, the permeability evidence |
| [docs/SELECTIVITY.md](docs/SELECTIVITY.md) | Counter-screens against caspases and cathepsins |
| [docs/DOCKING.md](docs/DOCKING.md) | Redocking validation and what it licenses |
| [docs/SHORTLIST.md](docs/SHORTLIST.md) | Ranked candidate table |
| [docs/CANDIDATES.md](docs/CANDIDATES.md) | Per-compound dossier: structures, poses, liabilities |
| [docs/REFERENCES.md](docs/REFERENCES.md) | The biological premise, and the case against it |
| [docs/DATA_EXPANSION.md](docs/DATA_EXPANSION.md) | Whether more data helped (it did not, measurably) |
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
make test       # 213 tests, no network required
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
| `tests/` | 213 tests, including crystallographic ground truth for warheads |
| `data/raw/` | Downloads, never hand-edited (gitignored) |
| `data/processed/` | Built by scripts (gitignored) |
| `results/` | Figures, metrics, models, docking output |
| `docs/` | Generated write-ups |

## A note on saving work

Commit **and push** at the end of every session. A Codespace that rebuilds re-clones
from GitHub, so a local-only commit is not safe either — this project lost its first
day of setup that way.
