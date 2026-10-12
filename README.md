# LGMN-Inhibition-Research-Project-Ruhee-Parikh

Computational Alzheimer's project trying to inhibit LGMN through machine learning,
molecular docking, and ADMET.

Legumain (LGMN, asparaginyl endopeptidase) cleaves tau at N368 and APP at N585, so
inhibiting it targets both the tau and amyloid pathways rather than just one. This
project searches for candidate small-molecule inhibitors computationally.

**This produces predictions, not validated results.** Nothing here has wet-lab support.

See [docs/PLAN.md](docs/PLAN.md) for the full pipeline and its known limitations.

## Setup

RDKit needs Python 3.12, and this Codespace image defaults to 3.14, so the project uses
a local conda environment:

```bash
# one time
/opt/conda/bin/conda create -y -p ./.conda-env python=3.12
./.conda-env/bin/pip install -r requirements.txt

# each session
source /opt/conda/etc/profile.d/conda.sh && conda activate ./.conda-env
```

## Running

```bash
python src/data/fetch_chembl.py     # pull LGMN bioactivity data from ChEMBL
```

## Layout

| Path | Contents |
|---|---|
| `data/raw/` | Downloads, never edited by hand (gitignored) |
| `data/processed/` | Cleaned datasets built by scripts (gitignored) |
| `src/data/` | Data acquisition and cleaning |
| `src/models/` | Machine learning |
| `src/docking/` | Receptor/ligand prep and docking |
| `src/admet/` | Drug-likeness and ADMET filters |
| `results/` | Figures, trained models, docking output |
| `notebooks/` | Exploration only |
| `docs/` | Plan, target notes, write-up |

## A note on saving work

Commit at the end of every working session. A Codespace that rebuilds discards anything
not committed to git — this project lost its first day of setup that way.
