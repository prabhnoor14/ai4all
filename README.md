# Wearable Mobility & Fall Risk Detection

Using wearable sensor data to detect early signs of mobility decline and fall risk in older adults, so caregivers can intervene proactively instead of reactively.

**Research question**: Can wearable sensor data and vital sign measurements be used to distinguish normal movements from near-fall and fall events in older adults?

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> **Note for teammates**: you almost certainly don't need to download any data. Clone the repo, run the two `Setup` commands above, and everything -- code, trained models, small datasets, derived feature CSVs -- is already there. The only reason to download the raw HAR70+/SisFall data (928MB, not in this repo) is if you're specifically editing the raw-signal feature-extraction code in `src/`. See "Getting the data" below for details.

## Getting the data

Most of what you need is already in the repo: `data/cStick.csv` (tiny) and all the **derived feature CSVs** (`data/cstick_clean.csv`, `data/sisfall_features.csv`, `data/har70_windows.csv`, `data/har70_windows_clustered.csv`) are committed, since those are small (a few MB total) and expensive to regenerate. **If you're building the Streamlit app, writing up results, or working from the trained models in `models/`, you don't need to download anything** — clone, install requirements, and go.

You only need the raw sensor data below if you're modifying the feature-extraction/windowing code itself (`src/sisfall_features.py`, `src/har70_features.py`) and need to re-run it from scratch. These two are too large for GitHub (928MB combined) and are gitignored:

| Dataset | Link | Place at |
|---|---|---|
| HAR70+ | [UCI ML Repository](https://archive.ics.uci.edu/) | `data/har70plus/*.csv` (one file per subject, e.g. `501.csv`) |
| SisFall | [Kaggle mirror](https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset) | `data/SisFall_dataset/<subject>/<code>_<subject>_R<trial>.txt` |

See [docs/DATASET_OVERVIEW.md](docs/DATASET_OVERVIEW.md) for what each dataset actually contains and why it's structured this way.

## Project structure

```
src/          reusable data-loading / feature-extraction functions
eda/          exploratory analysis scripts (run these first, per dataset)
training/     model training scripts (Phase 2 -- one per dataset)
models/       trained model artifacts (.pkl)
outputs/      generated plots
data/         small + derived datasets are committed; raw HAR70+/SisFall are gitignored (see "Getting the data" above)
docs/         DATASET_OVERVIEW.md -- full write-up of dataset structure + findings
```

## Running the pipeline

Each dataset has its own EDA → feature-extraction → training path:

```bash
# cStick
python eda/phase1_cstick.py          # cleans data, stats, ANOVA -> data/cstick_clean.csv
python training/phase2_cstick.py     # trains LogReg + Decision Tree -> models/cstick_model.pkl

# SisFall
python src/sisfall_features.py       # raw signal -> trial-level features -> data/sisfall_features.csv
python eda/eda_sisfall_features.py   # sanity-check feature separation
python training/phase2_sisfall.py    # trains + evaluates overall AND on elderly (SE) subset -> models/sisfall_model.pkl

# HAR70+
python src/har70_features.py             # raw signal -> windowed movement features -> data/har70_windows.csv
python training/phase2_har70_kmeans.py   # k-means clustering -> outputs/har70_kmeans_clusters.png
```

## Key findings (see [docs/DATASET_OVERVIEW.md](docs/DATASET_OVERVIEW.md) for full detail)

- **cStick**: Logistic Regression / Decision Tree hit ~100% accuracy, but this reflects non-overlapping synthetic feature ranges, not real predictive difficulty -- documented, not treated as a real-world result.
- **SisFall**: 93% overall accuracy, 92% Fall recall -- realistic and meaningful. **Fall recall drops to 58% on the elderly (SE) subject subset alone**, a concrete, measured version of the age-representation bias in the dataset.
- **HAR70+**: unsupervised k-means (fed no activity label) discovered 3 clusters that cleanly correspond to real movement-intensity levels (sedentary / moderate walking / brisk walking), confirming the clustering approach finds genuine structure in real elderly movement data.

## Algorithms

- Logistic Regression -- binary/multiclass baseline (cStick, SisFall)
- Decision Trees -- interpretable rules for caregiving context (cStick, SisFall)
- k-Means Clustering -- unsupervised pattern discovery (HAR70+)


