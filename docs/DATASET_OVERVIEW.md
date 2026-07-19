# Dataset Overview & EDA Findings

This document explains what each of our three datasets actually is, how it was collected, its structure, and what our exploratory analysis (see `eda/`) found.

## Quick glossary

- **Accelerometer**: measures acceleration (how fast velocity is changing) along X/Y/Z axes. Sitting still reads roughly constant "gravity" on one axis; a fall produces a sharp spike as the body decelerates suddenly on impact.
- **Gyroscope**: measures rotation rate (how fast the body/limb is turning). Useful for catching stumbles/twists, not just straight-line motion.
- **HRV (Heart Rate Variability)**: variation in time between heartbeats. Lower HRV is generally associated with physiological stress.
- **SpO2**: blood oxygen saturation (%). Drops can indicate physical distress.
- **Sampling rate (Hz)**: how many readings per second the sensor records. 50 Hz = one reading every 20ms; 200 Hz = one every 5ms.
- **Windowing**: raw sensor data is one long stream of readings, not one row per "event." To use it with models like Logistic Regression or Decision Trees (which expect one fixed-size row per example), you slice the stream into fixed-length time windows (e.g., 2 seconds) and compute summary statistics (mean, std, min, max, energy) per window — turning a signal into a row of features.

---

## 1. cStick — "Elderly Fall Prediction and Detection" (Kaggle)

**What it is**: a small, likely *simulated* dataset representing readings from a "smart stick" device — a walking cane instrumented with sensors. Each row is one snapshot reading, already labeled with an outcome.

**Source**: [kaggle.com/datasets/laavanya/elderly-fall-prediction-and-detection](https://www.kaggle.com/datasets/laavanya/elderly-fall-prediction-and-detection)

**Structure**: `data/cStick.csv` — 2,039 rows × 7 columns

| Column | Meaning |
|---|---|
| `Distance` | distance reading (likely from an ultrasonic/IR sensor on the stick) |
| `Pressure` | grip/load pressure category (0/1/2) |
| `HRV` | heart rate variability |
| `Sugar level` | blood sugar reading |
| `SpO2` | blood oxygen saturation |
| `Accelerometer` | binary motion flag (0/1) |
| `Decision` | **target label**: 0 = Normal, 1 = Near-Miss/Stumble, 2 = Fall |

**EDA findings** (`eda/eda_cstick.py`, plots in `outputs/cstick_*.png`):
- Perfectly balanced classes (~34% / 33% / 33%) and **zero** missing values or duplicates — real-world data is essentially never this clean, which is why we believe this is simulated rather than recorded from actual patients.
- **`Pressure` has a correlation of 1.00 with `Decision`** — it's a direct stand-in for the label, not an independent signal. Including it would let a model "cheat" by just reading the label back out.
- `Accelerometer` is also highly correlated (0.87) and looks like a coarse proxy, not real accelerometer data (it's binary — real accel data isn't).
- The other four features (`Distance`, `HRV`, `Sugar level`, `SpO2`) are genuinely, strongly separated across classes (ANOVA p ≈ 0 for all).

**Implication for modeling**: usable as our tabular vitals-based classifier, but **drop `Pressure` and `Accelerometer`** first. Document the "likely simulated data" caveat in the write-up — it means accuracy numbers from this dataset alone shouldn't be treated as proof the approach works on real patients.

**Phase 1 status: done** (`eda/phase1_cstick.py`, cleaning logic in `src/data_prep.py`). With the two leaky columns removed, the remaining four features still separate the classes strongly on their own (e.g. mean `Distance` drops from ~60 in Normal to ~5 in Fall; `SpO2` drops from ~95% to ~70%; ANOVA p ≈ 0 for all four). No feature is left with a suspicious 1.00 correlation to `Decision` — see `outputs/cstick_clean_corr.png`. Cleaned dataset saved to `data/cstick_clean.csv` for Phase 2 training.

---

## 2. HAR70+ (UCI Machine Learning Repository)

**What it is**: real accelerometer recordings from 18 adults aged 70+, each wearing two sensors (one on the lower back, one on the thigh) during free-living daily activity. A human annotator watched synchronized video and labeled what activity the person was doing at each moment. This is an **activity recognition** dataset — it tells you *what movement is happening*, not whether a fall or near-fall occurred.

**Source**: UCI ML Repository, from Ustad et al. 2023 (cited in our proposal).

**Structure**: `data/har70plus/501.csv` … `518.csv` — one CSV per subject.

| Column | Meaning |
|---|---|
| `timestamp` | time of reading |
| `back_x/y/z` | lower-back accelerometer, 3 axes |
| `thigh_x/y/z` | thigh accelerometer, 3 axes |
| `label` | activity code (1–9), see below |

**EDA findings** (`eda/eda_har70.py`, plot in `outputs/har70_transition.png`):
- 2.26M rows total, sampled at **50 Hz**, zero missing values.
- Label distribution is heavily skewed: walking 47.8%, lying 21.4%, sitting 18.5%, cycling (sit) 9.0%, stairs up 2.9%, **standing only 0.22%**, stairs down 0.20%. "Shuffling" and "cycling (stand)" never appear in our subjects at all.
- There is **no fall or near-fall label in this dataset** — it only tells you the activity type.
- We visually confirmed a sit-to-stand transition produces a clearly distinguishable signal shape in the thigh sensor (flat while sitting, then a burst of movement) — see the plot. This is the raw material our proposal's "slower sit-to-stand transition" idea depends on, but because "standing" is such a rare label, we'll need to detect the *boundary* between sitting and standing segments rather than relying on the "standing" label directly.

**Implication for modeling**: since there's no fall/near-fall label, this dataset **can't train a supervised classifier for our target question**. Its role is to supply mobility-pattern features (time spent per activity, sit-to-stand transition duration) for the **k-means clustering** part of our proposal — looking for "hidden" decline patterns without needing a label.

**Important scope caveat**: HAR70+ (like our other two datasets) is a **single recording session per subject**, not repeated measurements over weeks or months. None of our three datasets are longitudinal. That means nothing here can literally demonstrate "decline over time" the way the project's core pitch describes — what we *can* demonstrate is that unsupervised clustering finds real, meaningful structure in movement data. Actual longitudinal decline-tracking would come later, from a real user's data accumulating after deployment — which is exactly what Gemini's Phase 4 "7-day calibration window" / "rolling baseline" idea already anticipates. Worth stating explicitly in the report so this isn't read as a gap we missed.

**Phase 1+2 status: done** (`src/har70_features.py`, `training/phase2_har70_kmeans.py`). Sliced each subject's continuous stream into 5-second windows (250 samples @ 50Hz), computed movement-intensity features (mean/std of back and thigh accelerometer magnitude) per window — 9,032 windows total across 18 subjects. Ran k-means (k=3, chosen for interpretability; the elbow curve was gradual rather than sharply bent, so this isn't a hard statistical pick) **without giving it the activity label**, then cross-tabulated clusters against the true label afterward purely as a sanity check. Result: the 3 unsupervised clusters line up cleanly with real intensity levels — a near-zero-movement cluster (sedentary mix: lying/sitting/cycling), a moderate-movement cluster (88% walking), and a high-movement cluster (99.9% walking, likely brisk walking) — see `outputs/har70_kmeans_clusters.png`. This confirms the clustering approach finds genuine structure in real elderly movement data, which is the validation your proposal's k-means step needed; it does not, and can't, show "decline" directly given the cross-sectional data (see caveat above).

---

## 3. SisFall — "A Fall and Movement Dataset"

**What it is**: real accelerometer + gyroscope recordings from 38 subjects performing scripted trials — either an "Activity of Daily Living" (ADL, e.g. walking, sitting down, picking something up) or a simulated fall (in a padded/controlled environment). Recorded via a waist-mounted device with three sensors.

**Source**: Sucerquia et al. 2017 (cited in our proposal); we pulled it via the Kaggle mirror `nvnikhil0001/sis-fall-original-dataset` since the original university host is offline.

**Structure**: `data/SisFall_dataset/<subject>/<code>_<subject>_R<trial>.txt` — one file per trial.

- Subject folders: `SA01`–`SA23` (young adults) and `SE01`–`SE15` (elderly) — **38 subjects total**.
- Filename code: `D##` = ADL (daily activity), `F##` = simulated fall.
- Each file has 9 unlabeled columns, raw ADC counts (not yet converted to physical units): 3-axis accel (ADXL345), 3-axis gyro (ITG3200), 3-axis accel (MMA8451Q) — sampled at **200 Hz**.

**EDA findings** (`eda/eda_sisfall.py`, plot in `outputs/sisfall_fall_vs_adl.png`):
- 4,505 trial files total. Trial length varies by activity (~12s for most ADLs, up to 100s for long-duration ones like "walking slowly," 15s for falls).
- **Age imbalance in fall trials specifically**: SA (young adult) subjects contributed 1,723 fall trials; SE (elderly) subjects contributed only **75**. Elderly subjects did contribute substantially to ADL data (898 trials), just not falls — for obvious safety/ethical reasons, real elderly people aren't asked to fall repeatedly for research.
- Visually confirmed the expected signature: a sharp acceleration spike at the moment of impact during a fall trial, vs. steady rhythmic oscillation during an ADL trial (see plot) — this is a real, learnable pattern.

**Implication for modeling**: this is our strongest real-subject fall-vs-non-fall dataset, but it's raw time-series (needs the windowing step described above) and the elderly-fall imbalance needs to be handled explicitly — evaluate model performance on the SE (elderly) subset separately, don't just report an overall accuracy that's dominated by young-adult examples.

**Phase 1 status: done** (`src/sisfall_features.py`). Each trial file is one scripted event (one ADL performance or one simulated fall), so we aggregate it into a single row of summary features (mean/std/min/max/range of accelerometer and gyroscope magnitude) rather than slicing into sub-windows — sub-windowing an F-coded trial would mislabel the walk-up/recovery portions as "fall," since the actual impact is only a fraction of the 15-second recording. Built 4,505 trial-level rows (`data/sisfall_features.csv`), fall/non-fall class balance is 1,798/2,707 (~40% fall, not badly imbalanced overall). Unlike cStick, correlations with the label are moderate and realistic (0.42–0.66 for the strongest features), with genuine overlap between classes — see `outputs/sisfall_features_boxplots.png`. `accel_max`/`accel_range`/`gyro_max` are the strongest predictors (ANOVA p ≈ 0); `n_samples` (trial duration) was deliberately excluded as a feature since fall trials are a near-fixed 15s while ADLs vary, which would have been a shortcut, not a real signal.

**Phase 2 status: done** (`training/phase2_sisfall.py`). Stratified 80/20 split, Logistic Regression vs. Decision Tree, both evaluated overall *and* on the SE (elderly) subset alone. Logistic Regression wins overall (93% accuracy, 92% Fall recall) — a realistic, non-suspicious number, unlike cStick. **On the SE-only subset, Fall recall drops to 58%** (vs. 90% for the SA/young-adult-heavy overall set) — the model misses well over a third of real elderly falls it would catch class-wide. Sample size caveat: only 19 elderly fall trials landed in the test split, so the exact number should be read as directional, not precise — but it's a large enough drop to be a real effect, and it's the concrete, measured version of the age-representation bias flagged in our proposal. Confusion matrices in `outputs/sisfall_confusion_matrices.png`; model saved to `models/sisfall_model.pkl`. **This is the real accuracy story for the project** — a working model with a known, quantified, honestly-reported weakness, which is far more credible than cStick's inflated 100%.

---

## Summary table

| Dataset | Real or simulated | Rows/trials | Rate | Has fall label? | Role in project |
|---|---|---|---|---|---|
| cStick | Likely simulated | 2,039 rows | n/a (snapshot) | Yes (0/1/2) | Primary tabular classifier (after dropping leaky columns) |
| HAR70+ | Real | 2.26M rows | 50 Hz | No (activity only) | Feature source for k-means clustering |
| SisFall | Real | 4,505 trials | 200 Hz | Yes (Fall/ADL) | Secondary classifier from windowed real-subject signal |

**Phase 2 status: done** (`training/phase2_cstick.py`). Stratified 80/20 split, Logistic Regression vs. Decision Tree. **Logistic Regression scored 100% precision/recall/F1 on all three classes**; Decision Tree scored 99.75% macro recall (1 misclassification out of 408 test rows). Confusion matrices in `outputs/cstick_confusion_matrices.png`. Winning model (Logistic Regression) saved to `models/cstick_model.pkl`.

**Why it's 100% — verified, not just suspected.** We checked the per-class min/max range of each feature and found three of the four have **zero overlap between classes**:

| Feature | Normal range | Near-Miss range | Fall range |
|---|---|---|---|
| `Distance` | 50–70 | 10–30 | 0–10 |
| `HRV` | 60–90 | 90–105 | 105–125 |
| `SpO2` | 90–100 | 80–90 | 60–80 |
| `Sugar level` | 70–80 | 30–70 | 10–179 (overlaps) |

A single threshold rule (e.g. "if `Distance` < 10 then Fall") would already get near-perfect accuracy on this data — the model isn't learning subtle patterns, it's finding boundaries a generation script already drew. `Sugar level` is the one feature with real overlap between classes, and it's also the weakest one in the Phase 1 ANOVA test, consistent with it being the most "realistic" column here.

**What this means for the report**: state the 100% result alongside this explanation, never as a standalone headline claim — it reflects synthetic non-overlapping thresholds, not genuine predictive difficulty. Keep the cStick model as a working proof-of-concept for the full pipeline (train → evaluate → export → deploy), but **treat SisFall's windowed, real-subject results as the actual accuracy number for the project**, since that data can't be trivially threshold-separated the way this can.

## What's next

1. ~~Clean `cStick.csv` (drop `Pressure`, `Accelerometer`)~~ — done.
2. ~~Train baseline Logistic Regression + Decision Tree on cStick~~ — done, see cStick Phase 2 status above.
3. ~~Build SisFall trial-level features and train fall-vs-ADL classifier, evaluated overall and on the SE subset~~ — done, see SisFall Phase 1/2 status above.
4. ~~Apply windowing to HAR70+ and run k-means clustering~~ — done, see HAR70+ Phase 1+2 status above.
5. Update the proposal's bias/mitigation section with the real SisFall SE-subset numbers (58% vs. 90% Fall recall) instead of the speculative version, and add the "no dataset here is longitudinal" caveat to the training/testing considerations section.
6. All three algorithms from the proposal are now demonstrated end-to-end (Logistic Regression + Decision Tree on cStick and SisFall, k-Means on HAR70+). Remaining work is Phase 3 (Streamlit deployment) and folding these findings back into the written proposal/report.
