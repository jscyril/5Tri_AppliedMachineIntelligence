# Lab 06 — Explaining Black-Box Models with LIME

A detailed walkthrough of `lab06.ipynb`: what every section does, why it was done that way, what the results mean, and which alternative methods could have been used at each step.

---

## Contents

1. [The assignment in one paragraph](#1-the-assignment-in-one-paragraph)
2. [Files and how to run](#2-files-and-how-to-run)
3. [Background: how LIME works](#3-background-how-lime-works)
4. [Question 1 — Tabular (Breast Cancer + Random Forest)](#4-question-1--tabular-breast-cancer--random-forest)
5. [Question 2 — Text (IMDb + TF-IDF + Logistic Regression)](#5-question-2--text-imdb--tf-idf--logistic-regression)
6. [Question 3 — Image (MobileNetV2)](#6-question-3--image-mobilenetv2)
7. [Results summary](#7-results-summary)
8. [Alternatives to LIME](#8-alternatives-to-lime)
9. [Limitations and how to improve](#9-limitations-and-how-to-improve)
10. [Glossary](#10-glossary)
11. [Likely viva questions](#11-likely-viva-questions)

---

## 1. The assignment in one paragraph

Train three different "black-box" models on three different kinds of data (a table of tumour measurements, movie-review text, and a photograph) and use **LIME** to explain individual predictions: *which features, words or image regions made the model say what it said?* Each question then checks the explanation by **changing the input**: removing the most important features, words or regions and seeing whether the prediction really changes. On top of the required tasks, the notebook adds extra experiments (marked ★) that test **how stable** and **how faithful** LIME's explanations are and compare them with other ways of explaining a model.

| Task from the sheet | Where in notebook |
|---|---|
| **Q1** Load Breast Cancer dataset | §1.1 |
| Pre-process, train/test split | §1.2 |
| Train Random Forest | §1.3 |
| Accuracy, precision, recall, confusion matrix (+ heatmap) | §1.4 |
| `LimeTabularExplainer` on one test instance | §1.5 |
| Feature-importance bar chart; positive/negative contribution plot | §1.5 (Plots 1, 2) |
| Five most influential features | §1.6 |
| ★ Stability, local vs global importance, deletion test | §1.7 – §1.9 |
| **Q2** Load IMDb | §2.1 |
| Clean text, handle missing values | §2.2 |
| TF-IDF; Logistic Regression (+ ★ Naive Bayes) | §2.3 |
| Accuracy + classification report | §2.5 |
| `LimeTextExplainer` on one review | §2.6 |
| Word-importance bar chart; highlighted text; probability comparison | §2.6 – §2.7 (Plots 1–3) |
| Modify/remove two influential words | §2.7 |
| ★ Global coefficients, leave-one-word-out check, explaining a mistake | §2.8 – §2.10 |
| **Q3** Load MobileNetV2 | §3.1 |
| Select and pre-process an image | §3.2 |
| Top-3 predicted classes | §3.4 |
| `LimeImageExplainer` + superpixel segmentation | §3.5 |
| Original / segmented / supporting / opposing plots | §3.7 |
| Mask important regions, observe probability | §3.8 |
| ★ Weight heatmap for top-3 classes, deletion curve, segmentation comparison | §3.7, §3.9, §3.10 |

---

## 2. Files and how to run

```
lab06/
├── lab06.ipynb            # the notebook (already executed, outputs saved)
├── lab06_explanation.md   # this file
└── data/
    └── labrador.jpg       # the Q3 image, downloaded by §3.2
```

**Environment.** Uses the repository's `.venv` (Python 3.13). The lab needed three packages that were not installed yet; they were added with

```bash
uv pip install --python .venv/bin/python lime scikit-learn seaborn
```

`uv pip` was used instead of `uv add` on purpose: `uv add` re-syncs the environment to `uv.lock`, which would remove TensorFlow (it was installed into the venv outside the lockfile for Lab 04).

**Run.** Open the notebook in Jupyter and *Run All*, or headless:

```bash
cd lab06
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab06.ipynb
```

The first run downloads the IMDb data (~17 MB), MobileNetV2 weights (~14 MB) and the dog image into the Keras cache / `data/`. Full run time on a laptop CPU is about 3–4 minutes; the slowest parts are training the two text models (~20 s each) and the four image explanations (~10–25 s each).

**Reproducibility.** Everything random takes `SEED = 42`: the train/test splits, the forest, every LIME explainer (`random_state=` / `random_seed=`) and the random baselines. Re-running gives the same numbers.

---

## 3. Background: how LIME works

### 3.1 The problem
Accurate models (random forests, deep networks, even a linear model with 100,000 features) are hard for humans to read. We want to know **why one specific prediction** was made, for example to check the model is right for the right reasons, to debug a mistake, or because a doctor or customer is owed a reason.

### 3.2 The idea
A complicated decision boundary looks **roughly linear if you zoom in far enough** on one point. LIME:

1. Takes the instance **x** to explain.
2. Converts it into an **interpretable representation**: binary "is this feature/word/superpixel present?" switches *z′*.
3. Creates many **perturbed samples** by randomly switching parts off.
4. Gets the black-box model's prediction **f(z)** for every sample.
5. Weights each sample by its **proximity** to x with an exponential kernel, π<sub>x</sub>(z) = exp(−D(x, z)² / σ²), so near samples matter more.
6. Fits a simple, interpretable model **g** (weighted ridge regression) to imitate f on those samples.
7. Reports g's coefficients as the explanation.

Formally (Ribeiro, Singh & Guestrin, *"Why Should I Trust You?"*, KDD 2016):

> ξ(x) = argmin<sub>g ∈ G</sub> 𝓛(f, g, π<sub>x</sub>) + Ω(g)

*𝓛* = how badly g imitates f near x (weighted squared error), *Ω(g)* = complexity penalty (here: keep only the top K features).

### 3.3 What the output means
- **Weight sign:** positive = this feature being present/true pushes the probability of the *explained class* **up**.
- **Weight size:** how much, in probability units, roughly.
- **`score` (R²):** how well the linear surrogate fits the black box locally. Low R² (< 0.3) means the explanation is a poor summary and you should be careful.
- **`intercept` / `local_pred`:** the surrogate's baseline and its prediction for x, which you can compare with the real model's probability.

### 3.4 How the three explainers differ

| | Tabular | Text | Image |
|---|---|---|---|
| Interpretable unit | a discretized rule, e.g. `worst area > 1033.5` | a distinct word | a superpixel |
| "Switching off" | resample the feature from the training distribution | delete the word | paint the superpixel a fixed colour (black here) |
| Distance for the kernel | Euclidean on scaled features | cosine distance of the word on/off vector | cosine distance of the superpixel on/off vector |

---

## 4. Question 1 — Tabular (Breast Cancer + Random Forest)

### §0 Imports and reproducibility
Standard scientific stack plus `seaborn` for the heatmap. Two colour constants, `POS` (green) and `NEG` (red), are used in **every** plot of the notebook so "green = supports, red = opposes" never changes meaning. TensorFlow C++ logs are silenced via `TF_CPP_MIN_LOG_LEVEL=3`.

### §1.1 Load the dataset
`sklearn.datasets.load_breast_cancer()`: 569 tumours × 30 features. The 30 features are 10 measurements of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as **mean**, **standard error** and **worst** (mean of the three largest values). Target: `0 = malignant` (212), `1 = benign` (357).

*Why a DataFrame:* named columns make the LIME output readable and let `describe()` show scale differences (area is in the hundreds, smoothness is ~0.1).

### §1.2 Pre-processing and split
| Step | Done? | Why |
|---|---|---|
| Missing values | checked → 0 | sklearn's copy is clean, but you verify rather than assume |
| Duplicates | checked → 0 | duplicates across train/test inflate test scores |
| Scaling | **not applied** | trees split on thresholds, so scaling changes nothing for a Random Forest. Raw units also keep the LIME rules readable (`worst area > 1033.5` instead of `worst area > 0.87`). LIME scales internally for its distance kernel anyway. |
| Encoding | not needed | all features numeric |
| Split | 80/20, **stratified**, seed 42 | stratification keeps the 37 % malignant share in both sets (train 0.374, test 0.368), which matters with only 114 test rows |

*Alternatives:* `StandardScaler` / `MinMaxScaler` (required for SVM, k-NN, logistic regression, neural nets; useless for trees); k-fold cross-validation instead of one split (we do use 5-fold CV on the training set as a check); removing highly correlated features (radius, perimeter and area are ~0.99 correlated), which would actually make LIME explanations cleaner (see §1.8).

### §1.3 Train a Random Forest
`RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`. 5-fold CV accuracy on train: **0.958 ± 0.022**.

*Why a Random Forest:* the sheet asks for it, and it is a good LIME demo: accurate on tabular data out of the box, robust to unscaled and correlated features, but **not directly interpretable** (300 trees × dozens of splits each). Defaults were kept because tuning is not the point of this lab and RFs are not very sensitive to hyper-parameters.

*Alternatives:* gradient boosting (XGBoost, LightGBM, `HistGradientBoostingClassifier`), usually slightly more accurate; logistic regression, which is interpretable by itself and a useful baseline; SVM with RBF kernel (needs scaling).

### §1.4 Evaluation
| Metric | Value |
|---|---|
| Accuracy | 0.9474 |
| Precision (malignant) | 0.9286 |
| Recall (malignant) | 0.9286 |
| F1 (malignant) | 0.9286 |
| ROC-AUC | 0.9937 |

Confusion matrix (test, 114 rows): 39 malignant correctly caught, **3 malignant missed** (predicted benign), 3 benign false alarms, 69 benign correct.

*Why malignant is the positive class:* by default sklearn treats label `1` (benign) as positive, which would report "recall of benign". Clinically the dangerous error is a **missed cancer** (false negative for malignant), so precision/recall are computed with `pos_label=0`. The full `classification_report` shows both classes anyway.

*Why ROC-AUC too:* accuracy, precision and recall all depend on the 0.5 threshold. AUC (0.994) shows the forest ranks almost every malignant case above every benign one; lowering the threshold would trade a few false alarms for fewer missed cancers, which is what a screening tool should do.

*Alternatives:* precision-recall curve / average precision, Matthews correlation coefficient (robust to class imbalance), calibration curve (are the probabilities trustworthy?), cost-sensitive threshold tuning.

### §1.5 LIME on one test instance
```python
LimeTabularExplainer(X_train.values, feature_names=..., class_names=...,
                     mode="classification", discretize_continuous=True, random_state=42)
```
- **Training data is passed in** because LIME needs the feature distributions to (a) choose quartile boundaries for discretization and (b) sample realistic perturbations.
- **`discretize_continuous=True`** (default, quartile discretizer): explanations become rules (`worst radius > 18.55`), which are easier to read than "a 1-unit increase in worst radius adds 0.004".
- **Wrapper `rf_predict`:** LIME passes NumPy arrays, but the forest was fitted on a DataFrame; wrapping restores column names (avoids sklearn's feature-name warning and guarantees column order).
- **Instance choice:** the first test row that is malignant *and* predicted malignant. A correct malignant call is the clinically interesting case, and explaining a correct prediction first shows what "normal" reasoning looks like.
- **`num_samples=5000`** (default is 5000): more samples → more stable explanation (§1.7).
- **`labels=(label0,)`:** explain the *predicted* class, so positive weights mean "towards malignant".

Result: RF says P(malignant) = 1.00. LIME's local model: intercept 0.239, local prediction 0.987, **R² = 0.654**, a decent local fit.

**Plot 1 (importance bar chart)** sorts the 10 rules by |weight|. **Plot 2 (contributions)** shows the signed weights; all 10 are green here, meaning every top rule pushes towards malignant. (For a borderline patient you would see a mix of green and red.) We drew Plot 2 ourselves instead of using `exp.as_pyplot_figure()` to keep the colours consistent with the rest of the notebook; the numbers are identical. `exp.show_in_notebook()` is another option, but its JavaScript output does not render on GitHub or in PDF exports.

### §1.6 Five most influential features

| # | Rule | Patient's value | Train median | Weight |
|---|---|---|---|---|
| 1 | worst area > 1033.5 | 1926 | 683 | +0.136 |
| 2 | worst concave points > 0.16 | 0.194 | 0.097 | +0.127 |
| 3 | worst radius > 18.55 | 25.05 | 14.91 | +0.100 |
| 4 | worst perimeter > 124.95 | 178.6 | 97.6 | +0.098 |
| 5 | mean concave points > 0.07 | 0.114 | 0.033 | +0.063 |

**Interpretation:** the tumour's largest nuclei are about **2.8× the median area** with deep concavities. Large, irregular nuclei are the textbook signs of malignancy, so the model is right *for medically sensible reasons*. That is the main thing LIME lets you check.

`as_map()` is used instead of parsing the rule strings, because it returns `(feature_index, weight)`, which makes it trivial to look up the patient's raw value.

### ★ §1.7 Stability of the explanation
LIME is random. We re-ran the explanation 10× (seeds 0–9) at 500, 1,000 and 5,000 samples and computed the **Jaccard similarity** |A∩B| / |A∪B| of the top-5 feature sets for every pair of runs.

| num_samples | mean Jaccard | min Jaccard | s / explanation |
|---|---|---|---|
| 500 | 0.748 | 0.667 | ~0.43 |
| 1,000 | 0.844 | 0.667 | ~0.46 |
| 5,000 | 0.822 | 0.667 | ~0.54 |

- 500 samples is noticeably less stable. From 1,000 onwards it levels off (the small dip at 5,000 is noise between 45 pairs).
- The minimum never rises above 0.667 (4 of 5 shared) because **radius, perimeter and area are nearly the same feature** (r ≈ 0.99), so they swap ranks between runs. This is a known LIME weakness with correlated features, not a bug.

*Why it matters:* a single LIME run can be misleading; for any real decision, report stability or average multiple runs. *Alternatives:* fix the seed (hides the problem rather than solving it); **BayLIME** / **S-LIME** (variants with confidence intervals or a sample-size test); **SHAP**, which is deterministic for trees (TreeSHAP).

### ★ §1.8 Local vs global importance
Three global importance measures compared side by side:

| Method | What it measures | Pros | Cons |
|---|---|---|---|
| **Impurity (MDI)** `rf.feature_importances_` | total Gini decrease from splits on a feature | free, instant | computed on training data; biased towards high-cardinality / continuous features |
| **Permutation** `sklearn.inspection.permutation_importance` | accuracy drop on **test** data when a column is shuffled | model-agnostic, measured on held-out data | underestimates correlated features: shuffle one and the model uses its twin |
| **Aggregated LIME** | mean \|weight\| of 40 local explanations | built from the same explanations users see | slow; inherits LIME's randomness |

Spearman rank correlations: impurity ↔ aggregated LIME **0.87**, permutation ↔ others **≈ −0.3**. The negative correlation is exactly the correlated-feature effect: shuffling *worst radius* alone barely hurts because *worst area* and *worst perimeter* still carry the same information, so permutation importance spreads "credit" thinly and noisily over 30 features.

*Alternatives:* **SP-LIME** (submodular pick: `lime.submodular_pick.SubmodularPick`) chooses a small set of diverse, representative explanations instead of averaging; **grouped permutation importance** (shuffle correlated features together); **SHAP summary plots**; partial dependence / ICE plots.

### ★ §1.9 Deletion (faithfulness) test
Replace the patient's top-k LIME features with the training **median**, re-predict, and compare against replacing k **random** features (30 random draws per k).

- After removing LIME's top-5: **P(malignant) = 0.54**
- After removing 5 random features: **P(malignant) = 0.90**

LIME's features really drive the prediction: removing them takes the model from certain to a coin flip, while random features barely matter. It does not fall to 0 because the forest still sees many other malignant-looking features (worst concavity, area error, worst texture, ...).

*Why the median as "removed":* there is no "absent" value for a continuous measurement; the median is a neutral, realistic value. *Alternatives:* the mean, a random training sample's value (closer to what LIME and SHAP do), or retraining without the feature (ROAR, *RemOve And Retrain*), which is more rigorous but costs a retrain per k.

---

## 5. Question 2 — Text (IMDb + TF-IDF + Logistic Regression)

### §2.1 Load IMDb
`keras.datasets.imdb.load_data()` gives 25k + 25k reviews, perfectly balanced. Keras stores them as **integer word indices**, so we decode them back to words with `imdb.get_word_index()`. Indices are offset by 3 because 0, 1, 2 are reserved for padding, start and unknown, and the first token of each sequence is `<start>`, which we skip.

*Why this source:* TensorFlow was already installed, the download is small (17 MB) and cached, and it is the standard IMDb dataset (Maas et al., 2011). *Trade-off:* Keras has already lower-cased the text and removed punctuation, so the decoded reviews read slightly oddly (`robert redford's is an amazing actor`) and still contain `br` tokens from HTML `<br />` tags. *Alternatives:* the raw `aclImdb_v1.tar.gz` (84 MB, original text with punctuation), Hugging Face `datasets.load_dataset("imdb")`, NLTK's `movie_reviews` (2,000 reviews), or the Rotten Tomatoes sentence-polarity dataset.

### §2.2 Cleaning and missing values
1. **Missing values:** empty or whitespace-only reviews are converted to `NaN` with a regex and dropped. There were 0, but the check has to exist because real scraped data almost always contains some.
2. **Duplicates:** **423** duplicate reviews removed (49,577 remain). If a review is in both train and test, the test score is inflated.
3. **`clean_text`:** lower-case → remove `<br />` and its `br` debris → keep only `a–z` and `'` (so `don't` survives as one token) → collapse whitespace. A one-line `assert` checks the function on a hand-made example.
4. **No stop-word removal.** sklearn's English stop-word list contains `not`, `no`, `nor`, `never`, which carry sentiment ("not good"). With TF-IDF, very common words already get low weights automatically.
5. **No stemming/lemmatization.** With 40k training reviews there is enough data to learn `loved` and `love` separately, and LIME explanations are easier to read with real words.

*Alternatives:* NLTK/spaCy tokenization, lemmatization (`spaCy`), negation handling (prefix the words after "not" with `NOT_`), emoji/emoticon mapping, spell correction.

### §2.3 TF-IDF and the classifiers
```python
TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=100_000)
```
| Setting | Why |
|---|---|
| `ngram_range=(1,2)` | bigrams capture short phrases: `not good`, `waste of`, `must see` |
| `min_df=3` | drop words appearing in < 3 reviews (typos, names): less noise, smaller vocab |
| `max_df=0.9` | drop words in > 90 % of reviews (carry no signal) |
| `sublinear_tf=True` | use 1 + log(tf): a word said 10× is not 10× as important |
| `max_features=100_000` | caps memory; bigrams would otherwise give millions of columns |

TF-IDF = *term frequency* × *inverse document frequency*: a word scores high if it is frequent **in this review** but rare **across reviews**.

**Pipeline:** vectorizer and classifier live in one `make_pipeline(...)`. This matters for two reasons. (1) The vectorizer is fitted **only on training text** (no vocabulary leakage from test). (2) LIME's text explainer needs a function that takes **raw strings** → probabilities, and `pipe.predict_proba` is exactly that.

**Models:**
- **Logistic Regression** (`C=4`, `max_iter=2000`): the standard strong baseline for TF-IDF text. `C=4` (weaker regularization than the default 1) suits a large, sparse feature space with plenty of data.
- ★ **Multinomial Naive Bayes** (`alpha=0.3`): the classic fast text model; assumes words are independent given the class.

*Alternatives:* `CountVectorizer` (raw counts, best paired with NB), character n-grams (robust to typos), `HashingVectorizer` (no vocabulary in memory), word embeddings (averaged GloVe/word2vec), or a fine-tuned transformer (DistilBERT ~93 % on IMDb). Classifier alternatives: linear SVM (`LinearSVC`, similar accuracy, but needs `CalibratedClassifierCV` to output probabilities for LIME), SGDClassifier, NB-SVM (Wang & Manning 2012).

### §2.5 Evaluation
| Model | Accuracy | Neg F1 | Pos F1 |
|---|---|---|---|
| **Logistic Regression** | **0.9136** | 0.912 | 0.915 |
| Multinomial NB | 0.8844 | 0.884 | 0.885 |

Balanced classes, so accuracy is a fair headline metric; the report confirms neither class is neglected. LR wins by ~3 points because it can learn how much to trust each feature, while NB's independence assumption double-counts correlated n-grams (e.g. `waste` and `waste of`).

### §2.6 LIME on one review
**Review choice:** a correctly classified test review with **60–160 words** and confidence **0.75–0.92**. Short enough to read and highlight; *not* 0.99-confident, so removing two words can visibly move the probability. The chosen review is a positive one about an indie-film documentary, P(positive) = **0.863**.

```python
LimeTextExplainer(class_names=["negative", "positive"], random_state=42)
explain_instance(review, pipe.predict_proba, labels=(1,), num_features=12, num_samples=3000)
```
`labels=(1,)` explains **P(positive)**, so green = towards positive, red = towards negative. 3,000 samples is a balance between stability and run time (the default is 5,000). Local fit **R² = 0.58**.

**Plot 1 (bar chart)** — supporting words include `it`, `especially`, `different`, `fun`, `appreciate`, `hoot`, `pleasure`; opposing words include `be`, `painful`, `would`, `to`.

**Plot 2 (highlighted text)** — a small custom HTML renderer shades each explained word green/red with opacity ∝ |weight|, and hovering shows the exact weight. Written by hand (~15 lines) instead of `exp.show_in_notebook()` so it renders in exported HTML and keeps the notebook's colour scheme.

**An honest observation:** the two top words are `it` and `especially`, not obviously "sentiment words". That is what the model actually does: across 40k reviews, `it` (as in "loved it", "see it") and `especially` (as in "especially the acting") correlate with positive reviews. LIME exposes this kind of dataset artefact; that is a feature, not a failure. Note also that `painful` is correctly red even though the reviewer uses it positively ("painful but rewarding"). A bag-of-words model cannot see that.

### §2.7 Modifying the two most influential words
The two words that most support the predicted class (`it`, `especially`) were
- **(a) removed** everywhere (regex with word boundaries `\b...\b`, so `it` does not delete the "it" inside "with"), and
- **(b) replaced** by the model's single most negative unigram, read from the LR coefficients: **`worst`**.

| Version | P(negative) | P(positive) | Prediction |
|---|---|---|---|
| original | 0.137 | **0.863** | positive |
| removed `it`, `especially` | 0.267 | **0.733** | positive |
| replaced with `worst` | **0.591** | 0.409 | **negative** |

Removing just two words costs 13 points of confidence; swapping them for one strongly negative word **flips** the prediction. This confirms LIME found words the model really depends on, and shows how fragile bag-of-words models are: one swapped word changes the outcome.

*Why "the strongest opposite word" instead of hand-picked antonyms:* it is automatic and reproducible for any review, and it uses the model's own idea of "opposite". *Alternatives:* WordNet antonyms, masked-language-model substitution (BERT fill-mask), or adversarial attack libraries such as TextAttack.

**Plot 3 (probability comparison)** — left: horizontal bars for the three versions with a 0.5 decision line; right: ★ the original review under LR vs NB (both positive; NB less confident at ~0.78).

### ★ §2.8 Global view
For a linear model, the coefficients **are** a global explanation. Top negative n-grams (`worst`, `bad`, `awful`, `waste`, `boring`, ...) and positive ones (`great`, `excellent`, `perfect`, `best`, ...) look sensible. LIME adds the **local** part: which of these words are in *this* review and how much they matter here, given TF-IDF weighting.

### ★ §2.9 Checking LIME against leave-one-word-out
For each of LIME's 12 words, delete only that word and record the exact change in P(positive). This is a cheap, exact "ground truth" for single-word effects.

**Spearman correlation between LIME weights and leave-one-out effects: 0.85.** LIME ranks the words almost the same way the direct test does. It is not 1.0 because LIME fits all words *jointly* (it captures interactions) while leave-one-out moves one word at a time; for a near-linear model like LR on TF-IDF the two should, and do, mostly agree.

### ★ §2.10 Explaining a mistake
The most confidently wrong short review (true: **negative**, predicted P(positive) = **0.97**):

> "good films cannot solely be based on a **beautiful** garden and a hill top … surprised to see it has won two awards extremely **overrated** … i first saw that kind of films from china visually **stunning** but also with really something captivating to say … this is **not** one of them"

LIME shows why: `beautiful`, `stunning`, `captivating` are strong positive words, and the reviewer uses them to describe *other* films or what the film *lacks*. `overrated` and `not` pull towards negative but are outvoted. A bag-of-words model cannot represent "X is not one of them". This motivates sequence models or transformers that read word order and scope.

---

## 6. Question 3 — Image (MobileNetV2)

### §3.1 Model
`MobileNetV2(weights="imagenet")`: 3.5M parameters, 224×224×3 input, 1000-class softmax output, trained on ImageNet.

*Why MobileNetV2:* suggested by the sheet, and practical: LIME calls the model **once per perturbed sample** (1,000+ times), so a lightweight CNN keeps the experiment around 20 s on a CPU. *Alternatives:* ResNet50 (25M params, more accurate, ~5× slower), EfficientNet-B0, InceptionV3 (299×299 input, the model used in the original LIME paper), Vision Transformers (ViT).

### §3.2 Image and pre-processing
A photo of a yellow Labrador from TensorFlow's public tutorial images, downloaded with `keras.utils.get_file` into `lab06/data/` (cached after the first run).

1. `load_img(..., target_size=(224, 224))`: resize to the network's input size.
2. Divide by 255 → image in **[0, 1]**. LIME's segmentation, `mark_boundaries` and matplotlib all expect this range.
3. The **`classify` function** (what LIME calls) multiplies back to 0–255 and applies `mobilenet_v2.preprocess_input`, which maps pixels to **[−1, 1]**, the range MobileNetV2 was trained with. Putting the model-specific step *inside* the classifier function means LIME only ever deals with a plain image.

*Why it matters:* the wrong pre-processing (e.g. feeding [0, 1] directly) still produces predictions, just bad ones, with no error message. *Alternative:* center-crop instead of resize to preserve aspect ratio.

### §3.4 Top-3 predictions
| Rank | Class | Probability |
|---|---|---|
| 1 | Labrador_retriever | **0.469** |
| 2 | Eskimo_dog | 0.054 |
| 3 | Ibizan_hound | 0.024 |

Correct top-1. 0.47 is not "uncertain": the remaining mass is spread over hundreds of similar dog breeds (golden retriever, etc.), which is normal for a 1000-class softmax.

### §3.5 LimeImageExplainer and segmentation
**Superpixels** are needed because 224×224 = 50,176 pixels would be far too many features, and a single pixel means nothing to a human. **Quickshift** (`kernel_size=4, max_dist=200, ratio=0.2`, LIME's default and the setting used in the LIME paper) produced **33 superpixels** that follow the dog's outline.

```python
explain_instance(img, classify, labels=top3_ids[::-1], hide_color=0,
                 num_samples=1000, batch_size=50, segmentation_fn=quickshift, random_seed=42)
```
- **`labels=top3_ids`:** explain all three top classes in one run (the perturbed images and predictions are shared, so classes 2 and 3 are almost free).
- **Reversed order:** LIME stores only the R² of the *last* label it fits, so the predicted class is fitted last to get *its* R².
- **`hide_color=0`:** "switched-off" superpixels are painted black. *Alternative:* `hide_color=None` (LIME then uses each superpixel's mean colour, which is gentler and less out-of-distribution), blurring, or inpainting.
- **`num_samples=1000`:** with 33 superpixels, 1,000 samples (~30 per feature) gives a stable fit; took ~23 s.
- **`batch_size=50`:** images go through the network 50 at a time (fast without running out of memory).

Local R² for Labrador: **0.46**, acceptable for images (pixels interact more non-linearly than tabular features).

### §3.7 Plots
1. **Original image.**
2. **Superpixel boundaries** (`skimage.segmentation.mark_boundaries`, yellow).
3. **Supporting regions:** the top-5 positive superpixels kept bright, the rest dimmed: the dog's **head, neck/chest and front body**. The model looks at the dog, not the lawn.
4. **Opposing regions** (`negative_only=True`): 10 superpixels with negative weight, shown in red: mostly **grass** and parts of the legs. Their presence *lowers* P(Labrador) slightly, probably because "dog on grass" also matches other outdoor breeds.

★ **Weight heatmap** colours *every* superpixel by its LIME weight for each of the top-3 classes. Interesting contrast: the **head** is strongly green for *Labrador* but **red** for *Eskimo_dog* and *Ibizan_hound*. The face is what tells the model "this is a Labrador and not those other breeds", while the chest/body region supports all three ("it is a light-coloured dog").

### §3.8 Masking important regions
| Input | P(Labrador) | New top-1 |
|---|---|---|
| original | 0.469 | Labrador_retriever |
| mask top-3 supporting | **0.010** | Eskimo_dog |
| mask top-5 supporting | **0.004** | borzoi |
| mask 5 random superpixels | 0.041 | muzzle |
| mask opposing superpixels | **0.906** | Labrador_retriever |

- Blacking out the 3 most important regions destroys the prediction (47 % → 1 %).
- Blacking out the *opposing* regions **doubles** confidence to 91 %, confirming those regions really did argue against the class.
- Five random patches also hurt (→ 4 %). Big black blobs are unusual for a CNN, and with only 33 superpixels a random 5 often clip the dog. That is why the next section compares *curves* instead of single points.

### ★ §3.9 Deletion curve
Superpixels are blacked out one at a time in LIME's order (by |weight|) vs a random order (averaged over 5 orders), tracking P(Labrador). Area under the curve (lower = more faithful):

- **LIME order: AUC = 0.028**: probability collapses after **2** superpixels.
- **Random order: AUC = 0.104**: needs ~11 superpixels to reach the same level.

LIME's ranking is about **4× more efficient** at removing the evidence, strong quantitative support that the explanation is faithful. (This is the *deletion metric* of Petsiuk et al., RISE, 2018; its mirror image, the *insertion metric*, starts from a blank image and adds superpixels.)

### ★ §3.10 Segmentation comparison
| Segmenter | How it works | Superpixels | Local R² | Top-5 supporting regions |
|---|---|---|---|---|
| **Quickshift** | mode-seeking in (colour, x, y) space | 33 | 0.479 | head + neck + front body: one clean region |
| **SLIC** (n=60) | k-means in (Lab colour, x, y) | 55 | 0.420 | face, front leg, **one patch of grass** |
| **Felzenszwalb** | graph-based region merging | 40 | 0.382 | face, chest strip, back |

(500 samples each, so R² differs slightly from the 1,000-sample run.)

All three agree the **face and chest** matter most, but the details differ, and SLIC even includes a grass patch. **The explanation depends on how you cut up the image.** Quickshift's segments follow object edges and give the cleanest, best-fitting explanation here. SLIC's compact, near-uniform tiles cut across the dog, so evidence is split over more, smaller pieces. More superpixels = finer detail but more features for the same number of samples, so noisier weights.

*Practical rule:* try more than one segmentation and trust what they agree on.

---

## 7. Results summary

| | Q1 Tabular | Q2 Text | Q3 Image |
|---|---|---|---|
| Model performance | RF: acc 0.947, malignant recall 0.929, AUC 0.994 | LR: acc **0.914** (NB 0.884) | top-1 Labrador, p = 0.47 |
| What LIME says | "worst" size & concavity features → malignant | `it`, `especially`, `fun`, `different` → positive | head + chest → Labrador; grass → against |
| Local surrogate R² | 0.65 | 0.58 | 0.46 |
| Perturbation test | top-5 removed: P 1.00 → 0.54 (random: 0.90) | 2 words removed: 0.86 → 0.73; replaced: **flips** to negative | top-3 masked: 0.47 → 0.01; opposing masked: → 0.91 |
| ★ Faithfulness | deletion beats random | Spearman 0.85 vs leave-one-out | deletion AUC 0.028 vs 0.104 random |
| ★ Caveat found | correlated features swap ranks between runs | model relies on non-sentiment words; misled by praise of *other* films | explanation changes with segmentation algorithm |

---

## 8. Alternatives to LIME

| Method | Data | Idea | vs LIME |
|---|---|---|---|
| **SHAP** (KernelSHAP, TreeSHAP, DeepSHAP) | any | Shapley values from game theory: each feature's average marginal contribution over all coalitions | theoretically grounded (additivity, consistency); TreeSHAP is **exact and fast** for forests; KernelSHAP is LIME with a specific kernel. Usually the first alternative to try. |
| **Anchors** (Ribeiro 2018) | any | finds an IF-THEN rule that "anchors" the prediction with high precision | gives *sufficient conditions* ("if worst area > 1033 and concave points > 0.16, it's malignant 98 % of the time") instead of weights |
| **Counterfactuals** (DiCE, Wachter et al.) | tabular/text | the smallest change to the input that flips the prediction | answers "what would need to change?", actionable for users |
| **Permutation importance / PDP / ICE** | tabular | global importance; average/individual effect curves | global, not local (ICE is per-instance) |
| **Integrated Gradients, SmoothGrad, saliency maps** | image/text (differentiable models) | gradient of the output w.r.t. the input pixels | needs model internals (white-box), but pixel-precise and much faster (no sampling) |
| **Grad-CAM** | CNN images | gradient-weighted activations of the last conv layer | one forward + backward pass; coarse heatmap; CNN-specific |
| **Occlusion sensitivity** | image | slide a grey patch over the image and record the probability drop | simple, model-agnostic, no surrogate model; cost = #patches |
| **RISE** | image | random masks weighted by output score | model-agnostic like LIME, no segmentation needed |
| **Attention weights / attention rollout** | transformers | read the model's attention | cheap but debated as an explanation |
| **Glass-box models** (logistic regression, EBM / `interpret`, shallow trees) | tabular | use a model that is interpretable by design | no post-hoc explanation needed; may cost some accuracy |

---

## 9. Limitations and how to improve

1. **Instability.** Different seeds → different explanations (§1.7). *Fix:* more samples, average several runs, report variance, or use deterministic methods (TreeSHAP).
2. **Unrealistic perturbations.** Tabular LIME samples features independently, producing impossible tumours (huge radius, tiny area). Black superpixels are not natural images. *Fix:* conditional sampling, `hide_color=None`/blur, or methods that respect the data manifold.
3. **Kernel width is arbitrary.** Defaults are σ = 0.75·√(#features) for tabular, 25 for text and 0.25 for images; a different width can change the explanation. *Fix:* test a few widths, or use SHAP, whose kernel is derived rather than chosen.
4. **Correlated features share credit unpredictably** (Q1 radius/perimeter/area). *Fix:* drop or group correlated features before explaining.
5. **Local ≠ global.** One explanation says nothing about the model overall. *Fix:* SP-LIME, aggregated LIME (§1.8), or global methods.
6. **Bag-of-words limits in Q2.** The model ignores word order, so LIME can only explain words, never phrases or negation scope. *Fix:* a transformer classifier explained with LIME/SHAP over tokens, or Integrated Gradients.
7. **Q2 data is pre-processed by Keras** (no punctuation, lower-cased). *Fix:* use the raw `aclImdb` archive or Hugging Face `imdb`.
8. **Only one instance per question.** Conclusions from one example are anecdotes. *Fix:* run the deletion tests over many test instances and report average AUC.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **LIME** | Local Interpretable Model-agnostic Explanations |
| **Model-agnostic** | works with any model; needs only `predict_proba`, not the model's internals |
| **Local explanation** | explains one prediction, not the whole model |
| **Surrogate model** | the simple model (weighted ridge regression) LIME fits to imitate the black box near one point |
| **Interpretable representation** | binary on/off features (rule true?, word present?, superpixel visible?) |
| **Kernel / proximity weight** | how much each perturbed sample counts, based on its distance from the original |
| **Discretization** | turning a continuous feature into bins (quartiles) so it reads as a rule |
| **Superpixel** | a group of neighbouring pixels with similar colour, used as one image feature |
| **Quickshift / SLIC / Felzenszwalb** | three superpixel segmentation algorithms in scikit-image |
| **TF-IDF** | term frequency × inverse document frequency; a word weighting for text |
| **Faithfulness** | does the explanation reflect what the model *really* uses? |
| **Deletion curve / AUC** | probability vs number of features removed in explanation order; lower AUC = more faithful |
| **Jaccard similarity** | \|A∩B\| / \|A∪B\|; overlap between two sets |
| **Permutation importance** | drop in score when a feature's values are shuffled |
| **Stratified split** | a split that keeps class proportions equal in train and test |
| **SP-LIME** | submodular pick: chooses a few diverse explanations to summarise a model |
| **SHAP** | SHapley Additive exPlanations, a game-theory-based alternative |

---

## 11. Likely viva questions

**Q: What does "local" mean in LIME?**
The explanation is only valid in a small neighbourhood around the one instance being explained. The model may behave completely differently for another patient or review.

**Q: Why is LIME model-agnostic?**
It only calls `predict_proba` on perturbed inputs. It never looks at trees, weights or gradients, so the same code explains a forest, a pipeline or a CNN.

**Q: What does a positive LIME weight mean?**
That feature's presence (or that rule being true) increases the probability of the explained class, by about the weight's value in the local linear model.

**Q: Why discretize continuous features?**
Rules like `worst area > 1033.5` are easier for humans than slopes, and the quartile bins come from the training data, so they mean "top 25 % of tumours".

**Q: Why didn't you scale the Breast Cancer features?**
Random Forests are invariant to monotone transformations of features, so scaling would change nothing, and raw units keep the explanation readable. LIME scales internally for its distance kernel.

**Q: Why did you report precision/recall for malignant?**
sklearn treats label 1 (benign) as positive by default. The costly error is missing a cancer, so malignant is the clinically relevant positive class.

**Q: Why are LIME explanations different each run?**
The perturbed samples are random. With fewer samples or correlated features, the surrogate's coefficients vary. We measured this with Jaccard overlap (§1.7).

**Q: Why didn't you remove stop words for IMDb?**
The standard list includes `not`/`no`, which flip sentiment, and TF-IDF already down-weights very common words.

**Q: Why use a Pipeline for the text model?**
So the vectorizer is fitted only on training data, and so LIME can pass raw strings directly to `pipe.predict_proba`.

**Q: Why superpixels instead of pixels?**
50k individual pixels are too many features to sample and too small to be meaningful. Superpixels are human-sized parts of the object.

**Q: What does `hide_color=0` do, and is it a good choice?**
Hidden superpixels are painted black. It is simple but creates unnatural images; the mean colour (`hide_color=None`) or blurring is gentler.

**Q: How do you know the explanation is correct?**
You cannot prove it, but you can test it: removing the top-ranked features/words/regions should change the prediction much more than removing random ones. All three deletion tests in the notebook pass that check.

**Q: How is SHAP different from LIME?**
SHAP uses Shapley values, which have guarantees (contributions add up to the prediction minus the baseline; consistent across models). KernelSHAP is essentially LIME with a specific kernel and no regularisation, and TreeSHAP computes exact values for tree models without sampling.

**Q: Why did masking random superpixels also drop the probability a lot?**
Large black patches are out-of-distribution for a CNN, and with only 33 superpixels a random 5 often overlap the dog. That is why we compare full deletion *curves* (LIME AUC 0.028 vs random 0.104) rather than a single point.
