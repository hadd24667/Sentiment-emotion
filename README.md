# Sentiment & Emotion Classification (TweetEval + GoEmotions)

End-to-end NLP project on **noisy social media text** (tweets), covering:

* **EDA with visualizations**
* **Classical ML baselines** (TF‑IDF)
* **Transformer fine-tuning** (BERTweet / RoBERTa)
* **Correct evaluation** for multi-label emotion classification, including **threshold tuning**

> Implementation is provided as a single notebook: `Sentiment&Emotion.ipynb`.

---

## Contents

* [Tasks & Datasets](#tasks--datasets)
* [Workflow](#workflow)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Models](#models)
* [Results](#results)
* [How to Run](#how-to-run)
* [Notes on Metrics & Thresholds](#notes-on-metrics--thresholds)
* [Future Work](#future-work)

---

## Tasks & Datasets

### 1) TweetEval – Sentiment (single-label)

* **Labels**: 3 classes (negative / neutral / positive)
* **Goal**: tweet sentiment classification
* **Key challenges**: informal language, short context, mentions/hashtags, domain noise

### 2) GoEmotions – Emotion (multi-label)

* **Labels**: 28 emotion classes
* **Goal**: multi-label emotion recognition (one text can have multiple emotions)
* **Key challenges**: **severe label imbalance**, overlapping emotions, low human agreement

---

## Workflow

1. **Load & inspect datasets**
2. **EDA** (missing values, duplicates, length stats, label distribution, noise indicators)
3. **Text cleaning / normalization**
4. **Baselines**

   * TweetEval: TF‑IDF + Logistic Regression
   * GoEmotions: TF‑IDF + One-vs-Rest Logistic Regression
5. **Transformers**

   * TweetEval: BERTweet fine-tuning
   * GoEmotions: RoBERTa fine-tuning (multi-label)
6. **Threshold tuning (GoEmotions)** on **validation**
7. **Final test evaluation** and reporting

---

## Exploratory Data Analysis (EDA)

All EDA is produced in the notebook (plots generated via matplotlib).

### GoEmotions (Train)

**Data quality**

* Missing values: **0%**
* Duplicate texts: **183 / 43,410 (0.42%)**

**Text length** (train)

* Mean length: **68.4 chars**, **12.84 words**
* 95th percentile: **131 chars**

**Multi-label density**

* Mean labels/sample: **1.18**
* Max labels/sample: **5**

**Label frequency**

* Long-tail distribution (top labels dominate)

**Emoji signals (top)**

* 😂, ❤️, 🤣, 😭, 👏, …

> Suggested screenshots (if you want to add repo assets):
>
> * `assets/eda_goemotions_length.png`
> * `assets/eda_goemotions_label_freq.png`

---

### TweetEval (Train)

**Data quality**

* Missing values: **0%**
* Duplicate texts: **29 / 45,615 (0.06%)**

**Text length**

* Mean length: **106.9 chars**, **19.24 words**

**Label distribution**

* neutral (1): **20,673**
* positive (2): **17,849**
* negative (0): **7,093**

**Noise indicators**

* % has mention: **29.44%**
* % has hashtag: **18.69%**
* % has URL: **0.20%**
* % has emoji: **0%** (in this split / parsing)

> Suggested screenshots:
>
> * `assets/eda_tweeteval_length.png`
> * `assets/eda_tweeteval_label_dist.png`

---

## Models

### Baselines

#### (A) TweetEval — TF‑IDF + Logistic Regression

* TF‑IDF vectorization on cleaned text
* Multinomial Logistic Regression classifier

#### (B) GoEmotions — TF‑IDF + One‑vs‑Rest Logistic Regression

* TF‑IDF vectorization on cleaned text
* OneVsRestClassifier(LogisticRegression)
* Predictions require a **decision threshold** to convert probabilities → labels

---

### Transformers

#### (C) TweetEval — BERTweet Fine-tuning

* Fine-tune BERTweet for 3-class classification
* Save best checkpoint by validation Macro‑F1

#### (D) GoEmotions — RoBERTa Fine-tuning (Multi-label)

* Fine-tune RoBERTa with sigmoid outputs (BCE-style multi-label)
* Evaluate with Micro‑F1 / Macro‑F1
* **Threshold tuned on validation** for best Macro‑F1

---

## Results

### TweetEval — Sentiment (3-class)

| Model                        | VAL Macro‑F1 |    VAL Acc | TEST Macro‑F1 |   TEST Acc |
| ---------------------------- | -----------: | ---------: | ------------: | ---------: |
| TF‑IDF + Logistic Regression |       0.6366 |     0.6630 |        0.5837 |     0.5966 |
| **BERTweet (fine-tuned)**    |   **0.7442** | **0.7560** |    **0.7189** | **0.7173** |

---

### GoEmotions — Emotion (28-class, multi-label)

**Baseline (TF‑IDF + OvR LR)**

| Setting                          | Threshold | VAL Micro‑F1 | VAL Macro‑F1 | TEST Micro‑F1 | TEST Macro‑F1 |
| -------------------------------- | --------: | -----------: | -----------: | ------------: | ------------: |
| Default                          |      0.50 |       0.5067 |       0.4534 |        0.4966 |        0.4322 |
| **Tuned on VAL (best Macro‑F1)** |  **0.55** |       0.5122 |   **0.4611** |        0.5036 |    **0.4387** |

**Transformer (RoBERTa fine-tuned)**

| Setting                          | Threshold | VAL Micro‑F1 | VAL Macro‑F1 | TEST Micro‑F1 | TEST Macro‑F1 |
| -------------------------------- | --------: | -----------: | -----------: | ------------: | ------------: |
| Default (checkpoint selection)   |      0.50 |       0.5652 |       0.3744 |             — |             — |
| **Tuned on VAL (best Macro‑F1)** |  **0.20** |       0.5974 |   **0.4464** |    **0.5924** |  **0.4395** ✅ |

> Note: In multi-label emotion classification, **Macro‑F1** is intentionally difficult and reflects minority-label performance. Accuracy is not used as the primary metric.

---

## How to Run

### Option 1 — Run the notebook (recommended)

1. Open: `Sentiment&Emotion.ipynb`
2. Run cells in order:

   * Load data → EDA → Clean text
   * Baseline models
   * Transformer fine-tuning
   * Threshold tuning (GoEmotions)
   * Final test evaluation

### Option 2 — Environment

Typical dependencies:

* `torch`, `transformers`, `datasets`
* `scikit-learn`, `numpy`, `pandas`
* `matplotlib`, `tqdm`

> GPU (e.g., T4) strongly recommended for Transformer fine-tuning.

---

## Notes on Metrics & Thresholds

### Why Macro‑F1 matters (especially for GoEmotions)

* **Micro‑F1** is dominated by frequent labels.
* **Macro‑F1** treats each label equally → exposes poor performance on rare emotions.

### Why threshold tuning is required for multi-label

* Model outputs probabilities per label.
* A fixed `threshold=0.50` often hurts recall for rare labels.
* Threshold is tuned on **validation** to avoid test leakage.

---

## Future Work

* Per-label threshold optimization (different threshold per emotion)
* Class-weighted / focal loss to address imbalance
* More epochs + early stopping
* Error analysis: bottom labels, confusion patterns, qualitative examples
* Multi-seed runs / lightweight ensemble

---

## Author

Built as a research-oriented NLP portfolio project focused on:

* real-world noisy data,
* correct evaluation practices,
* and strong baseline-to-transformer comparisons.
