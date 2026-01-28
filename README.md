# GoEmotions & TweetEval – Emotion and Sentiment Classification

This repository contains an end-to-end NLP project for **sentiment analysis** and **emotion recognition** on noisy social media text (Twitter-style data). The project is designed with a **data-centric and research-oriented workflow**, covering baselines, Transformer models, and rigorous evaluation strategies.

---

## 🎯 Project Objectives

* Build robust NLP models that work well on **noisy, short, real-world text** (tweets).
* Compare **classical ML baselines** with **Transformer-based models**.
* Handle **multi-label emotion classification** with severe class imbalance.
* Apply **proper evaluation metrics** and **threshold optimization** instead of relying on misleading accuracy scores.

---

## 📊 Tasks & Datasets

### 1. TweetEval – Sentiment Classification

* **Task**: Sentiment analysis (Positive / Neutral / Negative)
* **Type**: Single-label, multi-class
* **Dataset**: TweetEval benchmark
* **Challenges**:

  * Informal language
  * Slang, emojis, sarcasm

### 2. GoEmotions – Emotion Classification

* **Task**: Emotion recognition
* **Type**: Multi-label (28 emotion classes)
* **Dataset**: Google GoEmotions
* **Challenges**:

  * One text can express multiple emotions
  * Extreme label imbalance
  * Low inter-annotator agreement (even for humans)

---

## 🧠 Modeling Approach

### Baseline Models

* **TF-IDF + Logistic Regression** (TweetEval)
* **TF-IDF + One-vs-Rest Logistic Regression** (GoEmotions)

Purpose:

* Establish strong, interpretable baselines
* Provide fair comparison with Transformer models

---

### Transformer Models

* **RoBERTa** (GoEmotions – multi-label)
* **BERTweet / RoBERTa** (TweetEval – sentiment)

Key techniques:

* Fine-tuning with binary cross-entropy loss (multi-label)
* GPU training (NVIDIA T4)
* Validation-based model selection

---

## 📐 Evaluation Strategy

### Metrics (Chosen by Task Characteristics)

#### TweetEval (Single-label)

* **Primary**: Macro-F1
* **Secondary**: Accuracy, Confusion Matrix

#### GoEmotions (Multi-label)

* **Primary**: Macro-F1
* **Secondary**: Micro-F1
* **Optional**: Per-label F1, Hamming Loss

> Accuracy is intentionally *not* used for GoEmotions, as it is misleading for multi-label problems.

---

## 🎚️ Threshold Optimization (Key Contribution)

For GoEmotions, raw model probabilities are converted to labels using a **decision threshold**.

* Default threshold (0.50) leads to poor recall on rare emotions
* Threshold is tuned **on the validation set** to maximize **Macro-F1**

### Result (RoBERTa – GoEmotions)

```
Threshold: 0.20
Micro-F1 : 0.5924
Macro-F1 : 0.4395
```

This significantly improves performance on minority emotion classes and reflects real-world best practices for multi-label NLP tasks.

---

## 📈 Results Summary

| Task           | Model                      | Macro-F1   |
| -------------- | -------------------------- | ---------- |
| TweetEval      | TF-IDF + LR                | **~0.70**      |
| TweetEval      | Transformer                | **~0.72**      |
| GoEmotions     | TF-IDF + OvR               | **~0.30**      |
| GoEmotions     | RoBERTa (th=0.5)           | **~0.37**      |
| **GoEmotions** | **RoBERTa (tuned th=0.2)** | **~0.44**  |

---

## 🗂️ Project Workflow

1. Data loading & inspection
2. Exploratory Data Analysis (EDA)
3. Text cleaning & normalization
4. Baseline modeling (TF-IDF)
5. Transformer fine-tuning
6. Validation-based threshold tuning
7. Final evaluation on test set
8. Error analysis & per-label inspection

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* NumPy / Pandas
* Matplotlib / Seaborn

---

## 🚀 Key Takeaways

* Multi-label emotion classification requires **different metrics and evaluation mindset** than sentiment analysis.
* Threshold tuning is **critical** for imbalanced multi-label datasets.
* Macro-F1 is a more honest metric than accuracy for emotion recognition.
* Strong baselines are essential before moving to Transformers.

---

## 📌 Future Work

* Per-label threshold optimization
* Focal loss or class-weighted loss
* Ensemble of multiple Transformer seeds
* Cross-dataset generalization experiments

---

## 👤 Author
**Phan Huu Quoc Hanh**

This project was developed as a **research-oriented NLP portfolio project**, with a strong focus on real-world data challenges and evaluation rigor.
