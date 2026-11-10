# Sentiment Analysis using Ensemble Learning & Pipeline

This project demonstrates an **end-to-end text classification workflow** that decides whether a movie review is **positive or negative**. It asks one clear question: **do Bagging and Boosting ensembles beat simple baseline classifiers on text?** Eight models are compared with one shared pipeline and one fair process, and the answer is measured, not assumed.

It follows **industrial best practices** by:

- Cleaning text with **NLTK** (tokenizing, stop-word removal, lemmatization) while **keeping negations** such as "not"
- Putting **TF-IDF vectorization inside a Pipeline**, so the vocabulary is learned from training reviews only (no data leakage)
- Giving every model the **same cross-validation folds**
- Choosing the final model with **cross-validation on training data only**, so the test set is never used to pick a winner
- Comparing ensembles with both a **strong baseline** (Logistic Regression) and a **weak baseline** (one Decision Tree), so gains are measured against the right thing
- Reporting a **confidence interval** for the test accuracy, because 400 test reviews is a small sample
- Saving the trained pipeline with **Joblib** and reusing it from a separate **prediction script**
- Saving plots, tables and metrics to files for **reproducibility**

---

## Dependencies

Requires **Python 3.9+** (tested on Python 3.12). Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy matplotlib scikit-learn joblib nltk
```

The NLTK data (`movie_reviews`, `stopwords`, `wordnet`) is **downloaded automatically on the first run**, so the first run needs an internet connection.

---

## Dataset Information

**Source:** the `movie_reviews` corpus from NLTK, which is the polarity dataset v2.0 by Pang and Lee ("A Sentimental Education", ACL 2004). It is not stored in this repository. The script loads it through NLTK.

- **2,000 reviews:** 1,000 positive and 1,000 negative (balanced), no duplicate texts
- **Length:** about 746 words per review on average (17 to 2,678), positive reviews a little longer than negative ones
- **Labels:** 0 = Negative, 1 = Positive (Positive is the "positive class" for precision, recall and F1)

Because the classes are balanced, **accuracy is a fair measure** here, and an "always guess one class" model scores 50%.

---

## Workflow

### Text Preprocessing (`text_preprocessing.py`)

1. Lowercase, and write "n't" as " not" ("didn't" becomes "did not")
2. Keep letters only
3. Remove stop words (NLTK English list), **but keep negations** such as "not", "no" and "never", because they flip the meaning of a review
4. Reduce each word to its dictionary form with the WordNet lemmatizer ("movies" becomes "movie")

### Train-Test Split

- Stratified split into **80% train (1,600 reviews)** and **20% test (400 reviews)**

### Pipeline Construction

- **Step 1:** `TfidfVectorizer` using the NLTK tokenizer, single words and two-word phrases (so "not good" is one feature), words that appear in at least 3 reviews, the top 5,000 features, and sub-linear term frequency
- **Step 2:** the classifier

The vectorizer sits inside the pipeline, so inside every cross-validation fold its vocabulary is built from that fold's training reviews only.

### Models Compared

| Model | Type | Tuning (5-fold CV on training data) |
|-------|------|-------------------------------------|
| Naive Bayes | Baseline | `alpha` in 0.1, 0.5, 1.0 |
| Logistic Regression | Baseline | `C` in 0.1, 1, 10, 100 |
| Decision Tree (single) | Weak baseline | `max_depth` in 10, None |
| Bagging (50 trees) | Ensemble | fixed |
| Random Forest (300 trees) | Ensemble | fixed |
| AdaBoost (200 rounds) | Ensemble | fixed |
| Gradient Boosting (100 rounds) | Ensemble | fixed |
| Voting (Logistic Regression + Naive Bayes + Random Forest) | Ensemble | fixed |

### Model Selection

- The final model has the best **cross-validated accuracy**, rounded to 3 decimals so that tiny differences count as ties (ties go to the higher F1)
- The test set is evaluated **once**, after the choice is made

### Model Saving & Loading

- Save the full pipeline with **Joblib** (`joblib.dump`)
- Save a **metadata file** (label meaning, vocabulary size, library versions, metrics)
- Load both in a separate script (`joblib.load`) and score new reviews without retraining

---

## Running the Project

```bash
python sentiment_analysis_ensemble.py
```

The run takes about 10 minutes on my single-core test machine, and much less on a multi-core PC. Expected output (shortened):

```text
Class 0 (Negative) :: 1000 reviews (50.0%)
Class 1 (Positive) :: 1000 reviews (50.0%)
Train reviews ::  1600 (positive share 0.500)
Test reviews  ::  400 (positive share 0.500)

  Naive Bayes              [Baseline] alpha=1.0              CV accuracy 0.8438 (+/- 0.0145)
  Logistic Regression      [Baseline] C=10                   CV accuracy 0.8750 (+/- 0.0120)
  Decision Tree (single)   [Baseline] max_depth=10           CV accuracy 0.6462 (+/- 0.0139)
  ...

Final model (best CV accuracy) :: Logistic Regression (CV accuracy 0.8750)
Test accuracy :: 0.8625 (95% interval 0.828 to 0.895)
...
Model saved to outputs/sentiment_model.joblib
```

**Generated files** (in `outputs/`):

```text
outputs/
├── figures/
│   ├── eda_overview.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrix_final.png
│   └── top_words.png
├── model_comparison.csv
├── test_predictions.csv          # actual vs predicted for the 400 test reviews (no review text)
├── metrics.json
├── model_metadata.json
└── sentiment_model.joblib
```

---

## Results

All numbers are from `random_state=42`. The test set has 400 reviews (200 positive, 200 negative). **With 400 reviews, the 95% interval around any accuracy is about +/- 3.5 points**, so differences of one or two points are not meaningful.

**Model comparison:**

| Model | Type | CV Acc | Train Acc | Test Acc | Test F1 | Test ROC-AUC |
|-------|------|--------|-----------|----------|---------|--------------|
| Always guess one class | - | - | - | 0.500 | - | 0.500 |
| Naive Bayes | Baseline | 0.844 | 0.929 | 0.818 | 0.815 | 0.917 |
| **Logistic Regression (selected)** | **Baseline** | **0.875** | 1.000 | **0.862** | **0.866** | **0.946** |
| Decision Tree (single) | Weak baseline | 0.646 | 0.888 | 0.642 | 0.664 | 0.645 |
| Bagging | Ensemble | 0.789 | 1.000 | 0.800 | 0.810 | 0.890 |
| Random Forest | Ensemble | 0.831 | 1.000 | 0.828 | 0.825 | 0.922 |
| AdaBoost | Ensemble | 0.787 | 0.919 | 0.808 | 0.811 | 0.888 |
| Gradient Boosting | Ensemble | 0.787 | 0.975 | 0.808 | 0.814 | 0.888 |
| Voting (LR + NB + RF) | Ensemble | 0.873 | 1.000 | 0.855 | 0.856 | 0.944 |

**Ensemble results in cross-validated accuracy points:**

| Ensemble | Gain over ONE decision tree | Difference from the best baseline (Logistic Regression) |
|----------|-----------------------------|----------------------------------------------------------|
| Bagging | +14.3 | -8.6 |
| Random Forest | +18.4 | -4.4 |
| AdaBoost | +14.1 | -8.8 |
| Gradient Boosting | +14.1 | -8.8 |
| Voting (LR + NB + RF) | +22.7 | -0.2 (a tie) |

**Selected model (Logistic Regression), test set:** accuracy 0.8625 (95% interval 0.828 to 0.895), precision 0.847, recall 0.885, F1 0.866, ROC-AUC 0.946.

```text
                 Pred Negative  Pred Positive
Actual Negative        168             32
Actual Positive         23            177
```

**Strongest words** (Logistic Regression weights):

| Point to NEGATIVE | bad, worst, nothing, supposed, boring, ridiculous, plot, unfortunately, awful, mess |
|---|---|
| **Point to POSITIVE** | **great, well, hilarious, overall, fun, perfect, excellent, performance, job, terrific** |

**Key findings:**

1. **Ensembles are a big upgrade over a single decision tree.** Every ensemble gains 14 to 23 accuracy points over one tree (0.646 up to 0.787 to 0.873). Ensembling works.
2. **No ensemble beats the simple Logistic Regression baseline.** The Voting ensemble ties it (-0.2 points), Random Forest is 4 points behind, and Bagging, AdaBoost and Gradient Boosting are about 9 points behind. The honest answer to the project's question is **"not on this kind of data"**.
3. **Why:** TF-IDF turns each review into thousands of mostly-zero columns, which suits linear models well. Trees must split on words that appear in only a few reviews, and with 1,600 training reviews they have little to learn from.
4. **Voting is competitive only because it contains Logistic Regression.** It is the only ensemble that comes close.
5. **The final test accuracy matches cross-validation.** 0.862 on the test set against 0.875 in cross-validation, so the choice generalises. Train accuracy of 1.00 is normal for text with thousands of features. What matters is the held-out score.
6. **The important words make sense.** "bad", "worst", "boring", "awful" and "waste" point to negative reviews, and "great", "hilarious", "perfect" and "excellent" to positive ones. A few words, such as "plot" and "script" (negative) or "performance" and "job" (positive), reflect how critics write about films, not the feelings of the words themselves.

**Limits:**

- **Meaning is not understood.** The model counts words and short phrases. "Not bad at all, I was pleasantly surprised." is a positive sentence, but the model scores it **negative** (probability 0.10).
- **Mixed reviews are uncertain.** "It was okay, some good scenes and some weak ones." gets a probability of 0.49 and is flagged as uncertain.
- **Domain.** The model was trained on English movie reviews, so it is less reliable for products, restaurants or social media.
- **Size.** 2,000 reviews is small by modern standards. This is a method comparison, not a production sentiment system.

---

## Visualizations

- **Class balance and review length:** `outputs/figures/eda_overview.png`
- **Baselines vs ensembles:** `outputs/figures/model_comparison.png`
- **ROC curves:** `outputs/figures/roc_curves.png`
- **Confusion matrix:** `outputs/figures/confusion_matrix_final.png`
- **Strongest words:** `outputs/figures/top_words.png`

Set `SHOW_PLOTS = True` at the top of the script to also open the plots in a window.

---

## Model Persistence with Joblib (Train Once, Use Many Times)

**The idea:** training takes minutes and needs the whole dataset. Scoring a review should take milliseconds and should not need either. So the trained model is written to a file with `joblib.dump`, and any other program can bring it back with `joblib.load` and call `predict`.

```text
 TRAINING (run once)                          PREDICTION (run any time)
 sentiment_analysis_ensemble.py               predict_sentiment.py
 ┌──────────────────────┐                     ┌──────────────────────────┐
 │ NLTK movie reviews   │                     │ new_reviews.csv / text   │
 │ tune + compare models│                     │ load model + metadata    │
 │ joblib.dump(model)   │ ──► .joblib  ──►    │ clean + vectorize        │
 │ write metadata.json  │ ──► .json    ──►    │ predict_proba            │
 └──────────────────────┘                     └──────────────────────────┘
```

**What is stored:**

- `outputs/sentiment_model.joblib`: the **whole pipeline** (the fitted TF-IDF vocabulary and the classifier), so new reviews are cleaned and turned into numbers exactly like the training reviews
- `outputs/model_metadata.json`: the label meaning, the vocabulary size (5,000), the training domain, the uncertainty band, library versions and the test metrics

**Why `text_preprocessing.py` is a separate file:** the saved model contains a TF-IDF vectorizer that uses the function `tokenize`, and Joblib stores it as a reference to `text_preprocessing.tokenize`. If that function lived inside the training script, no other script could load the model. **Keep `text_preprocessing.py` in the same folder as the model whenever you load it.**

**Score new reviews (no training, no dataset needed):**

```bash
python predict_sentiment.py
python predict_sentiment.py --text "Great film, I loved it"
python predict_sentiment.py reviews.txt --output scored_reviews.csv
```

With no argument the script scores `new_reviews.csv` (6 short reviews written for this project). It also accepts a CSV with a `text` column, a text file with one review per line, or one or more `--text` options. Train first (`python sentiment_analysis_ensemble.py`) so the `outputs/` folder with the model exists.

```text
                                                       Review  PositiveProbability Sentiment  Uncertain
0  A wonderful film with brilliant acting and a moving story.                0.768  Positive      False
1  Boring, predictable and far too long. A complete waste of...              0.005  Negative      False
2    The plot was not good and the acting was not convincing.                0.064  Negative      False
3           It was okay, some good scenes and some weak ones.                0.488  Negative       True
4              I loved every minute of it, a real masterpiece.               0.817  Positive      False
5                  Not bad at all, I was pleasantly surprised.               0.100  Negative      False
```

Review 5 is a real miss (see the limits above). Review 3 is marked **Uncertain** because its probability is within 0.1 of 0.5.

**What the prediction script checks:** empty input, a CSV without a `text` column, reviews shorter than 5 words (warning), and a scikit-learn version different from the one that trained the model (warning).

**Rules for using saved models:**

1. **Same library version:** load the model with the scikit-learn version that trained it.
2. **Trusted files only:** Joblib files can run code when opened, so load only files you created yourself.
3. **Retrain when the data changes:** a saved model does not learn. With new labelled reviews, rerun `sentiment_analysis_ensemble.py`.
4. **Git:** `.joblib` files are in `.gitignore`. After cloning, run `python sentiment_analysis_ensemble.py` once to create the model.

---

## Sample Prediction (from Python code)

```python
import pandas as pd
from sentiment_analysis_ensemble import load_model

model = load_model("outputs/sentiment_model.joblib")

reviews = pd.Series(["An excellent and enjoyable movie.", "Dull, slow and badly acted."])
print(model.predict_proba(reviews)[:, 1].round(3))   # probability of Positive
print(model.predict(reviews))                        # 1 = Positive, 0 = Negative
```

---

## Author

**Prathmesh Girme**  
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
