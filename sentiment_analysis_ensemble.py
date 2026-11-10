########################################################
# Sentiment Analysis using Ensemble Learning
#
# Pipeline : TfidfVectorizer (NLTK preprocessing) -> classifier
# Data     : NLTK movie_reviews corpus, 2,000 reviews, half positive and
#            half negative (Pang and Lee, polarity dataset v2.0)
# Goal     : Do Bagging and Boosting ensembles beat simple baseline
#            classifiers on text? The answer is measured, not assumed.
########################################################

########################################################
# Required Python Packages
########################################################
import json
import os
import platform
import warnings
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from text_preprocessing import ensure_nltk_data, tokenize

########################################################
# Settings
########################################################

warnings.filterwarnings("ignore", category=FutureWarning)

SHOW_PLOTS = False  # False: only save plots to files (works without a display)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must come after backend selection

########################################################
# File Paths
########################################################

OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_PATH = os.path.join(OUTPUT_DIR, "sentiment_model.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

########################################################
# Labels
########################################################

CLASS_LABELS = ["Negative", "Positive"]   # 0 = Negative, 1 = Positive (positive is the "positive class")

########################################################
# Model Settings
########################################################

RANDOM_STATE = 42
TRAIN_PERCENTAGE = 0.8
CV_FOLDS = 5
MODEL_STEP = "model"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


########################################################
# Function name : candidate_models
# Description   : The models to compare, each with a small tuning grid.
#                 Baselines : Naive Bayes, Logistic Regression, one Decision Tree
#                 Ensembles : Bagging, Random Forest, AdaBoost, Gradient
#                             Boosting, Voting
# Input         : Nothing
# Output        : Dictionary {name: (kind, estimator, parameter grid)}
# Author        : Prathmesh Sunil Girme
########################################################
def candidate_models():
    step = MODEL_STEP
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_STATE)),
            ("nb", MultinomialNB(alpha=0.5)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)),
        ],
        voting="soft",
    )
    return {
        "Naive Bayes": ("Baseline", MultinomialNB(), {f"{step}__alpha": [0.1, 0.5, 1.0]}),
        "Logistic Regression": ("Baseline", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                {f"{step}__C": [0.1, 1, 10, 100]}),
        "Decision Tree (single)": ("Baseline", DecisionTreeClassifier(random_state=RANDOM_STATE),
                                   {f"{step}__max_depth": [10, None]}),
        "Bagging (trees)": ("Ensemble", BaggingClassifier(estimator=DecisionTreeClassifier(),
                                                          n_estimators=50, max_features=0.3,
                                                          random_state=RANDOM_STATE, n_jobs=1), {}),
        "Random Forest": ("Ensemble", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                                             n_jobs=1), {}),
        "AdaBoost": ("Ensemble", AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE), {}),
        "Gradient Boosting": ("Ensemble", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                                     random_state=RANDOM_STATE), {}),
        "Voting (LR + NB + RF)": ("Ensemble", voting, {}),
    }


########################################################
# Function name : read_data
# Description   : Load the 2,000 movie reviews from the NLTK corpus
# Input         : Nothing
# Output        : Dataframe with file id, text and label (0 = neg, 1 = pos)
# Author        : Prathmesh Sunil Girme
########################################################
def read_data():
    ensure_nltk_data()
    from nltk.corpus import movie_reviews

    rows = [{"fileid": fileid, "text": movie_reviews.raw(fileid),
             "label": int(movie_reviews.categories(fileid)[0] == "pos")}
            for fileid in movie_reviews.fileids()]
    return pd.DataFrame(rows)


########################################################
# Function name : validate_dataset
# Description   : Check columns, labels and empty reviews
# Input         : Dataframe
# Output        : Nothing (raises ValueError if the data is not usable)
# Author        : Prathmesh Sunil Girme
########################################################
def validate_dataset(dataset):
    """Stop early with a clear message if the dataset is not as expected"""
    if not {"text", "label"}.issubset(dataset.columns):
        raise ValueError("Dataset needs 'text' and 'label' columns")
    if set(dataset["label"].unique()) - {0, 1}:
        raise ValueError("'label' must contain only 0 and 1")
    if (dataset["text"].str.strip() == "").any():
        raise ValueError("Dataset contains empty reviews")


########################################################
# Function name : remove_duplicates
# Description   : Drop reviews with identical text. A duplicate in both the
#                 training and test set would make the test score too good.
# Input         : Dataframe
# Output        : Dataframe without duplicates
# Author        : Prathmesh Sunil Girme
########################################################
def remove_duplicates(dataset):
    before = len(dataset)
    dataset = dataset.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Duplicate reviews removed :: {before - len(dataset)} (kept {len(dataset)})")
    return dataset


########################################################
# Function name : dataset_statistics
# Description   : Class balance and review length
# Input         : Dataframe
# Output        : Nothing (prints)
# Author        : Prathmesh Sunil Girme
########################################################
def dataset_statistics(dataset):
    counts = dataset["label"].value_counts().sort_index()
    for label in counts.index:
        print(f"Class {label} ({CLASS_LABELS[label]}) :: {counts[label]} reviews ({counts[label] / len(dataset):.1%})")
    print(f"Majority-class baseline accuracy :: {counts.max() / len(dataset):.3f}")

    words = dataset["text"].str.split().str.len()
    print(f"\nWords per review :: mean {words.mean():.0f}, median {words.median():.0f}, "
          f"min {words.min()}, max {words.max()}")
    print("Words per review by class:\n", words.groupby(dataset["label"]).mean().round(0).rename(CLASS_LABELS.__getitem__))


########################################################
# Function name : finish_plot
# Description   : Save the current figure (and show it if SHOW_PLOTS is True)
# Input         : File name
# Output        : Nothing (writes a PNG)
# Author        : Prathmesh Sunil Girme
########################################################
def finish_plot(file_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Plot saved to {path}")


########################################################
# Function name : plot_eda
# Description   : Class balance and review length by class
# Input         : Dataframe
# Output        : outputs/figures/eda_overview.png
# Author        : Prathmesh Sunil Girme
########################################################
def plot_eda(dataset):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    counts = dataset["label"].value_counts().sort_index()
    axes[0].bar(CLASS_LABELS, counts.values, color=["#e45756", "#4c78a8"])
    for i, value in enumerate(counts.values):
        axes[0].text(i, value, str(value), ha="center", va="bottom")
    axes[0].set_title("Class Balance")
    axes[0].set_ylabel("Reviews")

    words = dataset["text"].str.split().str.len()
    for label in (0, 1):
        axes[1].hist(words[dataset["label"] == label], bins=40, alpha=0.6, label=CLASS_LABELS[label])
    axes[1].set_title("Review Length (words)")
    axes[1].set_xlabel("Words per review")
    axes[1].legend()
    finish_plot("eda_overview.png")


########################################################
# Function name : split_dataset
# Description   : Stratified train/test split
# Input         : Dataframe, train percentage
# Output        : train_x, test_x, train_y, test_y (reviews and labels)
# Author        : Prathmesh Sunil Girme
########################################################
def split_dataset(dataset, train_percentage, random_state=RANDOM_STATE):
    return train_test_split(dataset["text"], dataset["label"], train_size=train_percentage,
                            random_state=random_state, stratify=dataset["label"])


########################################################
# Function name : build_pipeline
# Description   : Build a Pipeline:
#                 TfidfVectorizer : NLTK tokenizer, 1-2 word phrases, words in
#                                   3+ reviews, top 5,000 features. It is fitted
#                                   inside each CV fold, so the vocabulary
#                                   never sees the validation reviews.
#                 model           : the classifier
# Input         : Classifier
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
########################################################
def build_pipeline(estimator):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                 ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=5000,
                                 sublinear_tf=True)
    return Pipeline(steps=[("tfidf", vectorizer), (MODEL_STEP, estimator)])


########################################################
# Function name : make_cv
# Description   : Cross-validation splitter shared by every comparison
# Input         : Nothing
# Output        : StratifiedKFold object
# Author        : Prathmesh Sunil Girme
########################################################
def make_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


########################################################
# Function name : tune_candidate
# Description   : Grid search one algorithm with 5-fold CV on TRAINING data
# Input         : Model name, kind, classifier, parameter grid, training data
# Output        : Fitted best pipeline, dictionary of CV scores
# Author        : Prathmesh Sunil Girme
########################################################
def tune_candidate(name, kind, estimator, grid, train_x, train_y):
    search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING, refit="accuracy",
                          cv=make_cv(), n_jobs=-1)
    search.fit(train_x, train_y)

    best = search.best_index_
    results = search.cv_results_
    settings = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in search.best_params_.items()) or "defaults"
    row = {
        "Model": name,
        "Kind": kind,
        "Settings": settings,
        "CV_Accuracy": results["mean_test_accuracy"][best],
        "CV_Accuracy_Std": results["std_test_accuracy"][best],
        "CV_Precision": results["mean_test_precision"][best],
        "CV_Recall": results["mean_test_recall"][best],
        "CV_F1": results["mean_test_f1"][best],
        "CV_ROC_AUC": results["mean_test_roc_auc"][best],
    }
    print(f"  {name:<24} [{kind:<8}] {settings:<22} CV accuracy {row['CV_Accuracy']:.4f} "
          f"(+/- {row['CV_Accuracy_Std']:.4f})")
    return search.best_estimator_, row


########################################################
# Function name : evaluate_model
# Description   : Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
# Input         : Model, reviews, labels
# Output        : Dictionary of metrics, predicted labels, probabilities
# Author        : Prathmesh Sunil Girme
########################################################
def evaluate_model(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    predictions = (probability >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }
    return metrics, predictions, probability


########################################################
# Function name : compare_models
# Description   : Evaluate every tuned model on the test set ONCE
# Input         : Dictionary {name: model}, CV rows, train and test data
# Output        : Comparison dataframe
# Author        : Prathmesh Sunil Girme
########################################################
def compare_models(models, cv_rows, train_x, train_y, test_x, test_y):
    rows = []
    for row in cv_rows:
        model = models[row["Model"]]
        train_metrics, _, _ = evaluate_model(model, train_x, train_y)
        test_metrics, _, _ = evaluate_model(model, test_x, test_y)
        rows.append({
            **row,
            "Train_Accuracy": train_metrics["accuracy"],
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Precision": test_metrics["precision"],
            "Test_Recall": test_metrics["recall"],
            "Test_F1": test_metrics["f1"],
            "Test_ROC_AUC": test_metrics["roc_auc"],
        })
    return pd.DataFrame(rows)


########################################################
# Function name : select_best_model
# Description   : Choose the final model by CV accuracy rounded to 3 decimals
#                 (smaller differences count as ties), then by CV F1.
#                 The classes are balanced, so accuracy is a fair measure.
#                 The test set is not used.
# Input         : Comparison dataframe
# Output        : Name of the best model
# Author        : Prathmesh Sunil Girme
########################################################
def select_best_model(comparison):
    ranked = comparison.assign(_acc=comparison["CV_Accuracy"].round(3)).sort_values(
        ["_acc", "CV_F1"], ascending=False)
    return ranked.iloc[0]["Model"]


########################################################
# Function name : bootstrap_accuracy_interval
# Description   : 95% interval for the test accuracy by resampling test reviews
# Input         : True labels, predicted labels
# Output        : Lower and upper accuracy
# Author        : Prathmesh Sunil Girme
########################################################
def bootstrap_accuracy_interval(y_true, predictions, runs=2000):
    rng = np.random.RandomState(RANDOM_STATE)
    correct = (np.asarray(y_true) == predictions).astype(float)
    samples = [rng.choice(correct, size=len(correct)).mean() for _ in range(runs)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


########################################################
# Function name : plot_model_comparison
# Description   : CV accuracy next to test accuracy; baselines and ensembles
#                 are colored differently
# Input         : Comparison dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_model_comparison(comparison):
    ordered = comparison.sort_values("CV_Accuracy")
    positions = np.arange(len(ordered))
    colors = ["#4c78a8" if kind == "Baseline" else "#f58518" for kind in ordered["Kind"]]

    plt.figure(figsize=(9, 5.5))
    plt.barh(positions + 0.2, ordered["CV_Accuracy"], height=0.4, color=colors, label="Cross-validation (train)")
    plt.barh(positions - 0.2, ordered["Test_Accuracy"], height=0.4, color=colors, alpha=0.45, label="Test set")
    plt.yticks(positions, ordered["Model"])
    plt.axvline(0.5, linestyle="--", color="gray")
    plt.xlim(0.45, 0.95)
    plt.xlabel("Accuracy (dark = cross-validation, light = test; blue = baseline, orange = ensemble)")
    plt.title("Baselines vs Ensembles")
    finish_plot("model_comparison.png")


########################################################
# Function name : plot_confusion_matrix
# Description   : Display Confusion Matrix
# Input         : True labels, predicted labels, title, file name
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_confusion_matrix(y_true, y_pred, title, file_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.matshow(cm, cmap="Blues")
    fig.colorbar(image)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=40)
    finish_plot(file_name)


########################################################
# Function name : plot_roc_curves
# Description   : ROC curves of all models on the test set
# Input         : Dictionary {name: model}, test data
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_roc_curves(models, test_x, test_y):
    plt.figure(figsize=(7, 5.5))
    for name, model in models.items():
        probability = model.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, probability)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(test_y, probability):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curves (test set, AUC in brackets)")
    plt.legend(fontsize=7, loc="lower right")
    finish_plot("roc_curves.png")


########################################################
# Function name : top_words
# Description   : The words the Logistic Regression model links most strongly
#                 to positive and negative reviews
# Input         : Fitted Logistic Regression pipeline, number of words
# Output        : Dataframe with the strongest positive and negative words
# Author        : Prathmesh Sunil Girme
########################################################
def top_words(model, count=15):
    vocabulary = np.array(model.named_steps["tfidf"].get_feature_names_out())
    weights = model.named_steps[MODEL_STEP].coef_[0]
    order = np.argsort(weights)
    return pd.DataFrame({
        "Negative_Word": vocabulary[order[:count]], "Negative_Weight": weights[order[:count]],
        "Positive_Word": vocabulary[order[::-1][:count]], "Positive_Weight": weights[order[::-1][:count]],
    })


########################################################
# Function name : plot_top_words
# Description   : Bar chart of the strongest words
# Input         : Dataframe from top_words
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_top_words(words):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].barh(words["Negative_Word"][::-1], words["Negative_Weight"][::-1], color="#e45756")
    axes[0].set_title("Words that point to NEGATIVE reviews")
    axes[1].barh(words["Positive_Word"][::-1], words["Positive_Weight"][::-1], color="#4c78a8")
    axes[1].set_title("Words that point to POSITIVE reviews")
    for ax in axes:
        ax.set_xlabel("Logistic Regression weight")
    finish_plot("top_words.png")


########################################################
# Function name : predict_new_reviews
# Description   : Score new review texts
# Input         : Trained model, list of texts, uncertainty band
# Output        : Dataframe with probability and sentiment
# Author        : Prathmesh Sunil Girme
########################################################
def predict_new_reviews(model, texts, uncertain_band=0.1):
    probability = model.predict_proba(pd.Series(texts))[:, 1]
    return pd.DataFrame({
        "Review": [t if len(t) <= 60 else t[:57] + "..." for t in texts],
        "PositiveProbability": probability.round(3),
        "Sentiment": np.where(probability >= 0.5, "Positive", "Negative"),
        "Uncertain": np.abs(probability - 0.5) < uncertain_band,
    })


########################################################
# Function name : save_model / load_model
# Description   : Save or load the trained Pipeline with joblib.
#                 text_preprocessing.py must be importable when loading.
#                 Only load model files you created yourself.
# Input         : Model and path / path
# Output        : Nothing / model
# Author        : Prathmesh Sunil Girme
########################################################
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    ensure_nltk_data(["stopwords", "wordnet"])
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


########################################################
# Function name : json_default
# Description   : Let json.dump handle numpy numbers
# Author        : Prathmesh Sunil Girme
########################################################
def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


########################################################
# Function name : save_model_metadata
# Description   : Save a "model card" next to the .joblib file: label meaning,
#                 vocabulary size, library versions and test metrics.
# Output        : outputs/model_metadata.json
# Author        : Prathmesh Sunil Girme
########################################################
def save_model_metadata(model_name, model, train_rows, test_metrics, cv_accuracy, path=METADATA_PATH):
    estimator = model.named_steps[MODEL_STEP]
    simple_types = (int, float, str, bool, type(None))
    metadata = {
        "model_name": model_name,
        "model_class": type(estimator).__name__,
        "model_parameters": {k: v for k, v in estimator.get_params().items() if isinstance(v, simple_types)},
        "selection": {"rule": "best cross-validated accuracy on training data", "cv_accuracy": cv_accuracy},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_reviews": int(train_rows),
        "vocabulary_size": int(len(model.named_steps["tfidf"].vocabulary_)),
        "class_labels": {"0": CLASS_LABELS[0], "1": CLASS_LABELS[1]},
        "decision_threshold": 0.5,
        "uncertain_band": 0.1,
        "training_domain": "English movie reviews (NLTK movie_reviews corpus)",
        "test_metrics": test_metrics,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=json_default)
    print(f"Model metadata saved to {path}")


########################################################
# Function name : main
# Description   : Main function from where execution starts
# Author        : Prathmesh Sunil Girme
########################################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and understand the dataset
    print("\n--- 1) Dataset ---")
    dataset = read_data()
    validate_dataset(dataset)
    dataset = remove_duplicates(dataset)
    dataset_statistics(dataset)

    # 2) Exploratory plots
    print("\n--- 2) Exploratory analysis ---")
    plot_eda(dataset)

    # 3) Split (stratified)
    print("\n--- 3) Train/test split ---")
    train_x, test_x, train_y, test_y = split_dataset(dataset, TRAIN_PERCENTAGE)
    print("Train reviews :: ", len(train_x), f"(positive share {train_y.mean():.3f})")
    print("Test reviews  :: ", len(test_x), f"(positive share {test_y.mean():.3f})")

    # 4) Tune every model with the same CV folds
    print("\n--- 4) Tuning models (5-fold CV on training data) ---")
    models, cv_rows = {}, []
    for name, (kind, estimator, grid) in candidate_models().items():
        models[name], row = tune_candidate(name, kind, estimator, grid, train_x, train_y)
        cv_rows.append(row)

    # 5) Compare on the test set (once) and choose the final model
    print("\n--- 5) Model comparison ---")
    comparison = compare_models(models, cv_rows, train_x, train_y, test_x, test_y)
    comparison.round(4).to_csv(COMPARISON_PATH, index=False)
    print(comparison[["Model", "Kind", "CV_Accuracy", "Train_Accuracy", "Test_Accuracy", "Test_F1",
                      "Test_ROC_AUC"]].round(3).to_string(index=False))
    plot_model_comparison(comparison)
    plot_roc_curves(models, test_x, test_y)

    tree_cv = comparison.loc[comparison["Model"] == "Decision Tree (single)", "CV_Accuracy"].iloc[0]
    best_baseline = comparison[comparison["Kind"] == "Baseline"].sort_values("CV_Accuracy", ascending=False).iloc[0]
    print("\nEnsemble gain over ONE decision tree (CV accuracy, points):")
    for _, row in comparison[comparison["Kind"] == "Ensemble"].iterrows():
        print(f"  {row['Model']:<24} {100 * (row['CV_Accuracy'] - tree_cv):+.1f}")
    print(f"Best baseline: {best_baseline['Model']} (CV accuracy {best_baseline['CV_Accuracy']:.3f})")
    print("Ensembles vs the best baseline (CV accuracy, points):")
    for _, row in comparison[comparison["Kind"] == "Ensemble"].iterrows():
        print(f"  {row['Model']:<24} {100 * (row['CV_Accuracy'] - best_baseline['CV_Accuracy']):+.1f}")

    final_name = select_best_model(comparison)
    final_model = models[final_name]
    final_row = comparison[comparison["Model"] == final_name].iloc[0]
    print(f"\nFinal model (best CV accuracy) :: {final_name} (CV accuracy {final_row['CV_Accuracy']:.4f})")

    # 6) Detailed report for the final model
    final_metrics, predictions, probability = evaluate_model(final_model, test_x, test_y)
    low, high = bootstrap_accuracy_interval(test_y, predictions)
    final_metrics["accuracy_95_interval"] = [low, high]
    final_metrics["train_accuracy"] = float(final_row["Train_Accuracy"])
    print(f"\n===== FINAL: {final_name} =====")
    print(f"Test accuracy :: {final_metrics['accuracy']:.4f} (95% interval {low:.3f} to {high:.3f})")
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))
    print("Classification Report:\n", classification_report(test_y, predictions, target_names=CLASS_LABELS))
    plot_confusion_matrix(test_y, predictions, f"{final_name} - Confusion Matrix", "confusion_matrix_final.png")
    pd.DataFrame({"index": test_x.index, "actual": test_y.values, "predicted": predictions,
                  "positive_probability": probability.round(4)}).to_csv(PREDICTIONS_PATH, index=False)

    # 7) Which words matter? (from the Logistic Regression model)
    print("\n--- 6) Strongest words (Logistic Regression) ---")
    words = top_words(models["Logistic Regression"])
    print(words.round(2).to_string(index=False))
    plot_top_words(words)

    # 8) Predict for new reviews (written for this project)
    print("\n--- 7) New reviews (short examples written for this project) ---")
    new_reviews = [
        "A wonderful film with brilliant acting and a moving story.",
        "Boring, predictable and far too long. A complete waste of time.",
        "The plot was not good and the acting was not convincing.",
        "It was okay, some good scenes and some weak ones.",
    ]
    print(predict_new_reviews(final_model, new_reviews).to_string())

    # 9) Save the model and metadata, load the model, and test a sample
    print("\n--- 8) Save and load ---")
    save_model(final_model, MODEL_PATH)
    save_model_metadata(final_name, final_model, len(train_x), final_metrics, float(final_row["CV_Accuracy"]))
    loaded = load_model(MODEL_PATH)
    sample = pd.Series(["An excellent and enjoyable movie."])
    print(f"Loaded model prediction for a sample review: {CLASS_LABELS[int(loaded.predict(sample)[0])]}")

    summary = {
        "final_model": final_name,
        "selection_rule": "best cross-validated accuracy on training data (test set not used)",
        "final_model_test": final_metrics,
        "top_words": words.round(4).to_dict(orient="list"),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)
    print(f"Metrics saved to {METRICS_PATH}")


########################################################
# Application starter
########################################################

if __name__ == "__main__":
    main()
