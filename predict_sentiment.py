#!/usr/bin/env python3
"""Movie-review sentiment prediction - inference only.

Loads the model saved by sentiment_analysis_ensemble.py (joblib) and scores
review texts. No training happens here. text_preprocessing.py must stay in the
same folder, because the saved model uses it. The model was trained on movie
reviews, so it is less reliable for other kinds of text.

Usage:
    python predict_sentiment.py                                (scores new_reviews.csv)
    python predict_sentiment.py new_reviews.csv                (CSV with a "text" column)
    python predict_sentiment.py reviews.txt                    (one review per line)
    python predict_sentiment.py --text "Great film, loved it"  (one or more --text options)
    python predict_sentiment.py new_reviews.csv --output scored_reviews.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import pandas as pd
import sklearn

import text_preprocessing

LOGGER = logging.getLogger("sentiment_predict")

DEFAULT_MODEL_PATH = Path("outputs/sentiment_model.joblib")
DEFAULT_METADATA_PATH = Path("outputs/model_metadata.json")
DEFAULT_INPUT_PATH = Path("new_reviews.csv")
MIN_WORDS = 5


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read the model card written during training."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}. Run sentiment_analysis_ensemble.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    """Load the saved pipeline (TF-IDF + classifier) from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run sentiment_analysis_ensemble.py first.")
    text_preprocessing.ensure_nltk_data(["stopwords", "wordnet"])
    return joblib.load(path)


def warn_if_version_differs(metadata: Dict[str, Any]) -> None:
    """A model should be loaded with the scikit-learn version that built it."""
    trained_with = metadata.get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        LOGGER.warning("Model was trained with scikit-learn %s but %s is installed. "
                       "Results may be wrong or loading may fail.", trained_with, sklearn.__version__)


def read_reviews(path: Path) -> List[str]:
    """Read review texts from a CSV (column 'text') or a text file (one per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
        if "text" not in data.columns:
            raise ValueError("CSV input needs a column named 'text'")
        return data["text"].astype(str).tolist()

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_reviews(texts: List[str]) -> List[str]:
    """Reject empty input and warn about very short reviews."""
    texts = [t for t in texts if t.strip() and t.strip().lower() != "nan"]
    if not texts:
        raise ValueError("No review text found")

    short = [i for i, t in enumerate(texts) if len(t.split()) < MIN_WORDS]
    if short:
        LOGGER.warning("Reviews %s have fewer than %d words; short texts give unreliable results.",
                       short, MIN_WORDS)
    return texts


def score_reviews(model, texts: List[str], metadata: Dict[str, Any]) -> pd.DataFrame:
    """Score each review. 'Uncertain' marks probabilities close to 0.5."""
    probability = model.predict_proba(pd.Series(texts))[:, 1]
    band = metadata["uncertain_band"]
    return pd.DataFrame({
        "Review": [t if len(t) <= 60 else t[:57] + "..." for t in texts],
        "PositiveProbability": probability.round(3),
        "Sentiment": ["Positive" if p >= metadata["decision_threshold"] else "Negative" for p in probability],
        "Uncertain": [abs(p - 0.5) < band for p in probability],
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_sentiment.py",
        description="Score movie-review sentiment with the saved model (no retraining).")
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help=f"CSV with a 'text' column, or a text file with one review per line "
                             f"(default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--text", action="append", default=None, help="review text (can be repeated)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="joblib model file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="model metadata JSON")
    parser.add_argument("--output", type=Path, default=None, help="save scored reviews to CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    try:
        metadata = load_metadata(args.metadata)
        warn_if_version_differs(metadata)
        model = load_model(args.model)
        if args.text:
            texts = args.text
        else:
            texts = read_reviews(args.input or DEFAULT_INPUT_PATH)
        texts = check_reviews(texts)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1

    result = score_reviews(model, texts, metadata)
    LOGGER.info("Scored %d review(s)", len(result))
    print(result.to_string())
    print(f"\nTrained on: {metadata['training_domain']}. Results for other kinds of text may be less reliable.")

    if args.output:
        result.to_csv(args.output, index=False)
        LOGGER.info("Saved scored reviews to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
