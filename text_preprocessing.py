########################################################
# Text preprocessing for movie-review sentiment
#
# This lives in its own file on purpose. The saved model contains a
# TfidfVectorizer that uses the function "tokenize" below, and Joblib stores
# it as a reference to "text_preprocessing.tokenize". If the function were
# defined inside the training script, the prediction script could not load
# the model. Keep this file next to the model when you use it.
########################################################
import re
from functools import lru_cache

import nltk

NLTK_PACKAGES = {
    "movie_reviews": "corpora/movie_reviews",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

# Words that flip the meaning of a review ("not good") are kept, even though
# they are on NLTK's usual stop-word list.
NEGATIONS = {"not", "no", "nor", "never", "neither", "none", "nobody", "nothing", "nowhere"}

_WORD = re.compile(r"[a-z]+")
_STOP_WORDS = None
_LEMMATIZER = None


def ensure_nltk_data(packages=None):
    """Download the NLTK data this project needs, if it is not installed yet."""
    for package, path in NLTK_PACKAGES.items():
        if packages is not None and package not in packages:
            continue
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def _stop_words():
    global _STOP_WORDS
    if _STOP_WORDS is None:
        from nltk.corpus import stopwords
        _STOP_WORDS = set(stopwords.words("english")) - NEGATIONS
    return _STOP_WORDS


def _lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


@lru_cache(maxsize=None)
def _lemmatize(word):
    """Cached, because the same words appear again and again."""
    return _lemmatizer().lemmatize(word)


@lru_cache(maxsize=None)
def tokenize(text):
    """Turn a review into a tuple of clean tokens.

    The result is cached per review text: cross-validation sends the same
    review through this function many times, and the cache makes every repeat
    instant. Tuples are used because cached results must not be changed.

    1. lowercase, and write "n't" as " not" ("didn't" becomes "did not")
    2. keep only letters
    3. drop stop words (but keep negations)
    4. reduce each word to its dictionary form ("movies" becomes "movie")
    """
    text = text.lower().replace("n't", " not")
    stop_words = _stop_words()
    return tuple(_lemmatize(word) for word in _WORD.findall(text)
                 if word not in stop_words and len(word) > 1)
