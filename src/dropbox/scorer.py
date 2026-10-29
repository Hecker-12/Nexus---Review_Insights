"""Sentiment scorer using pre-trained TF-IDF and Logistic Regression models."""

import os
import joblib
from typing import List, Tuple
import sklearn

# Module-level singletons (loaded ONCE)
_VECTORIZER = None
_CLASSIFIER = None


def _get_models():
    """Load vectorizer and classifier models once, cached in memory."""
    global _VECTORIZER, _CLASSIFIER
    if _VECTORIZER is None or _CLASSIFIER is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        vec_path = os.path.join(repo_root, "models", "tfidf_vectorizer.joblib")
        clf_path = os.path.join(repo_root, "models", "logreg_sentiment.joblib")

        _VECTORIZER = joblib.load(vec_path)
        _CLASSIFIER = joblib.load(clf_path)

        assert list(_CLASSIFIER.classes_) == ["negative", "positive"], (
            f"Expected classifier classes to be ['negative', 'positive'], got {_CLASSIFIER.classes_}"
        )

    return _VECTORIZER, _CLASSIFIER


def score_texts(texts: List[str]) -> List[Tuple[str, float]]:
    """Predict sentiment label and probability for a list of texts.

    Returns:
        List of (label, probability) tuples where probability is the confidence
        in the predicted class.
    """
    if not texts:
        return []

    vec, clf = _get_models()
    X = vec.transform(texts)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    classes_list = list(clf.classes_)
    results = []
    for pred, prob_dist in zip(preds, probs):
        class_idx = classes_list.index(pred)
        confidence = float(prob_dist[class_idx])
        results.append((str(pred), round(confidence, 4)))

    return results


def get_model_info() -> dict:
    """Return model classes and scikit-learn version."""
    _, clf = _get_models()
    return {
        "sklearn_version": sklearn.__version__,
        "classes": list(clf.classes_)
    }
