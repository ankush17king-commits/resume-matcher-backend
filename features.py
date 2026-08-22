"""
features.py

Turns a (resume_text, jd_text) pair into a numeric feature vector.

Design choice: instead of feeding raw TF-IDF vectors straight into
logistic regression (which would need thousands of sparse dimensions
and mostly memorizes vocabulary), we engineer a small set of
INTERPRETABLE features. This is a deliberate, defensible choice:
- Small, interpretable feature set -> you can explain every coefficient
  in an interview.
- Works fine on ~1000 rows of data (raw TF-IDF would overfit badly at
  this dataset size).
- Mirrors how real "fit scoring" systems are often built in practice:
  a handful of engineered similarity/overlap signals feeding a simple,
  auditable model -- easier to explain to non-technical stakeholders
  than a black-box embedding classifier.

Features used:
1. tfidf_cosine_sim   - cosine similarity of TF-IDF vectors (resume vs JD)
2. keyword_overlap    - number of shared keyword tokens
3. keyword_jaccard    - Jaccard similarity of keyword sets
4. resume_len         - word count of resume (normalized)
5. jd_len             - word count of JD (normalized)
6. len_ratio          - resume length / JD length (very short resumes
                         vs long JDs, or vice versa, are a weak signal)
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.]*")


def tokenize(text):
    return set(t.lower() for t in TOKEN_RE.findall(text))


class FeatureExtractor:
    """
    Fits a shared TF-IDF vocabulary on the training corpus (all resumes +
    all JDs), then converts any (resume, jd) pair into a feature vector.
    Must be fit once on the training set and reused (via predict.py) on
    new inference examples -- exactly like you'd do with any sklearn
    preprocessing step, to avoid train/inference skew.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

    def fit(self, resumes, jds):
        corpus = list(resumes) + list(jds)
        self.vectorizer.fit(corpus)
        return self

    def _tfidf_cosine(self, resume_text, jd_text):
        vecs = self.vectorizer.transform([resume_text, jd_text])
        return float(cosine_similarity(vecs[0], vecs[1])[0][0])

    def transform_one(self, resume_text, jd_text):
        resume_tokens = tokenize(resume_text)
        jd_tokens = tokenize(jd_text)

        overlap = len(resume_tokens & jd_tokens)
        union = len(resume_tokens | jd_tokens)
        jaccard = overlap / union if union else 0.0

        resume_len = len(resume_text.split())
        jd_len = len(jd_text.split())
        len_ratio = resume_len / jd_len if jd_len else 0.0

        return [
            self._tfidf_cosine(resume_text, jd_text),
            overlap,
            jaccard,
            resume_len / 50.0,   # scaled down, roughly O(1) with other feats
            jd_len / 50.0,
            len_ratio,
        ]

    def transform(self, resumes, jds):
        return np.array([
            self.transform_one(r, j) for r, j in zip(resumes, jds)
        ])


FEATURE_NAMES = [
    "tfidf_cosine_sim",
    "keyword_overlap",
    "keyword_jaccard",
    "resume_len_scaled",
    "jd_len_scaled",
    "len_ratio",
]
