"""
predict.py

Loads the trained classifier + scaler + vectorizer and scores a new
(resume_text, jd_text) pair. Also reports missing keywords (JD terms
not found in the resume) as a simple explainability layer on top of
the classifier's probability output.
"""

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from features import FeatureExtractor, tokenize

MODEL_PATH = "models/classifier.joblib"
SCALER_PATH = "models/scaler.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"


class ResumeMatcher:
    def __init__(self):
        self.clf = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        # Rebuild a FeatureExtractor around the already-fitted vectorizer
        self.extractor = FeatureExtractor()
        self.extractor.vectorizer = vectorizer

    def score(self, resume_text: str, jd_text: str):
        X = self.extractor.transform([resume_text], [jd_text])
        X_s = self.scaler.transform(X)
        prob_good_fit = float(self.clf.predict_proba(X_s)[0][1])
        label = "good_fit" if prob_good_fit >= 0.5 else "poor_fit"

        resume_tokens = tokenize(resume_text)
        jd_tokens = tokenize(jd_text)
        missing = sorted(jd_tokens - resume_tokens)

        # Filter out stopwords/very short tokens so "missing keywords" stays useful
        missing = [
            w.strip(".,") for w in missing
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS
        ][:15]

        return {
            "match_score": round(prob_good_fit * 100, 1),  # as a percentage
            "prediction": label,
            "missing_keywords": missing,
        }


if __name__ == "__main__":
    matcher = ResumeMatcher()

    sample_resume = (
        "Skills: React, Node.js, Express.js, MongoDB, REST APIs, JWT "
        "authentication, JavaScript, Git. Built a full-stack college "
        "guidance platform with secure authentication."
    )
    sample_jd = (
        "We are hiring a backend engineer with experience in Node.js, "
        "Express.js, MongoDB, REST API design, and authentication systems. "
        "Familiarity with cloud deployment is a plus."
    )

    result = matcher.score(sample_resume, sample_jd)
    print(result)
