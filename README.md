# Resume ↔ Job Description Matcher (with trained classifier)

Predicts whether a resume is a "good fit" or "poor fit" for a job
description, using a **logistic regression classifier trained on
engineered NLP features** — not just a raw similarity score.

## Why a trained classifier, not just cosine similarity?

Cosine similarity alone treats "how similar is the text" as your
entire signal. Instead, this project:

1. Engineers multiple features from the resume/JD pair (TF-IDF
   similarity, keyword overlap, Jaccard similarity, length ratio).
2. **Trains** a logistic regression model on labeled examples to learn
   *how much each feature actually matters* for predicting fit.
3. Evaluates the model properly on a held-out test set (accuracy,
   precision, recall, F1, ROC-AUC, confusion matrix) rather than just
   eyeballing a few examples.

This is the difference between "I called a pretrained model" and
"I trained and evaluated a model" — the latter is what gets probed
in a technical interview for AI/data roles.

## Project structure

```
resume_matcher/
├── data/
│   └── generate_dataset.py   # synthesizes labeled resume-JD pairs
├── features.py                # feature engineering (TF-IDF, overlap, etc.)
├── train_classifier.py        # trains + evaluates logistic regression
├── predict.py                 # inference on new resume/JD pairs
├── app.py                     # FastAPI backend (/match endpoint)
├── models/                    # saved model, scaler, vectorizer (after training)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the synthetic labeled dataset
python3 data/generate_dataset.py

# 2. Train and evaluate the classifier
python3 train_classifier.py

# 3. Try inference on a sample pair
python3 predict.py

# 4. Start the API for your React frontend
uvicorn app:app --reload --port 8000
```

Then from React, POST a multipart form to `http://localhost:8000/match`
with `jd_text` and either `resume_text` or a `resume_file` (PDF).

## About the dataset

There's no public labeled "resume ↔ JD fit" dataset, so this project
**synthesizes one** (`data/generate_dataset.py`): resumes and JDs are
generated from domain skill pools (web dev, data science, android,
devops, design) with realistic noise (cross-domain skills), and
labeled with a soft, randomized rule based on skill overlap — not a
hard threshold, so the classifier has something genuine to learn
rather than just re-deriving the generation rule.

**Be upfront about this in an interview.** It's a normal, sensible
approach for a fresher project without access to real labeled data —
the important part is that you understand *why* you did it and what
its limitations are (see below).

## Results (on held-out 20% test set)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.66  |
| Precision | 0.49  |
| Recall    | 0.63  |
| F1        | 0.55  |
| ROC-AUC   | 0.65  |

These numbers are honest, not inflated — a real classifier on noisy,
synthetic labels. Being able to explain *why* it's ~65% and not 95%
(noisy labels, small feature set, synthetic data) is a stronger
interview answer than a suspiciously perfect score.

## Interview talking points (this is the important part)

**"Walk me through your pipeline."**
Text → tokenize/clean → engineer 6 features (TF-IDF cosine similarity,
keyword overlap, Jaccard similarity, length features) → standardize
features → logistic regression → probability of "good fit."

**"Why logistic regression and not a neural net?"**
Small dataset (~1,200 rows), need interpretable coefficients to explain
predictions, and it's the right-sized model for a structured, low-dimensional
feature set — a deep model would overfit and add no real accuracy gain here.

**"Why did you engineer features instead of raw TF-IDF vectors as input?"**
Raw TF-IDF vectors are high-dimensional and sparse; with only ~1,200
rows, feeding thousands of raw features into logistic regression would
overfit. A small set of engineered, interpretable features generalizes
better at this data size and lets me explain exactly what's driving
each prediction.

**"How did you evaluate it, and why not just look at accuracy?"**
Used precision/recall/F1/ROC-AUC on a held-out test set (not the
training set), because accuracy alone is misleading on an imbalanced
dataset (67% poor-fit / 33% good-fit here) — a model that always
predicts "poor fit" would score 67% accuracy while being useless.

**"What are the limitations, and how would you improve it?"**
- Labels are synthetic/soft-rule-based, not real hiring outcomes — a
  production version would need real labeled data (e.g. actual
  applicant outcomes) to be trustworthy.
- TF-IDF only captures lexical overlap, not semantic meaning — a
  resume saying "supervised model training" and a JD saying "machine
  learning experience" wouldn't score well without shared vocabulary.
  Adding sentence embeddings (`sentence-transformers`) as an extra
  feature is the natural next step; I designed `features.py` so a new
  feature can be added without changing the rest of the pipeline.
- Small feature set — could add named entity recognition (spaCy) to
  extract structured skills instead of raw token overlap.

## Optional upgrade: semantic embeddings

To add semantic similarity (catches synonyms, not just exact keyword
matches), install `sentence-transformers` and add one more feature to
`features.py`:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding_cosine(resume_text, jd_text):
    emb = model.encode([resume_text, jd_text])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]
```
**Frontend repo:** https://github.com/ankush17king-commits/resume-matcher-ui
