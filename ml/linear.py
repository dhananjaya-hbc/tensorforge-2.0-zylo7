"""Model A: character + word n-gram TF-IDF features with three logistic-regression heads.

Character n-grams absorb typos and inconsistent transliteration (drivr / driver, Singlish spellings)
and work for every script without a tokenizer. The denoised view carries the signal; the full view
keeps context the denoiser may have removed.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.labels import CAT2ID, CATEGORIES
from ml.preprocess import Preprocessor

N = len(CATEGORIES)


@dataclass
class LinearModel:
    c_primary: float = 10.0
    c_secondary: float = 10.0
    c_urgent: float = 5.0
    seed: int = 42
    pre: Preprocessor = field(default_factory=Preprocessor)

    def _vectorizers(self):
        return [
            ("denoised", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                         max_features=200_000, dtype=np.float32)),
            ("denoised", TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                         sublinear_tf=True, dtype=np.float32)),
            ("full", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
                                     max_features=100_000, dtype=np.float32)),
        ]

    def _views(self, rows):
        return {"denoised": [self.pre.denoised_view(r) for r in rows], "full": [self.pre.full_view(r) for r in rows]}

    def _features(self, rows, fit=False):
        views = self._views(rows)
        if fit:
            self.vecs = self._vectorizers()
            return sp.hstack([v.fit_transform(views[name]) for name, v in self.vecs]).tocsr()
        return sp.hstack([v.transform(views[name]) for name, v in self.vecs]).tocsr()

    def fit(self, rows: list[dict]) -> "LinearModel":
        self.pre.fit(rows)
        X = self._features(rows, fit=True)
        y = np.array([CAT2ID[r["category"]] for r in rows])
        self.primary = LogisticRegression(C=self.c_primary, class_weight="balanced", max_iter=3000,
                                          random_state=self.seed).fit(X, y)
        # Secondary: one binary "is this issue also present?" head per category that ever occurs as secondary.
        self.secondary = {}
        for k, cat in enumerate(CATEGORIES):
            yk = np.array([r["secondary_category"] == cat for r in rows])
            if yk.sum() >= 5:
                self.secondary[k] = LogisticRegression(C=self.c_secondary, class_weight="balanced", max_iter=3000,
                                                       random_state=self.seed).fit(X, yk)
        yu = np.array([bool(r["is_urgent"]) for r in rows])
        self.urgent = LogisticRegression(C=self.c_urgent, class_weight="balanced", max_iter=3000,
                                         random_state=self.seed).fit(X, yu)
        return self

    def predict_proba(self, rows: list[dict]) -> dict[str, np.ndarray]:
        X = self._features(rows)
        sec = np.zeros((len(rows), N), dtype=np.float32)
        for k, clf in self.secondary.items():
            sec[:, k] = clf.predict_proba(X)[:, 1]
        return {"primary": self.primary.predict_proba(X), "secondary": sec, "urgent": self.urgent.predict_proba(X)[:, 1]}
