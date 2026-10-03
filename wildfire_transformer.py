import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler


class SkewAwareScaler(BaseEstimator, TransformerMixin):
    """
    Robust-scale highly skewed numeric columns and
    standard-scale the remaining numeric columns.

    Columns are selected during fit(), so the decision is
    made using training-fold data only.
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.feature_names_in_ = X.columns.to_list()

        skew = X.skew(numeric_only=True)

        high = (
            skew[skew.abs() > self.skew_threshold]
            .index
            .tolist()
        )

        low = (
            skew[skew.abs() <= self.skew_threshold]
            .index
            .tolist()
        )

        transformers = []

        if high:
            transformers.append(
                ("high_skew", RobustScaler(), high)
            )

        if low:
            transformers.append(
                ("low_skew", StandardScaler(), low)
            )

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )

        self.preprocessor_.fit(X)

        return self

    def transform(self, X):
        X = pd.DataFrame(
            X,
            columns=self.feature_names_in_,
        ).copy()

        return self.preprocessor_.transform(X)
