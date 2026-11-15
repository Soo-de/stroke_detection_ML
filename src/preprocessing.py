"""Feature engineering, encoding, scaling and SMOTE as one imblearn Pipeline.

    from preprocessing import load_data, build_pipeline

    X_train, X_test, y_train, y_test = load_data()
    pipe = build_pipeline(LogisticRegression(max_iter=3000), balancing="smote")
    pipe.fit(X_train, y_train)

Pipeline steps: fe -> encode -> balance (only with balancing="smote") -> model.
Model hyperparameters are addressed as "model__<name>" in GridSearchCV.
Every step is fitted on the training data only; SMOTE runs only during fit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import DATA_DIR, RANDOM_STATE, TARGET

CONTINUOUS = ["AGE", "TOTCHOL", "SYSBP", "DIABP", "BMI", "HEARTRTE", "GLUCOSE", "CIGPDAY"]
BINARY = ["CURSMOKE", "DIABETES", "BPMEDS", "PREVCHD", "PREVAP", "PREVMI", "PREVHYP"]
SEX = "SEX"  # 1 = male, 2 = female
ORDINAL = ["educ"]  # education level 1-4
FEATURES = CONTINUOUS + BINARY + [SEX] + ORDINAL

FE_NUMERIC = ["pulse_pressure", "mean_arterial_pressure", "comorbidity_count"]
FE_CATEGORICAL = ["bp_stage", "bmi_category"]

BALANCING_OPTIONS = ("smote", "class_weight", "none")


def split_xy(train, test):
    """Splits the train/test tables written by notebook 01 into X_train, X_test, y_train, y_test."""
    missing = [c for c in FEATURES + [TARGET] if c not in train.columns]
    if missing:
        raise ValueError(f"train verisinde beklenen sütunlar yok: {missing}")
    return train[FEATURES], test[FEATURES], train[TARGET].astype(int), test[TARGET].astype(int)


def load_data(data_dir=None):
    """Reads train.csv and test.csv from data_dir (default: config.DATA_DIR) and calls split_xy."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    paths = [data_dir / "train.csv", data_dir / "test.csv"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} bulunamadı. Önce 01 notebook'unu çalıştırın; Colab'da train.csv ve "
                "test.csv dosyalarını sol menüden yükleyin.")
    return split_xy(*(pd.read_csv(p) for p in paths))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds blood pressure, BMI and comorbidity features.

    Thresholds are fixed clinical values; nothing is learned from the data.
    - pulse_pressure         : SYSBP - DIABP
    - mean_arterial_pressure : DIABP + (SYSBP - DIABP) / 3
    - bp_stage               : ACC/AHA 2017: normal (<120 and <80) / elevated (120-129 and <80) /
                               stage1 (130-139 or 80-89) / stage2 (>=140 or >=90)
    - bmi_category           : WHO: underweight (<18.5) / normal (<25) / overweight (<30) / obese
    - comorbidity_count      : DIABETES + PREVHYP + PREVCHD (0-3)
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        sys_bp, dia_bp = X["SYSBP"], X["DIABP"]
        X["pulse_pressure"] = sys_bp - dia_bp
        X["mean_arterial_pressure"] = dia_bp + (sys_bp - dia_bp) / 3
        X["bp_stage"] = np.select(
            [(sys_bp >= 140) | (dia_bp >= 90), (sys_bp >= 130) | (dia_bp >= 80), sys_bp >= 120],
            ["stage2", "stage1", "elevated"],
            default="normal",
        )
        X["bmi_category"] = pd.cut(
            X["BMI"], bins=[-np.inf, 18.5, 25, 30, np.inf], right=False,
            labels=["underweight", "normal", "overweight", "obese"],
        ).astype(str)
        X["comorbidity_count"] = X["DIABETES"] + X["PREVHYP"] + X["PREVCHD"]
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(list(input_features) + FE_NUMERIC[:2] + FE_CATEGORICAL + FE_NUMERIC[2:],
                        dtype=object)


def _is_female(X):
    return (np.asarray(X) == 2).astype(int)


def _female_name(*_):
    return ["SEX_female"]


def build_encoder(feature_engineering=True):
    """StandardScaler on continuous columns and educ, SEX -> SEX_female (0/1),
    one-hot bp_stage / bmi_category, 0/1 columns unchanged. Output keeps column names."""
    numeric = CONTINUOUS + ORDINAL + (FE_NUMERIC if feature_engineering else [])
    steps = [
        ("num", StandardScaler(), numeric),
        ("sex", FunctionTransformer(_is_female, feature_names_out=_female_name), [SEX]),
    ]
    if feature_engineering:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FE_CATEGORICAL))
    steps.append(("bin", "passthrough", BINARY))
    encoder = ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)
    return encoder.set_output(transform="pandas")


def _preprocessing_steps(feature_engineering):
    steps = [("fe", FeatureEngineer())] if feature_engineering else []
    steps.append(("encode", build_encoder(feature_engineering)))
    return steps


def build_preprocessor(feature_engineering=True):
    """Feature engineering + encoding + scaling only, without SMOTE or a model."""
    return Pipeline(_preprocessing_steps(feature_engineering)).set_output(transform="pandas")


def scale_pos_weight(y):
    """XGBoost's counterpart of class_weight: negatives / positives."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def build_pipeline(estimator, balancing="smote", feature_engineering=True):
    """Builds fe -> encode -> balance -> model around a clone of estimator.

    The passed estimator object is never modified, so it can be reused.
    balancing : "smote"        -> SMOTE on the training data during fit only
                "class_weight" -> class_weight="balanced" (LR, SVM, RF; for XGBoost use
                                  balancing="none" and scale_pos_weight=scale_pos_weight(y_train))
                "none"         -> no balancing
    feature_engineering : adds the 5 FeatureEngineer columns
    """
    if balancing not in BALANCING_OPTIONS:
        raise ValueError(f"balancing şunlardan biri olmalı: {BALANCING_OPTIONS}, gelen: {balancing!r}")

    model = clone(estimator)
    steps = _preprocessing_steps(feature_engineering)

    if balancing == "smote":
        steps.append(("balance", SMOTE(random_state=RANDOM_STATE)))
    elif balancing == "class_weight":
        if "class_weight" not in model.get_params():
            raise ValueError(
                f"{type(model).__name__} modelinde class_weight parametresi yok. "
                "XGBoost için balancing='none' seçip modeli "
                "XGBClassifier(scale_pos_weight=scale_pos_weight(y_train)) ile kurun.")
        model.set_params(class_weight="balanced")

    steps.append(("model", model))
    return ImbPipeline(steps)
