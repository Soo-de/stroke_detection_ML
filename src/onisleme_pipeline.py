"""
onisleme_pipeline.py
Ortak ön işleme pipeline'ı: Encoding, Ölçekleme, Feature Engineering ve SMOTE - İnme Tespiti Projesi (Framingham)

Kişi 1'in hazırladığı train.csv / test.csv verisini alır; feature engineering,
encoding, ölçekleme ve dengeleme (SMOTE) adımlarını model ile tek bir Pipeline'da birleştirir.

Herkes kendi modelini şu şekilde kurar:

    import pandas as pd
    from onisleme_pipeline import build_pipeline, veriyi_ayir

    X_train, X_test, y_train, y_test = veriyi_ayir(pd.read_csv("train.csv"), pd.read_csv("test.csv"))
    pipe = build_pipeline(LogisticRegression(max_iter=3000), balancing="smote")
    pipe.fit(X_train, y_train)
    pipe.predict_proba(X_test)[:, 1]

    # GridSearch'te model parametreleri "model__" ön ekiyle yazılır:
    # GridSearchCV(pipe, {"model__C": [0.1, 1, 10]}, ...)

Pipeline adımları:
    fe      : feature engineering (pulse_pressure, mean_arterial_pressure, bp_stage,
              bmi_category, comorbidity_count)
    encode  : SEX → 0/1, kategoriler → one-hot, sayısallar → StandardScaler
    balance : SMOTE (sadece balancing="smote" ise)
    model   : sizin modeliniz

VERİ SIZINTISI: Tüm adımlar Pipeline içinde olduğu için ölçekleme parametreleri ve SMOTE
yalnızca eğitim verisinden öğrenilir. SMOTE yalnızca fit sırasında çalışır; predict /
predict_proba sırasında ve cross validation'ın validation katında devreye girmez.

Gereksinimler: pandas, numpy, scikit-learn, imbalanced-learn
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
HEDEF = "STROKE"

# Kişi 1'in teslim ettiği özellikler (sızıntılı sütunlar zaten çıkarılmış, eksikler doldurulmuş)
SUREKLI = ["AGE", "TOTCHOL", "SYSBP", "DIABP", "BMI", "HEARTRTE", "GLUCOSE", "CIGPDAY"]
IKILI = ["CURSMOKE", "DIABETES", "BPMEDS", "PREVCHD", "PREVAP", "PREVMI", "PREVHYP"]
CINSIYET = "SEX"          # 1 = erkek, 2 = kadın
SIRALI = ["educ"]         # eğitim düzeyi 1-4
OZELLIKLER = SUREKLI + IKILI + [CINSIYET] + SIRALI

# Feature engineering ile eklenen sütunlar
FE_SAYISAL = ["pulse_pressure", "mean_arterial_pressure", "comorbidity_count"]
FE_KATEGORIK = ["bp_stage", "bmi_category"]

DENGELEME_SECENEKLERI = ("smote", "class_weight", "none")


# ----------------------------------------------------------------------------
# Veri
# ----------------------------------------------------------------------------
def veriyi_ayir(train: pd.DataFrame, test: pd.DataFrame):
    """Kişi 1'in train/test tablolarından X_train, X_test, y_train, y_test döndürür."""
    eksik = [c for c in OZELLIKLER + [HEDEF] if c not in train.columns]
    if eksik:
        raise ValueError(f"train verisinde beklenen sütunlar yok: {eksik}")
    return (train[OZELLIKLER], test[OZELLIKLER],
            train[HEDEF].astype(int), test[HEDEF].astype(int))


# ----------------------------------------------------------------------------
# 1) Feature engineering
# ----------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Tansiyon, kilo ve eşlik eden hastalık bilgilerinden yeni özellikler türetir.

    Eşikler sabit tıbbi değerlerdir, veriden öğrenilmez (sızıntı riski yoktur).

    - pulse_pressure          : nabız basıncı = SYSBP - DIABP (damar sertliği göstergesi)
    - mean_arterial_pressure  : ortalama arter basıncı = DIABP + (SYSBP - DIABP) / 3
    - bp_stage                : ACC/AHA 2017 tansiyon evresi
                                normal (<120 ve <80) / yuksek_normal (120-129 ve <80) /
                                evre1 (130-139 veya 80-89) / evre2 (>=140 veya >=90)
    - bmi_category            : WHO: zayif (<18.5) / normal (<25) / fazla_kilolu (<30) / obez (30+)
    - comorbidity_count       : eşlik eden hastalık sayısı = DIABETES + PREVHYP + PREVCHD (0-3)
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
            [(sys_bp >= 140) | (dia_bp >= 90),
             (sys_bp >= 130) | (dia_bp >= 80),
             (sys_bp >= 120)],
            ["evre2", "evre1", "yuksek_normal"],
            default="normal",
        )
        X["bmi_category"] = pd.cut(
            X["BMI"], bins=[-np.inf, 18.5, 25, 30, np.inf], right=False,
            labels=["zayif", "normal", "fazla_kilolu", "obez"],
        ).astype(str)
        X["comorbidity_count"] = X["DIABETES"] + X["PREVHYP"] + X["PREVCHD"]
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(list(input_features) + FE_SAYISAL[:2] + FE_KATEGORIK + FE_SAYISAL[2:], dtype=object)


# ----------------------------------------------------------------------------
# 2) Encoding + ölçekleme
# ----------------------------------------------------------------------------
def _kadin_mi(X):
    """SEX: 1 = erkek, 2 = kadın  →  SEX_kadin: 0 = erkek, 1 = kadın"""
    return (np.asarray(X) == 2).astype(int)


def _kadin_isim(*_):
    return ["SEX_kadin"]


def build_encoder(feature_engineering: bool = True) -> ColumnTransformer:
    """- Sürekli sayısallar + educ : StandardScaler (ortalama 0, standart sapma 1)
       - SEX                       : 1/2 → 0/1 (SEX_kadin)
       - bp_stage, bmi_category    : one-hot
       - Zaten 0/1 olan sütunlar   : olduğu gibi geçer
    """
    sayisal = SUREKLI + SIRALI + (FE_SAYISAL if feature_engineering else [])
    adimlar = [
        ("num", StandardScaler(), sayisal),
        ("sex", FunctionTransformer(_kadin_mi, feature_names_out=_kadin_isim), [CINSIYET]),
    ]
    if feature_engineering:
        adimlar.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FE_KATEGORIK))
    adimlar.append(("bin", "passthrough", IKILI))

    encoder = ColumnTransformer(adimlar, remainder="drop", verbose_feature_names_out=False)
    # Pandas çıktısı: sütun isimleri korunur (SHAP grafiklerinde "AGE", "SYSBP" gibi isimler görünür)
    return encoder.set_output(transform="pandas")


def _onisleme_adimlari(feature_engineering: bool = True) -> list:
    adimlar = [("fe", FeatureEngineer())] if feature_engineering else []
    adimlar.append(("encode", build_encoder(feature_engineering)))
    return adimlar


def build_preprocessor(feature_engineering: bool = True) -> Pipeline:
    """Sadece ön işleme (FE + encoding + ölçekleme). Model içermez."""
    return Pipeline(_onisleme_adimlari(feature_engineering)).set_output(transform="pandas")


# ----------------------------------------------------------------------------
# 3) Tam pipeline: ön işleme + dengeleme + model
# ----------------------------------------------------------------------------
def scale_pos_weight(y) -> float:
    """XGBoost için class_weight karşılığı: negatif örnek sayısı / pozitif örnek sayısı."""
    return float((y == 0).sum() / (y == 1).sum())


def build_pipeline(estimator, balancing: str = "smote", feature_engineering: bool = True) -> ImbPipeline:
    """Herkesin modelini kuracağı fonksiyon.

    estimator           : sklearn modeli (LogisticRegression(), SVC(probability=True), ...)
    balancing           : "smote"        → SMOTE ile sentetik inme örnekleri (varsayılan)
                          "class_weight" → model inme hatalarını daha ağır cezalandırır
                                           (LR, SVM, RF; XGBoost'ta yok, bkz. scale_pos_weight)
                          "none"         → dengeleme yok
    feature_engineering : True ise 5 tıbbi özellik eklenir
    """
    if balancing not in DENGELEME_SECENEKLERI:
        raise ValueError(f"balancing şunlardan biri olmalı: {DENGELEME_SECENEKLERI}, gelen: {balancing!r}")

    adimlar = _onisleme_adimlari(feature_engineering)

    if balancing == "smote":
        adimlar.append(("balance", SMOTE(random_state=RANDOM_STATE)))
    elif balancing == "class_weight":
        if "class_weight" in estimator.get_params():
            estimator.set_params(class_weight="balanced")
        else:
            raise ValueError(
                f"{type(estimator).__name__} modelinde class_weight parametresi yok. "
                "XGBoost için balancing='none' seçip modeli "
                "XGBClassifier(scale_pos_weight=scale_pos_weight(y_train)) ile kurun."
            )

    adimlar.append(("model", estimator))
    return ImbPipeline(adimlar)
