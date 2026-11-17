"""Shared explainability: SHAP (global) and LIME (local, one patient).

    from explain import explain, lime_cases
    result = explain(pipe, X_train, X_test, model_name="Random Forest",
                     save_prefix=FIG_DIR / MODEL_CODE)
    cases = lime_cases(pipe, X_train, X_test, y_test, threshold, model_name="Random Forest",
                       save_prefix=FIG_DIR / MODEL_CODE)

The score shown by LIME is the model's risk score (metrics.predict_score). It is not a
calibrated probability: SMOTE / class_weight shift it upwards and SVC() has no probability at all.

model is a fitted classifier or Pipeline (e.g. from build_pipeline).
X_train / X_test are the raw feature tables from load_data(). SHAP runs on the pipeline's
transformed columns (SEX_female, bp_stage_stage2, ...); LIME runs the whole pipeline on raw
columns. Models without predict_proba (e.g. SVC()) are explained through
metrics.predict_score, the same score the evaluation functions use.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline

from config import CLASS_NAMES, RANDOM_STATE
from metrics import predict_score


def _save(save_prefix, name):
    """Saves the current figure as <save_prefix>_<name>.png when save_prefix is given."""
    if save_prefix is not None:
        plt.gcf().savefig(f"{save_prefix}_{name}.png", dpi=200, bbox_inches="tight")

ü
TREE_MODELS = ("RandomForest", "ExtraTrees", "DecisionTree", "GradientBoosting", "XGB", "LGBM",
               "CatBoost")
LINEAR_MODELS = ("LogisticRegression", "LinearSVC", "RidgeClassifier", "SGDClassifier")


def _split_pipeline(model, X):
    """Final estimator and X passed through the fitted preprocessing steps (samplers skipped)."""
    model = getattr(model, "best_estimator_", model)  # GridSearchCV / RandomizedSearchCV -> best model
    if not isinstance(model, Pipeline):
        return model, X
    index = X.index
    last = None
    for _, step in model.steps[:-1]:
        if step is None or step == "passthrough" or hasattr(step, "fit_resample"):
            continue
        X = step.transform(X)
        last = step
    if not isinstance(X, pd.DataFrame):
        values = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
        try:
            names = list(last.get_feature_names_out())
        except (AttributeError, ValueError):
            names = [f"feature_{i}" for i in range(values.shape[1])]
        X = pd.DataFrame(values, columns=names, index=index)
    return model.steps[-1][1], X.astype(float)


def _explainer_kind(estimator):
    name = type(estimator).__name__
    if any(k in name for k in TREE_MODELS):
        return "tree"
    if any(k in name for k in LINEAR_MODELS):
        return "linear"
    return "kernel"


def _positive_class(shap_values):
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def shap_global(model, X_train, X_test, model_name="Model", sample_size=150, background_size=50,
                kernel_nsamples="auto", plot=True, save_prefix=None, random_state=RANDOM_STATE):
    """Which features the model relies on overall (mean |SHAP| on the test set).

    Explainer by model type: trees (RF, XGBoost) -> TreeExplainer, linear (LR) ->
    LinearExplainer, others (KNN, SVM) -> KernelExplainer. KernelExplainer is slow, so it
    explains sample_size test rows against background_size k-means centres of X_train.
    kernel_nsamples: feature coalitions KernelExplainer evaluates per row ("auto" = 2 * features
    + 2048). Runtime grows with sample_size * kernel_nsamples * background_size; for KNN/SVM
    e.g. sample_size=100, background_size=20, kernel_nsamples=300 runs in about a minute.
    save_prefix: e.g. FIG_DIR / "rf" -> rf_shap_bar.png, rf_shap_summary.png

    Returns a table of features sorted by mean |SHAP|.
    """
    estimator, Xtr = _split_pipeline(model, X_train)
    _, Xte = _split_pipeline(model, X_test)
    kind = _explainer_kind(estimator)

    if kind == "tree":
        values = _positive_class(shap.TreeExplainer(estimator).shap_values(Xte))
    elif kind == "linear":
        values = _positive_class(shap.LinearExplainer(estimator, Xtr).shap_values(Xte))
    else:
        Xte = Xte.sample(n=min(sample_size, len(Xte)), random_state=random_state)
        background = shap.kmeans(Xtr, min(background_size, len(Xtr)))
        columns = Xtr.columns
        explainer = shap.KernelExplainer(
            lambda a: predict_score(estimator, pd.DataFrame(a, columns=columns)), background)
        values = np.asarray(explainer.shap_values(Xte, nsamples=kernel_nsamples, silent=True))

    importance = (pd.DataFrame({"feature": Xte.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False)
                  .reset_index(drop=True))

    if plot:
        plt.figure()
        shap.summary_plot(values, Xte, plot_type="bar", show=False)
        plt.title(f"{model_name}: SHAP Özellik Önemi")
        plt.tight_layout()
        _save(save_prefix, "shap_bar")
        plt.show()

        plt.figure()
        shap.summary_plot(values, Xte, show=False)
        plt.title(f"{model_name}: SHAP Özet Grafiği (İnme Var sınıfı)")
        plt.tight_layout()
        _save(save_prefix, "shap_summary")
        plt.show()

    return importance


def lime_local(model, X_train, X_test, index=None, model_name="Model", num_features=8, plot=True,
               save_prefix=None, save_name="lime", title=None, random_state=RANDOM_STATE):
    """Explains the model's decision for one test patient.

    index : row position in X_test; by default the patient with the highest stroke score.
    LIME runs the whole pipeline on raw X, so it works the same way for every model type.

    Returns a list of (condition, weight).
    """
    columns = list(X_train.columns)

    def predict_fn(a):
        score = predict_score(model, pd.DataFrame(a, columns=columns))
        return np.column_stack([1 - score, score])

    if index is None:
        index = int(np.argmax(predict_score(model, X_test)))

    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=columns,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )
    patient = X_test.iloc[index].to_numpy(dtype=float)
    explanation = explainer.explain_instance(patient, predict_fn, num_features=num_features,
                                             labels=(1,))

    risk = predict_fn(patient.reshape(1, -1))[0, 1]
    # Not a calibrated probability (SMOTE / class_weight / SVC decision_function), so not shown as %.
    print(f"[{model_name}] Test hastası #{index}: model risk skoru = {risk:.3f}")

    if plot:
        explanation.as_pyplot_figure(label=1)
        plt.title(title or f"{model_name}: LIME Açıklaması (hasta #{index})")
        plt.tight_layout()
        _save(save_prefix, save_name)
        plt.show()

    return explanation.as_list(label=1)


def explain(model, X_train, X_test, model_name="Model", lime_index=None, plot=True, save_prefix=None,
            **shap_kwargs):
    """SHAP (global) + LIME (local) in one call; shap_kwargs (sample_size, ...) go to shap_global.

    Returns {"shap_importance": table, "lime": list}.
    """
    print(f"===== {model_name}: XAI Analizi =====")
    importance = shap_global(model, X_train, X_test, model_name=model_name, plot=plot,
                             save_prefix=save_prefix, **shap_kwargs)
    print("\nEn önemli 5 özellik (SHAP):")
    print(importance.head(5).to_string(index=False))
    print()
    lime_result = lime_local(model, X_train, X_test, index=lime_index, model_name=model_name,
                             plot=plot, save_prefix=save_prefix)
    return {"shap_importance": importance, "lime": lime_result}


def lime_cases(model, X_train, X_test, y_test, threshold, model_name="Model", plot=True,
               save_prefix=None):
    """LIME for two real stroke cases at the chosen threshold:
    - caught : stroke patient with the highest risk score (score >= threshold)
    - missed : stroke patient with the lowest risk score (score < threshold)

    model must be fitted (best.fit(X_train, y_train) before test_evaluate).
    Saves <save_prefix>_lime_caught.png and <save_prefix>_lime_missed.png.
    Returns {"caught": (index, lime list) or None, "missed": (index, lime list) or None}.
    """
    model = getattr(model, "best_estimator_", model)
    score = predict_score(model, X_test)
    y = np.asarray(y_test)
    caught = np.where((y == 1) & (score >= threshold))[0]
    missed = np.where((y == 1) & (score < threshold))[0]
    print(f"[{model_name}] Eşik {threshold:.3f}: {len(caught)} inme yakalandı, {len(missed)} kaçırıldı.")

    result = {"caught": None, "missed": None}
    for key, rows, pick, label in (("caught", caught, np.argmax, "yakalanan vaka"),
                                   ("missed", missed, np.argmin, "kaçırılan vaka")):
        if len(rows) == 0:
            print(f"  {label}: yok, atlandı.")
            continue
        index = int(rows[pick(score[rows])])
        lime = lime_local(model, X_train, X_test, index=index, model_name=model_name, plot=plot,
                          save_prefix=save_prefix, save_name=f"lime_{key}",
                          title=f"{model_name}: LIME, {label} (hasta #{index})")
        result[key] = (index, lime)
    return result


def compare_importance(results):
    """Side-by-side SHAP importance shares from {model_name: explain() output}.

    Each model's mean |SHAP| is divided by its sum, so models on different SHAP scales
    (log-odds vs probability) can be compared.
    """
    table = {}
    for name, result in results.items():
        imp = result["shap_importance"].set_index("feature")["mean_abs_shap"]
        table[name] = imp / imp.sum()
    df = pd.DataFrame(table).fillna(0)
    return df.sort_values(df.columns[0], ascending=False)
