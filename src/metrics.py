"""Shared model evaluation: cross validation, threshold choice, learning curve, test set, comparison.

    from metrics import cv_evaluate, threshold_analysis, test_evaluate
    result = cv_evaluate(pipe, X_train, y_train, model_name="Random Forest")

Works with any classifier or (imblearn) Pipeline. Models without predict_proba
(e.g. SVC()) are scored with sigmoid(decision_function): a risk score, not a probability.
"""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, learning_curve

from config import CLASS_NAMES, CV_FOLDS, MAIN_METRIC, RANDOM_STATE, RESULTS_FILE, TARGET_RECALL

METRICS = ["recall", "precision", "f1", "specificity", "roc_auc", "pr_auc", "accuracy"]

# GridSearchCV / RandomizedSearchCV: scoring=SCORING, refit=MAIN_METRIC
SCORING = {
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
    "specificity": make_scorer(recall_score, pos_label=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
}


def shared_cv():
    """The CV split everyone uses: stratified, CV_FOLDS folds, fixed seed."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def predict_score(model, X):
    """Score in [0, 1] for the stroke class. Without predict_proba, sigmoid(decision_function)
    is used; sigmoid(0) = 0.5, so threshold 0.5 matches the model's own predict()."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return expit(np.asarray(model.decision_function(X), dtype=float))


def compute_metrics(y, score, threshold=0.5):
    """All METRICS for one set of labels and scores; threshold applies to the threshold-based ones."""
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "accuracy": accuracy_score(y, pred),
    }


def _rows(X, index):
    return X.iloc[index] if hasattr(X, "iloc") else X[index]


def _final_params(model):
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    return str(estimator.get_params(deep=False))


def _steps(model):
    if hasattr(model, "steps"):
        return " -> ".join(type(step).__name__ for _, step in model.steps
                           if step not in (None, "passthrough"))
    return type(model).__name__


def cv_evaluate(model, X_train, y_train, model_name="Model", threshold=0.5, cv=None,
                save=True, results_file=RESULTS_FILE, note="", verbose=True):
    """Trains a fresh clone of the model on each shared CV fold and scores train and validation folds.

    The train score is computed on the original (not resampled) training fold; a large
    train - CV gap means overfitting.
    note : what this row answers, e.g. "SMOTE recall'u artırıyor mu?"
    save : appends one row to results_file.

    Returns a dict:
      "summary"   : per metric train_mean, cv_mean, cv_std, gap (train - cv)
      "folds"     : per-fold metrics
      "oof_score" : out-of-fold score of every training row (for threshold_analysis)
      "y"         : labels aligned with oof_score
    """
    cv = cv or shared_cv()
    y = np.asarray(y_train)
    oof = np.full(len(y), np.nan)
    records, fit_times = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y), start=1):
        m = clone(model)
        X_fit, X_val = _rows(X_train, train_idx), _rows(X_train, val_idx)

        start = time.perf_counter()
        m.fit(X_fit, y[train_idx])
        fit_times.append(time.perf_counter() - start)

        val_score = predict_score(m, X_val)
        oof[val_idx] = val_score
        for split, yy, ss in (("train", y[train_idx], predict_score(m, X_fit)),
                              ("cv", y[val_idx], val_score)):
            for metric, value in compute_metrics(yy, ss, threshold).items():
                records.append({"fold": fold, "split": split, "metric": metric, "value": value})

    folds = pd.DataFrame(records)
    grouped = folds.groupby(["metric", "split"])["value"]
    summary = pd.DataFrame({
        "train_mean": grouped.mean().xs("train", level="split"),
        "cv_mean": grouped.mean().xs("cv", level="split"),
        "cv_std": grouped.std().xs("cv", level="split"),
    }).reindex(METRICS)
    summary["gap"] = summary["train_mean"] - summary["cv_mean"]

    if verbose:
        print(f"===== {model_name}: {cv.get_n_splits()} katlı CV (eşik = {threshold}) =====")
        print(f"Kat başına ortalama eğitim süresi: {np.mean(fit_times):.2f} sn\n")
        print(summary.round(3).to_string())
        gap = summary.loc[MAIN_METRIC, "gap"]
        if gap > 0.10:
            print(f"\nUyarı: {MAIN_METRIC} için eğitim - CV farkı {gap:.3f}. Overfitting olabilir.")

    if save:
        row = {"model_name": model_name,
               "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "note": note, "threshold": threshold,
               "train_size": len(y), "stroke_rate": round(y.mean(), 4)}
        for metric in METRICS:
            row[f"{metric}_cv"] = summary.loc[metric, "cv_mean"]
            row[f"{metric}_std"] = summary.loc[metric, "cv_std"]
            row[f"{metric}_train"] = summary.loc[metric, "train_mean"]
        row["fit_time_s"] = float(np.mean(fit_times))
        row["steps"] = _steps(model)
        row["params"] = _final_params(model)

        table = pd.DataFrame([row])
        path = Path(results_file)
        if path.exists():
            table = pd.concat([pd.read_csv(path), table], ignore_index=True)
        table.to_csv(path, index=False)
        if verbose:
            print(f"\nSonuç '{results_file}' dosyasına eklendi.")

    return {"model_name": model_name, "summary": summary, "folds": folds,
            "oof_score": oof, "y": y, "fit_time_s": float(np.mean(fit_times))}


def threshold_analysis(y, score, target_recall=TARGET_RECALL, step=0.05, model_name="Model",
                       plot=True):
    """Caught cases and false alarms per threshold, and the threshold to use.

    y, score : result["y"] and result["oof_score"] from cv_evaluate, so the threshold is
               chosen without looking at the test set.
    Chooses the highest threshold whose recall is at least target_recall (= that recall
    with the fewest false alarms).

    Returns (table, chosen_threshold).
    """
    y = np.asarray(y)
    score = np.asarray(score)

    def row(threshold):
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"threshold": threshold,
                "caught": tp, "missed": fn, "false_alarms": fp,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "false_alarms_per_1000": fp / len(y) * 1000}

    thresholds = np.round(np.arange(step, 1.0, step), 4)
    table = pd.DataFrame([row(t) for t in thresholds])

    _, recalls, pr_thresholds = precision_recall_curve(y, score)
    ok = np.where(recalls[:-1] >= target_recall)[0]
    chosen = float(pr_thresholds[ok.max()]) if len(ok) else float(pr_thresholds.min())
    c = row(chosen)

    print(f"===== {model_name}: Eşik analizi (toplam {int(y.sum())} inme, {len(y)} kişi) =====")
    print(table.round(3).to_string(index=False))
    print(f"\nRecall >= {target_recall:.0%} için seçilen eşik: {chosen:.3f}")
    print(f"  {c['caught']}/{c['caught'] + c['missed']} inme yakalanır "
          f"(recall {c['recall']:.1%}), precision {c['precision']:.1%}, "
          f"1000 kişide {c['false_alarms_per_1000']:.0f} yanlış alarm.")
    d = row(0.5)
    print(f"  Karşılaştırma, eşik 0.5: recall {d['recall']:.1%}, "
          f"1000 kişide {d['false_alarms_per_1000']:.0f} yanlış alarm.")

    if plot:
        fine = pd.DataFrame([row(t) for t in np.linspace(0.01, 0.99, 99)])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for metric in ("recall", "precision", "specificity"):
            ax.plot(fine["threshold"], fine[metric], label=metric)
        ax.axvline(chosen, color="gray", linestyle="--", label=f"seçilen eşik ({chosen:.2f})")
        ax.axvline(0.5, color="lightgray", linestyle=":", label="varsayılan eşik (0.5)")
        ax.set_xlabel("Eşik")
        ax.set_ylabel("Değer")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{model_name}: Eşiğe göre recall / precision / specificity")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return table, chosen


def test_evaluate(model, X_test, y_test, threshold=0.5, model_name="Model", cv_result=None,
                  plot=True):
    """Evaluates the final model (fitted on all training data) on the test set, once.

    threshold : the one chosen by threshold_analysis
    cv_result : cv_evaluate output; CV (out-of-fold, same threshold) and test are shown side by side.

    Returns {"metrics", "confusion_matrix", "score"}.
    """
    y = np.asarray(y_test)
    score = predict_score(model, X_test)
    metrics = compute_metrics(y, score, threshold)
    cm = confusion_matrix(y, (score >= threshold).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"===== {model_name}: Test seti (eşik = {threshold:.3f}) =====")
    print(f"{tp}/{tp + fn} inme yakalandı, {fn} inme kaçırıldı, {fp} yanlış alarm "
          f"({len(y)} kişi).\n")
    table = pd.DataFrame({"test": metrics})
    if cv_result is not None:
        table.insert(0, "cv_same_threshold",
                     pd.Series(compute_metrics(cv_result["y"], cv_result["oof_score"], threshold)))
        table["cv_minus_test"] = table["cv_same_threshold"] - table["test"]
    print(table.round(3).to_string())

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
        axes[0].set_title(f"Karmaşıklık matrisi (eşik {threshold:.2f})")
        axes[0].set_xlabel("Tahmin")
        axes[0].set_ylabel("Gerçek")

        fpr, tpr, _ = roc_curve(y, score)
        axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
        axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="rastgele")
        axes[1].set_xlabel("Yanlış alarm oranı (1 - specificity)")
        axes[1].set_ylabel("Recall")
        axes[1].set_title("ROC eğrisi")
        axes[1].legend()

        p, r, _ = precision_recall_curve(y, score)
        axes[2].plot(r, p, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
        axes[2].axhline(y.mean(), color="gray", linestyle="--",
                        label=f"rastgele (inme oranı {y.mean():.2f})")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall eğrisi")
        axes[2].legend()

        plt.suptitle(f"{model_name}: Test seti")
        plt.tight_layout()
        plt.show()

    return {"metrics": metrics, "confusion_matrix": cm, "score": score}


def compare_models(results_file=RESULTS_FILE, metrics=None, latest_only=True, plot=True):
    """Compares the models recorded by cv_evaluate.

    latest_only : keep only the latest row per model_name
    Returns CV means per model and metric, sorted by MAIN_METRIC.
    """
    metrics = metrics or METRICS
    df = pd.read_csv(results_file)
    if latest_only:
        df = df.drop_duplicates("model_name", keep="last")
    df = df.set_index("model_name").sort_values(f"{MAIN_METRIC}_cv", ascending=False)

    mean = df[[f"{m}_cv" for m in metrics]].set_axis(metrics, axis=1)
    std = df[[f"{m}_std" for m in metrics]].set_axis(metrics, axis=1)
    view = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    view[f"{MAIN_METRIC}_train_cv_gap"] = (df[f"{MAIN_METRIC}_train"] - df[f"{MAIN_METRIC}_cv"]).round(3)
    view["fit_time_s"] = df["fit_time_s"].round(2)
    print(f"===== Model karşılaştırması (CV ortalaması ± std, {MAIN_METRIC}'e göre sıralı) =====")
    print(view.to_string())

    if plot:
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(metrics)), 4.5))
        mean.T.plot(kind="bar", yerr=std.T, ax=ax, capsize=3, rot=0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("CV ortalaması")
        ax.set_title("Modellerin metrik karşılaştırması (hata çubuğu: katlar arası std)")
        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        pd.DataFrame({"Eğitim": df[f"{MAIN_METRIC}_train"],
                      "CV": df[f"{MAIN_METRIC}_cv"]}).plot(kind="bar", ax=ax, rot=0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(MAIN_METRIC)
        ax.set_title(f"Overfitting kontrolü: eğitim ve CV {MAIN_METRIC}")
        plt.tight_layout()
        plt.show()

    return mean


def plot_curves(results, title="CV tahminleri"):
    """ROC and PR curves of several models on one figure.

    results : {model_name: (y, score)}
      CV  : {"RF": (rf_cv["y"], rf_cv["oof_score"]), ...}
      test: {"RF": (y_test, rf_test["score"]), ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (y, score) in results.items():
        y = np.asarray(y)
        fpr, tpr, _ = roc_curve(y, score)
        p, r, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, score):.3f})")
        axes[1].plot(r, p, label=f"{name} ({average_precision_score(y, score):.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].axhline(y.mean(), color="gray", linestyle="--", label="rastgele")
    axes[0].set(xlabel="Yanlış alarm oranı (1 - specificity)", ylabel="Recall",
                title="ROC eğrileri (ROC-AUC)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR eğrileri (PR-AUC)")
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def learning_curve_analysis(model, X_train, y_train, model_name="Model", metric=MAIN_METRIC,
                            train_sizes=np.linspace(0.1, 1.0, 6), plot=True):
    """Train and CV score as the training set grows, on the shared CV folds.

    A large gap that does not close: overfitting. Both low and close: underfitting.
    """
    sizes, train_scores, cv_scores = learning_curve(
        model, X_train, y_train, cv=shared_cv(), scoring=SCORING[metric],
        train_sizes=train_sizes, shuffle=True, random_state=RANDOM_STATE, n_jobs=-1)
    table = pd.DataFrame({"train_size": sizes,
                          "train_mean": train_scores.mean(axis=1),
                          "cv_mean": cv_scores.mean(axis=1),
                          "cv_std": cv_scores.std(axis=1)})

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, scores in (("Eğitim", train_scores), ("CV", cv_scores)):
            mean, std = scores.mean(axis=1), scores.std(axis=1)
            ax.plot(sizes, mean, marker="o", label=label)
            ax.fill_between(sizes, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Eğitim örneği sayısı")
        ax.set_ylabel(metric)
        ax.set_title(f"{model_name}: Öğrenme eğrisi ({metric})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return table
