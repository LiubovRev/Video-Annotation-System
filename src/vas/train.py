"""Step 5 — train and evaluate frame-level behaviour classifiers.

Evaluation is designed to avoid temporal leakage between near-identical
neighbouring frames:

* ``session`` split — whole recording sessions are held out (preferred,
  measures generalisation to new sessions);
* ``time`` split — the last ``test_size`` of every session is held out and
  frames within ``purge_gap_s`` of the boundary are dropped.

Hyper-parameter search and early stopping use the same grouping, never a
random frame-level split. Feature selection is fitted on the training
split only. The selected model, label names, feature list and settings
are saved together in one bundle so prediction cannot drift from training.
"""

from __future__ import annotations

import inspect
import logging
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .config import model_bundle_path, resolve_paths
from .features import build_features
from .utils import write_json

logger = logging.getLogger(__name__)
BUNDLE_VERSION = 2


# =============================================================================
# Data preparation
# =============================================================================
def filter_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mt = cfg.get("model_training", {})
    l2c = mt.get("label_to_class") or {}
    non_posture = set(mt.get("non_posture_classes") or [])
    min_n = int(mt.get("min_samples_per_class", 30))

    df = df[df["annotation_label"].notna()].copy()
    df["label_class"] = df["annotation_label"].map(l2c)
    df = df[~df["label_class"].isin(non_posture)]
    counts = df["annotation_label"].value_counts()
    rare = counts[counts < min_n]
    if len(rare):
        logger.info("Dropping rare codes (< %d frames): %s", min_n, {k: int(v) for k, v in rare.items()})
    df = df[df["annotation_label"].isin(counts[counts >= min_n].index)]
    if df["annotation_label"].nunique() < 2:
        raise ValueError("Need at least two annotation codes with enough frames to train a classifier.")
    return df.reset_index(drop=True)


# =============================================================================
# Splitting
# =============================================================================
def _choose_strategy(df: pd.DataFrame, cfg: dict) -> str:
    s = cfg.get("model_training", {}).get("split_strategy", "auto")
    n_sessions = df["session"].nunique()
    if s == "auto":
        return "session" if n_sessions >= 3 else "time"
    if s == "session" and n_sessions < 2:
        logger.warning("split_strategy=session needs ≥ 2 sessions; falling back to time split.")
        return "time"
    return s


def session_split(df: pd.DataFrame, test_size: float, seed: int, tries: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole sessions; pick the split whose train set covers most classes."""
    y, groups = df["annotation_label"].to_numpy(), df["session"].to_numpy()
    all_classes = set(y)
    best, best_score = None, -1.0
    gss = GroupShuffleSplit(n_splits=tries, test_size=test_size, random_state=seed)
    for tr, te in gss.split(df, y, groups):
        cov_train = len(set(y[tr])) / len(all_classes)
        cov_test = len(set(y[te]) & set(y[tr])) / len(all_classes)
        score = 2 * cov_train + cov_test
        if score > best_score:
            best, best_score = (tr, te), score
    return best


def time_split(df: pd.DataFrame, test_size: float, purge_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Last ``test_size`` of each session (by time) is the test set."""
    t = df["time_s"].to_numpy()
    cut = df.groupby("session")["time_s"].transform(lambda s: s.quantile(1 - test_size)).to_numpy()
    tr = np.flatnonzero(t < cut - purge_gap_s)
    te = np.flatnonzero(t > cut + purge_gap_s)
    return tr, te


def cv_groups(df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Groups for inner CV: sessions if there are enough, else contiguous time blocks."""
    if df["session"].nunique() >= n_folds:
        return df["session"].astype("category").cat.codes.to_numpy()
    rank = df.groupby("session")["time_s"].rank(pct=True).to_numpy()
    block = np.minimum((rank * n_folds).astype(int), n_folds - 1)
    return (df["session"].astype("category").cat.codes.to_numpy() * n_folds + block)


def time_tail_mask(df: pd.DataFrame, frac: float = 0.15) -> np.ndarray:
    """True for the last ``frac`` of each session — used as early-stopping set."""
    rank = df.groupby("session")["time_s"].rank(pct=True).to_numpy()
    return rank > 1 - frac


# =============================================================================
# Feature selection (fitted on train only)
# =============================================================================
def select_features(X: pd.DataFrame, threshold: float, seed: int, max_rows: int = 20000) -> tuple[list[str], dict]:
    var = X.var(numeric_only=True)
    zero_var = var[(var < 1e-10) | var.isna()].index.tolist()
    Xs = X.drop(columns=zero_var)
    sample = Xs.sample(min(len(Xs), max_rows), random_state=seed) if len(Xs) > max_rows else Xs
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    redundant = [c for c in upper.columns if (upper[c] > threshold).any()]
    keep = [c for c in Xs.columns if c not in redundant]
    return keep, {"zero_variance": zero_var, "redundant": redundant}


# =============================================================================
# Models
# =============================================================================
def _fit_lightgbm(X, y, w, val_mask, seed, n_classes):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=1.0, class_weight="balanced", random_state=seed,
        n_jobs=-1, verbose=-1,
    )
    if val_mask.any() and (~val_mask).any() and len(np.unique(y[~val_mask])) == n_classes:
        params = inspect.signature(lgb.LGBMClassifier.fit).parameters
        eval_kw = ({"eval_X": (X[val_mask],), "eval_y": (y[val_mask],)} if "eval_X" in params
                   else {"eval_set": [(X[val_mask], y[val_mask])]})
        model.fit(X[~val_mask], y[~val_mask], callbacks=[lgb.early_stopping(50, verbose=False)], **eval_kw)
        best = model.best_iteration_ or model.n_estimators
        model.set_params(n_estimators=max(best, 10))
    model.fit(X, y)
    return model


def _fit_xgboost(X, y, w, groups, seed, n_iter, n_folds):
    from xgboost import XGBClassifier

    search = RandomizedSearchCV(
        XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="mlogloss", tree_method="hist"),
        param_distributions={
            "n_estimators": [100, 200, 300, 500], "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.5, 0.7, 0.9], "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.3], "reg_alpha": [0, 0.5, 1.0], "reg_lambda": [0.5, 1.0, 2.0],
        },
        n_iter=n_iter, cv=GroupKFold(n_splits=n_folds), scoring="f1_macro",
        n_jobs=1, random_state=seed, error_score=np.nan,
    )
    search.fit(X, y, groups=groups, sample_weight=w)
    logger.info("XGBoost best CV f1_macro=%.4f params=%s", search.best_score_, search.best_params_)
    return search.best_estimator_


def _fit_hist(X, y, w, val_mask, seed):
    supports_val = "X_val" in inspect.signature(HistGradientBoostingClassifier.fit).parameters
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
        early_stopping=True, n_iter_no_change=20,
        **({} if supports_val else {"validation_fraction": 0.15}),
    )
    if supports_val and val_mask.any() and (~val_mask).any():
        model.fit(X[~val_mask], y[~val_mask], X_val=X[val_mask], y_val=y[val_mask])
    else:
        model.fit(X, y)
    return model


# =============================================================================
# Evaluation
# =============================================================================
def compute_metrics(y_true, y_pred, proba=None, labels=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if proba is not None and labels is not None:
        from sklearn.metrics import roc_auc_score

        present = np.unique(y_true)
        if len(present) > 1:
            try:
                cols = [list(labels).index(c) for c in present]
                p = proba[:, cols] / np.clip(proba[:, cols].sum(axis=1, keepdims=True), 1e-12, None)
                m["roc_auc_ovr_macro"] = roc_auc_score(y_true, p, multi_class="ovr", labels=present)
            except ValueError:
                pass
    return {k: float(v) for k, v in m.items()}


def overfit_status(gap: float) -> str:
    return "severe" if gap > 0.15 else "moderate" if gap > 0.10 else "ok"


def plot_model_comparison(test_m: dict, train_m: dict, metric: str, path: Path, dpi: int) -> None:
    keys = ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.DataFrame(test_m).T[keys].plot(kind="bar", ax=axes[0], edgecolor="black", rot=0)
    axes[0].set(title="Held-out performance", ylim=(0, 1.05), ylabel="score")
    axes[0].grid(alpha=0.3, axis="y")
    gap = pd.Series({k: train_m[k][metric] - test_m[k][metric] for k in test_m})
    gap.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black", rot=0)
    axes[1].axhline(0.10, color="orange", ls="--", label="moderate (0.10)")
    axes[1].axhline(0.15, color="red", ls="--", label="severe (0.15)")
    axes[1].set(title=f"Train − test gap ({metric})", ylabel="gap")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_feature_importance(models: dict, features: list[str], path: Path, dpi: int) -> None:
    capable = {n: m for n, m in models.items() if hasattr(m, "feature_importances_")}
    if not capable:
        return
    fig, axes = plt.subplots(1, len(capable), figsize=(6 * len(capable), 6), squeeze=False)
    for ax, (name, m) in zip(axes[0], capable.items()):
        imp = np.asarray(m.feature_importances_, dtype=float)
        idx = np.argsort(imp)[-20:]
        ax.barh(range(len(idx)), imp[idx], color="steelblue")
        ax.set_yticks(range(len(idx)), [features[i] for i in idx], fontsize=8)
        ax.set_title(f"{name} — top 20")
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels, path: Path, dpi: int, title: str) -> None:
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 4, 0.8 * len(labels) + 3))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# =============================================================================
# Main entry
# =============================================================================
def train_model(labeled: pd.DataFrame, cfg: dict, output_dir: Path | None = None) -> dict:
    """Train, evaluate, and save the model bundle. Returns a summary dict."""
    mt = cfg.get("model_training", {})
    seed = int(mt.get("random_state", 42))
    test_size = float(mt.get("test_size", 0.2))
    metric = mt.get("selection_metric", "f1_macro")
    n_folds = int(mt.get("cv_folds", 3))
    dpi = cfg.get("plotting", {}).get("dpi", 150)
    output_dir = Path(output_dir or resolve_paths(cfg).outputs / "training")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = filter_labels(labeled, cfg)
    logger.info("Training data: %d frames, %d codes, %d session(s)",
                len(df), df["annotation_label"].nunique(), df["session"].nunique())

    df, feats = build_features(df, cfg)
    le = LabelEncoder().fit(df["annotation_label"])
    y_all = le.transform(df["annotation_label"])

    strategy = _choose_strategy(df, cfg)
    if strategy == "session":
        tr, te = session_split(df, test_size, seed)
    else:
        tr, te = time_split(df, test_size, float(mt.get("purge_gap_s", 2.0)))
    unseen = set(y_all[te]) - set(y_all[tr])
    if unseen:
        logger.warning("Codes only present in the test split are dropped from evaluation: %s",
                       list(le.inverse_transform(sorted(unseen))))
        te = te[~np.isin(y_all[te], list(unseen))]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("Empty train or test split — not enough data.")
    split_info = {
        "strategy": strategy, "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_sessions": sorted(df["session"].iloc[tr].unique().tolist()),
        "test_sessions": sorted(df["session"].iloc[te].unique().tolist()),
    }
    logger.info("Split (%s): %d train / %d test frames", strategy, len(tr), len(te))

    train_df = df.iloc[tr].reset_index(drop=True)
    keep, removed = select_features(train_df[feats], float(mt.get("redundancy_threshold", 0.95)), seed)
    logger.info("Features: %d kept (removed %d zero-variance, %d redundant)",
                len(keep), len(removed["zero_variance"]), len(removed["redundant"]))

    # Encode on the training classes only so model outputs map to le.classes_.
    classes = np.unique(y_all[tr])
    remap = {c: i for i, c in enumerate(classes)}
    label_names = le.inverse_transform(classes).tolist()
    X_tr = train_df[keep].to_numpy(dtype=np.float32)
    y_tr = np.vectorize(remap.get)(y_all[tr])
    X_te = df.iloc[te][keep].to_numpy(dtype=np.float32)
    y_te = np.vectorize(remap.get)(y_all[te])
    w_tr = compute_sample_weight("balanced", y_tr)
    val_mask = time_tail_mask(train_df)
    groups = cv_groups(train_df, n_folds)
    n_folds_eff = max(2, min(n_folds, len(np.unique(groups))))

    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for name in mt.get("models", ["LightGBM", "XGBoost", "HistGradientBoosting"]):
            logger.info("Fitting %s ...", name)
            try:
                if name == "LightGBM":
                    models[name] = _fit_lightgbm(X_tr, y_tr, w_tr, val_mask, seed, len(classes))
                elif name == "XGBoost":
                    models[name] = _fit_xgboost(X_tr, y_tr, w_tr, groups, seed,
                                                int(mt.get("n_iter_search", 30)), n_folds_eff)
                elif name == "HistGradientBoosting":
                    models[name] = _fit_hist(X_tr, y_tr, w_tr, val_mask, seed)
                else:
                    logger.warning("Unknown model '%s' — skipped", name)
            except ImportError as e:
                logger.warning("%s unavailable (%s) — skipped", name, e)
    if not models:
        raise RuntimeError("No model could be trained.")

    train_m, test_m, status = {}, {}, {}
    idx = np.arange(len(classes))
    for name, m in models.items():
        train_m[name] = compute_metrics(y_tr, m.predict(X_tr), m.predict_proba(X_tr), idx)
        test_m[name] = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), idx)
        gap = train_m[name][metric] - test_m[name][metric]
        status[name] = overfit_status(gap)
        logger.info("%-22s test %s=%.4f  bal_acc=%.4f  (train−test gap %.3f: %s)",
                    name, metric, test_m[name][metric], test_m[name]["balanced_accuracy"], gap, status[name])

    best_name = max(test_m, key=lambda k: test_m[k][metric])
    best = models[best_name]
    y_pred = best.predict(X_te)
    report = classification_report(y_te, y_pred, labels=idx, target_names=label_names, zero_division=0,
                                   output_dict=True)

    plot_model_comparison(test_m, train_m, metric, output_dir / "model_comparison.png", dpi)
    plot_feature_importance(models, keep, output_dir / "feature_importance.png", dpi)
    plot_confusion([label_names[i] for i in y_te], [label_names[i] for i in y_pred], label_names,
                   output_dir / "confusion_matrix.png", dpi, f"{best_name} — held-out ({strategy} split)")

    metrics = {
        "best_model": best_name, "selection_metric": metric, "split": split_info,
        "train_metrics": train_m, "test_metrics": test_m, "overfitting_status": status,
        "classification_report": report, "n_features": len(keep), "removed_features": removed,
        "classes": label_names,
    }
    write_json(metrics, output_dir / "model_metrics.json")
    write_json(keep, output_dir / "feature_names.json")

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "model": best,
        "model_name": best_name,
        "classes": label_names,
        "feature_names": keep,
        "features_config": cfg.get("features", {}),
        "roles": cfg.get("roles", {}),
        "metrics": {"test": test_m[best_name], "train": train_m[best_name], "split": split_info},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    bundle_path = model_bundle_path(cfg)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    logger.info("Best model: %s (%s=%.4f) → %s", best_name, metric, test_m[best_name][metric], bundle_path)

    return {"model_name": best_name, "score": test_m[best_name][metric], "metric": metric,
            "bundle_path": bundle_path, "metrics": metrics, "models": models}
