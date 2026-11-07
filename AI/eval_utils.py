"""Shared evaluation helpers for the MRI / US / CT classifiers.
Depends only on numpy, scipy and scikit-learn."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import confusion_matrix, roc_auc_score


def per_class_metrics(y_true, y_pred, n_classes):
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=list(range(n_classes)))
    total, rows = cm.sum(), []
    for k in range(n_classes):
        tp = cm[k, k]; fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp; tn = total - tp - fn - fp
        rows.append({"class": k,
                     "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                     "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan")})
    return rows, cm


def ovr_auc(y_true, probs):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    out = []
    for k in range(probs.shape[1]):
        pos = y_true == k
        out.append(float("nan") if pos.all() or (~pos).all()
                   else float(roc_auc_score(pos, probs[:, k])))
    return out


def macro_auc(y_true, probs):
    return float(np.nanmean(ovr_auc(y_true, probs)))


def accuracy(y_true, probs):
    return float((np.argmax(probs, 1) == np.asarray(y_true)).mean())


def fit_temperature(logits, y_true):
    """Fit on VALIDATION logits only."""
    logits, y = np.asarray(logits, float), np.asarray(y_true)
    def nll(log_t):
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(logits, T):
    return softmax(np.asarray(logits, float) / T, axis=1)


def brier_multiclass(probs, y_true):
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability_table(probs, y_true, edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)):
    probs, y = np.asarray(probs), np.asarray(y_true)
    conf, correct = probs.max(1), probs.argmax(1) == y
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append({"range": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()),
                     "mean_conf": float(conf[m].mean()) if m.any() else float("nan"),
                     "accuracy": float(correct[m].mean()) if m.any() else float("nan")})
    return rows


def bootstrap_ci(metric_fn, y_true, probs, n_boot=1000, seed=0, alpha=0.05):
    """metric_fn(y_true, probs) -> float. Returns (point, low, high)."""
    y, p = np.asarray(y_true), np.asarray(probs)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = metric_fn(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y, p)), float(lo), float(hi)


def pick_review_threshold(probs, y_true, target_acc=0.95, min_n=20):
    """Smallest confidence cutoff where accuracy above it reaches target_acc
    (use VALIDATION data). Returns None if never reached."""
    probs, y = np.asarray(probs), np.asarray(y_true)
    conf, correct = probs.max(1), probs.argmax(1) == y
    for t in np.unique(conf):
        m = conf >= t
        if m.sum() < min_n:
            break
        if correct[m].mean() >= target_acc:
            return float(t)
    return None