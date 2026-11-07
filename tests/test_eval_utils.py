import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "AI"))
import eval_utils as eu


def test_per_class_metrics():
    rows, _ = eu.per_class_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert rows[0]["sensitivity"] == 0.5 and rows[0]["specificity"] == 1.0
    assert rows[1]["sensitivity"] == 1.0 and rows[1]["specificity"] == 0.5


def test_auc_perfect():
    p = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    assert eu.ovr_auc([0, 0, 1, 1], p) == [1.0, 1.0]


def test_temperature():
    logits = np.array([[10., 0.]] * 4)           # very confident, right half the time
    assert eu.fit_temperature(logits, [0, 0, 1, 1]) > 1.0
    assert np.allclose(eu.apply_temperature(logits, 1.0), np.exp(logits) / np.exp(logits).sum(1, keepdims=True))


def test_brier():
    assert eu.brier_multiclass(np.eye(2)[[0, 1]], [0, 1]) == 0.0
    assert abs(eu.brier_multiclass(np.full((4, 2), 0.5), [0, 1, 0, 1]) - 0.5) < 1e-9


def test_reliability_counts():
    p = np.array([[.9, .1], [.6, .4], [.55, .45]])
    assert sum(r["n"] for r in eu.reliability_table(p, [0, 0, 1])) == 3


def test_bootstrap():
    y = np.array([0, 1] * 50); p = np.eye(2)[y]; p[:10] = p[:10][:, ::-1]   # 10 wrong
    pt, lo, hi = eu.bootstrap_ci(eu.accuracy, y, p, n_boot=300)
    assert abs(pt - 0.9) < 1e-9 and lo <= pt <= hi and lo < hi


def test_review_threshold():
    p = np.tile([0.9, 0.1], (100, 1))
    assert eu.pick_review_threshold(p, np.zeros(100, int)) == 0.9
    assert eu.pick_review_threshold(p, np.ones(100, int)) is None