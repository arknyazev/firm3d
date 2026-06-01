"""Estimator metrics + CSV writing shared by the three MC-comparison
workflows.

Definitions (same across all three methods):

    Y_i                = per-sample estimator contribution
                         * forward MC           : Y_i = A_i
                         * IS                   : Y_i = A_i * w_i
    Q_hat              = mean(Y_i)
    sample_variance    = var(Y_i, ddof=1)
    estimator_variance = sample_variance / N
    standard_error     = sqrt(estimator_variance)
    cv_estimator       = standard_error / Q_hat
    cv_single_sample   = sqrt(sample_variance) / Q_hat
    N_target_cv_c%     = (cv_single_sample / c)**2

For IS methods we additionally report
    w_min / w_max / w_mean / w_median / w_std
    effective_sample_size = (sum w)^2 / sum(w^2)
"""
import csv
import numpy as np


def estimator_metrics(Y, method_name, N, cv_targets=(0.10, 0.05, 0.02)):
    Y = np.asarray(Y, dtype=np.float64)
    Q_hat = float(Y.mean())
    var_sample = float(Y.var(ddof=1)) if Y.size > 1 else 0.0
    var_estimator = var_sample / N
    se = float(np.sqrt(var_estimator))
    if Q_hat > 0.0:
        cv_est = se / Q_hat
        cv_one = float(np.sqrt(var_sample)) / Q_hat
    else:
        cv_est = cv_one = float("nan")
    out = {
        "method":             method_name,
        "N":                  int(N),
        "Q_hat":              Q_hat,
        "sample_variance":    var_sample,
        "estimator_variance": var_estimator,
        "standard_error":     se,
        "cv_estimator":       cv_est,
        "cv_single_sample":   cv_one,
    }
    for c in cv_targets:
        key = f"N_target_cv_{int(round(100 * c)):02d}pct"
        out[key] = (float((cv_one / c) ** 2)
                    if np.isfinite(cv_one) else float("nan"))
    return out


def is_weight_diagnostics(w):
    w = np.asarray(w, dtype=np.float64)
    N = int(w.size)
    ess = (float(w.sum() ** 2 / np.sum(w ** 2))
           if np.any(w > 0) else 0.0)
    return {
        "w_min":                 float(w.min()) if N else 0.0,
        "w_max":                 float(w.max()) if N else 0.0,
        "w_mean":                float(w.mean()) if N else 0.0,
        "w_median":              float(np.median(w)) if N else 0.0,
        "w_std":                 float(w.std(ddof=1)) if N > 1 else 0.0,
        "effective_sample_size": ess,
    }


def summarize_stop_codes(stop_codes):
    uniq, counts = np.unique(np.asarray(stop_codes).astype(int),
                             return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def write_metrics_csv(path, rows):
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})