"""
Evaluation upgrade script for Phase 1.
Computes bootstrap confidence intervals, paired McNemar exact tests,
confidence calibration (ECE and reliability diagram), slice evaluations,
and a fixed-weight ensemble on the 10,000 held-out test reviews.

All inputs are read-only.
Writes only new files to results/:
- results/evaluation_upgrade.csv
- results/mcnemar_tests.csv
- results/calibration.csv
- results/calibration.png
- results/slice_analysis.csv
- results/evaluation_upgrade.md
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 1: EVALUATION UPGRADE")
    print("=" * 70)

    # 1. Paths
    test_preds_path = os.path.join("data", "processed", "test_preds.parquet")
    tfidf_vec_path = os.path.join("models", "tfidf_vectorizer.joblib")
    logreg_path = os.path.join("models", "logreg_sentiment.joblib")
    topics_path = os.path.join("data", "processed", "topics_v1_merged.parquet")
    theme_labels_path = os.path.join("data", "processed", "theme_labels_final.json")

    # Output paths
    eval_upgrade_csv = os.path.join("results", "evaluation_upgrade.csv")
    mcnemar_csv = os.path.join("results", "mcnemar_tests.csv")
    calibration_csv = os.path.join("results", "calibration.csv")
    calibration_summary_csv = os.path.join("results", "calibration_summary.csv")
    calibration_png = os.path.join("results", "calibration.png")
    slice_csv = os.path.join("results", "slice_analysis.csv")
    eval_upgrade_md = os.path.join("results", "evaluation_upgrade.md")

    # 2. Load test predictions (read-only)
    print(f"\nLoading test predictions from {test_preds_path}...")
    df = pd.read_parquet(test_preds_path)
    N = len(df)
    assert N == 10000, f"Expected 10,000 rows, got {N}"
    y_true_str = df["sentiment"]
    y_true = (y_true_str == "positive").astype(np.int32).values

    # 3. Check baseline metrics match results/sentiment_comparison.csv
    print("\n--- Baseline Check (vs results/sentiment_comparison.csv) ---")
    expected_baselines = {
        "VADER default": (df["pred_vader"], 0.7189, 0.7041),
        "VADER tuned (0.7)": (df["pred_vader_tuned"], 0.7707, 0.7706),
        "TF-IDF+LogReg": (df["pred_tfidf"], 0.9017, 0.9017),
        "DistilBERT zero-shot": (df["pred_bert"], 0.8980, 0.8980),
        "DistilBERT fine-tuned": (df["pred_ft"], 0.9468, 0.9468),
    }

    baseline_match_all = True
    for m_name, (preds, exp_acc, exp_f1) in expected_baselines.items():
        yp = (preds == "positive").astype(np.int32).values
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fn = np.sum((y_true == 1) & (yp == 0))
        tn = np.sum((y_true == 0) & (yp == 0))

        acc = round(float((tp + tn) / N), 4)
        f1_pos = 2 * tp / (2 * tp + fp + fn)
        f1_neg = 2 * tn / (2 * tn + fp + fn)
        f1_macro = round(float((f1_pos + f1_neg) / 2.0), 4)

        acc_ok = (acc == exp_acc)
        f1_ok = (f1_macro == exp_f1)
        match_ok = acc_ok and f1_ok
        if not match_ok:
            baseline_match_all = False
        print(f"  {m_name:22s}: Accuracy={acc:.4f} (Match: {acc_ok}), Macro F1={f1_macro:.4f} (Match: {f1_ok})")

    if not baseline_match_all:
        print("\nERROR: Baseline metrics did not match results/sentiment_comparison.csv! STOPPING.")
        sys.exit(1)
    print("Baseline checks: ALL TRUE")

    # 4. TF-IDF Probabilities and validation
    print("\n--- TF-IDF Probabilities & Label Alignment ---")
    vec = joblib.load(tfidf_vec_path)
    clf = joblib.load(logreg_path)
    assert list(clf.classes_) == ["negative", "positive"], f"Unexpected classes: {clf.classes_}"
    
    X_test = vec.transform(df["text"])
    probas_tfidf = clf.predict_proba(X_test)
    preds_tfidf_calc = clf.predict(X_test)

    matches_tfidf = int(np.sum(preds_tfidf_calc == df["pred_tfidf"].values))
    print(f"TF-IDF predicted labels matching test_preds['pred_tfidf']: {matches_tfidf} / {N}")
    if matches_tfidf != N:
        mismatches = df[preds_tfidf_calc != df["pred_tfidf"].values]
        print(f"ERROR: Found {len(mismatches)} mismatches. Showing first 5:")
        print(mismatches[["review_id", "sentiment", "pred_tfidf"]].head(5))
        sys.exit(1)

    p_pos_tfidf = probas_tfidf[:, 1]
    conf_tfidf = np.max(probas_tfidf, axis=1)

    # 5. Zero-shot and Fine-tuned Probabilities
    # Reconstructed P(positive)
    p_pos_bert = np.where(df["pred_bert"] == "positive", df["bert_conf"].values, 1.0 - df["bert_conf"].values)
    p_pos_ft = np.where(df["pred_ft"] == "positive", df["ft_conf"].values, 1.0 - df["ft_conf"].values)

    conf_bert = df["bert_conf"].values
    conf_ft = df["ft_conf"].values

    # 6. Ensemble (fixed weights 0.5 / 0.5, threshold 0.5, ties to positive)
    print("\n--- Ensemble Evaluation (TF-IDF + Fine-Tuned) ---")
    p_pos_ens = 0.5 * p_pos_tfidf + 0.5 * p_pos_ft
    pred_ens_binary = (p_pos_ens >= 0.5).astype(np.int32)
    pred_ens_labels = np.where(pred_ens_binary == 1, "positive", "negative")

    tp_ens = np.sum((y_true == 1) & (pred_ens_binary == 1))
    fp_ens = np.sum((y_true == 0) & (pred_ens_binary == 1))
    fn_ens = np.sum((y_true == 1) & (pred_ens_binary == 0))
    tn_ens = np.sum((y_true == 0) & (pred_ens_binary == 0))

    acc_ens = (tp_ens + tn_ens) / N
    f1_pos_ens = 2 * tp_ens / (2 * tp_ens + fp_ens + fn_ens)
    f1_neg_ens = 2 * tn_ens / (2 * tn_ens + fp_ens + fn_ens)
    f1_ens = (f1_pos_ens + f1_neg_ens) / 2.0
    print(f"Ensemble: Accuracy={acc_ens:.4f}, Macro F1={f1_ens:.4f}")

    # 7. Efficient Bootstrap (2,000 resamples in chunks of 200, seed 42)
    print("\n--- Running Bootstrap (2,000 resamples, seed 42) ---")
    t_boot_start = time.time()
    rng = np.random.default_rng(42)
    n_resamples = 2000
    chunk_size = 200
    n_chunks = n_resamples // chunk_size

    eval_models = {
        "VADER default": (df["pred_vader"] == "positive").astype(np.int32).values,
        "VADER tuned (0.7)": (df["pred_vader_tuned"] == "positive").astype(np.int32).values,
        "TF-IDF+LogReg": (df["pred_tfidf"] == "positive").astype(np.int32).values,
        "DistilBERT zero-shot": (df["pred_bert"] == "positive").astype(np.int32).values,
        "DistilBERT fine-tuned": (df["pred_ft"] == "positive").astype(np.int32).values,
        "Ensemble (TF-IDF + FT)": pred_ens_binary,
    }

    # Precompute indicators for each model
    indicators = {}
    for m_name, yp in eval_models.items():
        indicators[m_name] = {
            "tp": ((y_true == 1) & (yp == 1)).astype(np.int32),
            "fp": ((y_true == 0) & (yp == 1)).astype(np.int32),
            "fn": ((y_true == 1) & (yp == 0)).astype(np.int32),
            "tn": ((y_true == 0) & (yp == 0)).astype(np.int32),
        }

    boot_results = {m_name: {"acc": [], "f1": []} for m_name in eval_models}

    for chunk in range(n_chunks):
        # resample indices chunk: shape (chunk_size, N)
        idx = rng.integers(0, N, size=(chunk_size, N), dtype=np.int32)
        for m_name in eval_models:
            tp_cnt = indicators[m_name]["tp"][idx].sum(axis=1)
            fp_cnt = indicators[m_name]["fp"][idx].sum(axis=1)
            fn_cnt = indicators[m_name]["fn"][idx].sum(axis=1)
            tn_cnt = indicators[m_name]["tn"][idx].sum(axis=1)

            acc = (tp_cnt + tn_cnt) / N
            f1_pos = 2 * tp_cnt / (2 * tp_cnt + fp_cnt + fn_cnt)
            f1_neg = 2 * tn_cnt / (2 * tn_cnt + fp_cnt + fn_cnt)
            f1_macro = (f1_pos + f1_neg) / 2.0

            boot_results[m_name]["acc"].extend(acc)
            boot_results[m_name]["f1"].extend(f1_macro)

    boot_runtime = time.time() - t_boot_start
    print(f"Bootstrap finished in {boot_runtime:.2f} s")

    # Paired accuracy differences across models
    acc_ft_boot = np.array(boot_results["DistilBERT fine-tuned"]["acc"])
    acc_tfidf_boot = np.array(boot_results["TF-IDF+LogReg"]["acc"])
    acc_zs_boot = np.array(boot_results["DistilBERT zero-shot"]["acc"])

    diff_ft_tfidf = acc_ft_boot - acc_tfidf_boot
    diff_zs_tfidf = acc_zs_boot - acc_tfidf_boot
    diff_ft_zs = acc_ft_boot - acc_zs_boot

    # Build evaluation_upgrade.csv rows
    eval_upgrade_rows = []
    for m_name in eval_models:
        acc_arr = np.array(boot_results[m_name]["acc"])
        f1_arr = np.array(boot_results[m_name]["f1"])

        # point estimates
        tp = int(indicators[m_name]["tp"].sum())
        fp = int(indicators[m_name]["fp"].sum())
        fn = int(indicators[m_name]["fn"].sum())
        tn = int(indicators[m_name]["tn"].sum())
        pt_acc = (tp + tn) / N
        f1_pos = 2 * tp / (2 * tp + fp + fn)
        f1_neg = 2 * tn / (2 * tn + fp + fn)
        pt_f1 = (f1_pos + f1_neg) / 2.0

        acc_ci_l = np.percentile(acc_arr, 2.5)
        acc_ci_h = np.percentile(acc_arr, 97.5)
        f1_ci_l = np.percentile(f1_arr, 2.5)
        f1_ci_h = np.percentile(f1_arr, 97.5)

        eval_upgrade_rows.append({
            "model": m_name,
            "metric": "accuracy",
            "value": pt_acc,
            "ci_low": acc_ci_l,
            "ci_high": acc_ci_h,
        })
        eval_upgrade_rows.append({
            "model": m_name,
            "metric": "macro_f1",
            "value": pt_f1,
            "ci_low": f1_ci_l,
            "ci_high": f1_ci_h,
        })

    # Add paired differences to evaluation_upgrade_rows
    paired_diffs = [
        ("Fine-tuned minus TF-IDF", 0.9468 - 0.9017, diff_ft_tfidf),
        ("Zero-shot minus TF-IDF", 0.8980 - 0.9017, diff_zs_tfidf),
        ("Fine-tuned minus Zero-shot", 0.9468 - 0.8980, diff_ft_zs),
    ]
    for p_name, pt_val, d_arr in paired_diffs:
        eval_upgrade_rows.append({
            "model": p_name,
            "metric": "accuracy_diff",
            "value": pt_val,
            "ci_low": np.percentile(d_arr, 2.5),
            "ci_high": np.percentile(d_arr, 97.5),
        })

    df_eval_upgrade = pd.DataFrame(eval_upgrade_rows)
    df_eval_upgrade.to_csv(eval_upgrade_csv, index=False)
    print(f"Saved {eval_upgrade_csv}")

    # 8. McNemar Exact Test (binomial on discordant pairs)
    print("\n--- McNemar Exact Test (Binomial, two-sided, p=0.5) ---")
    corr_ft = (eval_models["DistilBERT fine-tuned"] == y_true)
    corr_tfidf = (eval_models["TF-IDF+LogReg"] == y_true)
    corr_zs = (eval_models["DistilBERT zero-shot"] == y_true)
    corr_ens = (eval_models["Ensemble (TF-IDF + FT)"] == y_true)

    mcnemar_pairs = [
        ("Fine-tuned vs TF-IDF", corr_ft, corr_tfidf),
        ("Zero-shot vs TF-IDF", corr_zs, corr_tfidf),
        ("Fine-tuned vs Zero-shot", corr_ft, corr_zs),
        ("Ensemble vs Fine-tuned", corr_ens, corr_ft),
    ]

    mcnemar_rows = []
    for comp_name, c1, c2 in mcnemar_pairs:
        b = int(np.sum(c1 & ~c2))
        c = int(np.sum(~c1 & c2))
        n_disc = b + c
        if n_disc > 0:
            res = binomtest(b, n=n_disc, p=0.5, alternative="two-sided")
            p_val = res.pvalue
        else:
            p_val = 1.0

        mcnemar_rows.append({
            "comparison": comp_name,
            "b": b,
            "c": c,
            "total_discordant": n_disc,
            "p_value": p_val,
        })
        print(f"  {comp_name:25s}: b={b:4d}, c={c:4d}, total={n_disc:4d}, p-value={p_val:.8e}")

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv(mcnemar_csv, index=False)
    print(f"Saved {mcnemar_csv}")

    # 9. Calibration & Reliability Diagram
    print("\n--- Confidence Calibration (10 bins on [0.5, 1.0]) ---")
    calib_configs = [
        ("TF-IDF+LogReg", conf_tfidf, eval_models["TF-IDF+LogReg"]),
        ("DistilBERT zero-shot", conf_bert, eval_models["DistilBERT zero-shot"]),
        ("DistilBERT fine-tuned", conf_ft, eval_models["DistilBERT fine-tuned"]),
    ]

    calibration_rows = []
    reliability_data = {}

    bin_edges = np.linspace(0.5, 1.0, 11)

    for model_name, conf_arr, yp_arr in calib_configs:
        is_correct = (yp_arr == y_true).astype(float)
        conf_clamped = np.clip(conf_arr, 0.5, 1.0)
        bin_idx = np.floor((conf_clamped - 0.5) / 0.05).astype(int)
        bin_idx = np.clip(bin_idx, 0, 9)

        ece = 0.0
        b_confs = []
        b_accs = []
        b_counts = []

        print(f"\nModel: {model_name}")
        for b in range(10):
            b_low = round(float(bin_edges[b]), 2)
            b_high = round(float(bin_edges[b + 1]), 2)
            mask = (bin_idx == b)
            cnt = int(np.sum(mask))

            if cnt > 0:
                mean_c = float(np.mean(conf_clamped[mask]))
                acc = float(np.mean(is_correct[mask]))
                ece += (cnt / N) * abs(acc - mean_c)
                print(f"  Bin {b} [{b_low:.2f}, {b_high:.2f}]: count={cnt:5d}, mean_conf={mean_c:.4f}, acc={acc:.4f}")
            else:
                mean_c = float((b_low + b_high) / 2.0)
                acc = np.nan
                print(f"  Bin {b} [{b_low:.2f}, {b_high:.2f}]: EMPTY")

            b_confs.append(mean_c)
            b_accs.append(acc)
            b_counts.append(cnt)

            calibration_rows.append({
                "model": model_name,
                "bin_index": b,
                "bin_lower": b_low,
                "bin_upper": b_high,
                "count": cnt,
                "mean_confidence": round(mean_c, 4),
                "accuracy": round(acc, 4) if not np.isnan(acc) else np.nan,
                "ece_running_total": round(float(ece), 6),
            })

        print(f"  => ECE for {model_name}: {ece:.6f}")
        reliability_data[model_name] = {
            "mean_conf": b_confs,
            "accuracy": b_accs,
            "count": b_counts,
            "ece": ece,
        }

    df_calibration = pd.DataFrame(calibration_rows)
    df_calibration.to_csv(calibration_csv, index=False)
    print(f"Saved {calibration_csv}")

    # Save calibration summary
    calib_summary_rows = [
        {"model": m_name, "ece": round(float(reliability_data[m_name]["ece"]), 4)}
        for m_name in ["TF-IDF+LogReg", "DistilBERT zero-shot", "DistilBERT fine-tuned"]
    ]
    df_calib_summary = pd.DataFrame(calib_summary_rows)
    df_calib_summary.to_csv(calibration_summary_csv, index=False)
    print(f"Saved {calibration_summary_csv}")

    # Plot reliability diagram
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", label="Perfect Calibration (y = x)", linewidth=1.5)

    colors = {"TF-IDF+LogReg": "#1f77b4", "DistilBERT zero-shot": "#ff7f0e", "DistilBERT fine-tuned": "#2ca02c"}
    markers = {"TF-IDF+LogReg": "o", "DistilBERT zero-shot": "s", "DistilBERT fine-tuned": "^"}

    for m_name, d in reliability_data.items():
        valid = [i for i, a in enumerate(d["accuracy"]) if not np.isnan(a)]
        x_pts = [d["mean_conf"][i] for i in valid]
        y_pts = [d["accuracy"][i] for i in valid]
        label = f"{m_name} (ECE={d['ece']:.4f})"
        ax.plot(x_pts, y_pts, marker=markers[m_name], color=colors[m_name], label=label, linewidth=2, markersize=6)

    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.40, 1.02)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_title("Reliability Diagram (Confidence Calibration on 10,000 Test Reviews)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(calibration_png, dpi=200)
    plt.close(fig)
    print(f"Saved {calibration_png}")

    # 10. Slice Analysis
    print("\n--- Slice Analysis ---")
    topics_df = pd.read_parquet(topics_path)
    merged_test = df.merge(topics_df, on="review_id", how="left")
    missing_topics = merged_test["topic_v1_merged"].isna().sum()
    print(f"Test reviews with no matching topic in topics_v1_merged: {missing_topics}")
    assert missing_topics == 0, f"Expected 0 missing topic matches, found {missing_topics}"

    with open(theme_labels_path, "r", encoding="utf-8") as f:
        theme_labels = json.load(f)

    slice_rows = []

    # A. Review length quartiles (words)
    word_counts = df["text"].str.split().str.len()
    q_cuts = pd.qcut(word_counts, q=4)
    q_intervals = q_cuts.unique().sort_values()

    for i, interval in enumerate(q_intervals, 1):
        mask = (q_cuts == interval)
        n_slice = int(mask.sum())
        acc_tfidf = float(np.mean(eval_models["TF-IDF+LogReg"][mask] == y_true[mask]))
        acc_zs = float(np.mean(eval_models["DistilBERT zero-shot"][mask] == y_true[mask]))
        acc_ft = float(np.mean(eval_models["DistilBERT fine-tuned"][mask] == y_true[mask]))
        slice_rows.append({
            "slice_category": "Review Length Quartiles",
            "slice_name": f"Q{i} ({int(interval.left)+1} to {int(interval.right)} words)",
            "n": n_slice,
            "small_n": bool(n_slice < 200),
            "acc_tfidf": acc_tfidf,
            "acc_zero_shot": acc_zs,
            "acc_fine_tuned": acc_ft,
        })

    # B. Fine-tuned confidence bands (<0.7, 0.7-0.9, >0.9)
    conf_bands = [
        ("< 0.70", (conf_ft < 0.70)),
        ("0.70 to 0.90", (conf_ft >= 0.70) & (conf_ft <= 0.90)),
        ("> 0.90", (conf_ft > 0.90)),
    ]
    for band_name, mask in conf_bands:
        n_slice = int(mask.sum())
        acc_tfidf = float(np.mean(eval_models["TF-IDF+LogReg"][mask] == y_true[mask]))
        acc_zs = float(np.mean(eval_models["DistilBERT zero-shot"][mask] == y_true[mask]))
        acc_ft = float(np.mean(eval_models["DistilBERT fine-tuned"][mask] == y_true[mask]))
        slice_rows.append({
            "slice_category": "Fine-Tuned Confidence Bands",
            "slice_name": band_name,
            "n": n_slice,
            "small_n": bool(n_slice < 200),
            "acc_tfidf": acc_tfidf,
            "acc_zero_shot": acc_zs,
            "acc_fine_tuned": acc_ft,
        })

    # C. Merged product themes with at least 100 test reviews
    # Unclustered row (-1)
    topic_counts = merged_test["topic_v1_merged"].value_counts()
    for t_id, cnt in topic_counts.items():
        if cnt >= 100:
            if t_id == -1:
                t_label = "Unclustered"
            else:
                t_label = theme_labels.get(str(t_id), f"Theme {t_id}")

            mask = (merged_test["topic_v1_merged"] == t_id).values
            n_slice = int(cnt)
            acc_tfidf = float(np.mean(eval_models["TF-IDF+LogReg"][mask] == y_true[mask]))
            acc_zs = float(np.mean(eval_models["DistilBERT zero-shot"][mask] == y_true[mask]))
            acc_ft = float(np.mean(eval_models["DistilBERT fine-tuned"][mask] == y_true[mask]))
            slice_rows.append({
                "slice_category": "Product Themes (>= 100 test reviews)",
                "slice_name": f"{t_label} (Topic {t_id})",
                "n": n_slice,
                "small_n": bool(n_slice < 200),
                "acc_tfidf": acc_tfidf,
                "acc_zero_shot": acc_zs,
                "acc_fine_tuned": acc_ft,
            })

    df_slice = pd.DataFrame(slice_rows)
    df_slice.to_csv(slice_csv, index=False)
    print(f"Saved {slice_csv}")

    # 11. Generate Markdown Report (results/evaluation_upgrade.md)
    print("\n--- Generating results/evaluation_upgrade.md ---")
    md_content = []
    md_content.append("# Evaluation Upgrade: Statistical Rigor & Detailed Slices\n")
    md_content.append("This document records the empirical results of the Phase 1 evaluation upgrade on the 10,000 held-out test reviews from `data/processed/test_preds.parquet`. All bootstrap confidence intervals reflect 2,000 resamples with random seed 42.\n")

    md_content.append("## 1. Model Performance with 95% Bootstrap Confidence Intervals\n")
    md_content.append("| Model | Metric | Value | 95% CI Lower | 95% CI Upper |")
    md_content.append("|:---|:---:|:---:|:---:|:---:|")
    for _, r in df_eval_upgrade.iterrows():
        md_content.append(f"| {r['model']} | {r['metric']} | {r['value']:.4f} | {r['ci_low']:.4f} | {r['ci_high']:.4f} |")

    md_content.append("\n## 2. McNemar Exact Discordant Tests\n")
    md_content.append("Exact two-sided binomial test on discordant pairs ($p=0.5$):\n")
    md_content.append("| Comparison | Model 1 Right, Model 2 Wrong (b) | Model 1 Wrong, Model 2 Right (c) | Total Discordant (b + c) | Two-Sided p-value |")
    md_content.append("|:---|:---:|:---:|:---:|:---:|")
    for _, r in df_mcnemar.iterrows():
        md_content.append(f"| {r['comparison']} | {int(r['b']):d} | {int(r['c']):d} | {int(r['total_discordant']):d} | {r['p_value']:.8e} |")

    md_content.append("\n## 3. Confidence Calibration & Expected Calibration Error (ECE)\n")
    md_content.append("Calculated across 10 equal-width bins on $[0.50, 1.00]$. Overall ECE values:\n")
    for m_name, d in reliability_data.items():
        md_content.append(f"- **{m_name}**: ECE = **{d['ece']:.4f}**")
    md_content.append("\nece_running_total is the cumulative sum of bin contributions; its last row per model is the model's ECE.\n")
    md_content.append("| Model | Bin Index | Bin Range | Count | Mean Confidence | Empirical Accuracy | Cumulative ECE (ece_running_total) |")
    md_content.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_calibration.iterrows():
        acc_str = f"{r['accuracy']:.4f}" if pd.notnull(r['accuracy']) else "EMPTY"
        md_content.append(f"| {r['model']} | {int(r['bin_index']):d} | [{r['bin_lower']:.2f}, {r['bin_upper']:.2f}] | {int(r['count']):d} | {r['mean_confidence']:.4f} | {acc_str} | {r['ece_running_total']:.6f} |")

    md_content.append("\n## 4. Slice Analysis\n")
    md_content.append("| Slice Category | Slice Name | Sample Size (n) | Small n (< 200) | TF-IDF Accuracy | Zero-Shot Accuracy | Fine-Tuned Accuracy |")
    md_content.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_slice.iterrows():
        small_n_str = "Yes" if r['small_n'] else "No"
        md_content.append(f"| {r['slice_category']} | {r['slice_name']} | {int(r['n']):d} | {small_n_str} | {r['acc_tfidf']:.4f} | {r['acc_zero_shot']:.4f} | {r['acc_fine_tuned']:.4f} |")

    # Fine-tuned accuracy by confidence band subtable
    acc_ft_above90 = float(np.mean(eval_models["DistilBERT fine-tuned"][conf_ft >= 0.90] == y_true[conf_ft >= 0.90]))
    n_below90 = int(np.sum(conf_ft < 0.90))
    n_above90 = int(np.sum(conf_ft >= 0.90))
    md_content.append("\n### Fine-Tuned Accuracy by Confidence Band\n")
    md_content.append("| Confidence Band | Review Count (n) | Fine-Tuned Accuracy |")
    md_content.append("|:---|:---:|:---:|")
    for band_name, mask in conf_bands:
        n_b = int(mask.sum())
        acc_b = float(np.mean(eval_models["DistilBERT fine-tuned"][mask] == y_true[mask]))
        md_content.append(f"| {band_name} | {n_b} | {acc_b:.4f} |")
    md_content.append(f"\nExcluding the {n_below90} reviews with fine-tuned confidence below 0.90, accuracy on the remaining {n_above90:,} reviews is {acc_ft_above90:.4f}. The band edges were fixed in advance; this is descriptive and not a tuned threshold.\n")

    md_content.append("\n## 5. Methodological Limits\n")
    md_content.append("- **Sampling variability**: Confidence intervals reflect test set sampling variability only and do not account for data collection bias or annotation shifts.")
    md_content.append("- **Confidence calibration**: Calibration is evaluated against binary correctness using the maximum predicted class probability.")
    md_content.append("- **Sample size in slices**: Several slices have sample size below 200 reviews (`small_n = True`), increasing variance in slice-level estimates.")
    md_content.append("- **Untuned ensemble**: The ensemble model combines TF-IDF and fine-tuned transformer probabilities using fixed equal weights (0.5 / 0.5) without parameter tuning.")
    md_content.append("- **Single test set**: All evaluations are conducted on a single held-out 10,000-review test split.")

    full_md_text = "\n".join(md_content) + "\n"
    with open(eval_upgrade_md, "w", encoding="utf-8") as f:
        f.write(full_md_text)
    print(f"Saved {eval_upgrade_md}")

    # 12. Print Full Tables (with 4 decimals)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY TABLES (4 Decimals)")
    print("=" * 70)

    print("\nTABLE 1: MODEL PERFORMANCE & 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print(df_eval_upgrade.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTABLE 2: MCNEMAR EXACT DISCORDANT PAIRS & P-VALUES")
    print(df_mcnemar.to_string(index=False, formatters={
        "b": "{:d}".format, "c": "{:d}".format, "total_discordant": "{:d}".format, "p_value": "{:.8e}".format
    }))

    print("\nTABLE 3: CONFIDENCE CALIBRATION & EXPECTED CALIBRATION ERROR (ECE)")
    print(df_calibration.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTABLE 4: SLICE ANALYSIS")
    print(df_slice.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    total_time = time.time() - start_time
    print(f"\nPhase 1 script execution completed in {total_time:.2f} s")

if __name__ == "__main__":
    main()

