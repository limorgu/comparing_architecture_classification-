import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_config() -> Dict:
    return load_json(ROOT / "benchmark_config.json")


def make_run_root() -> Path:
    run_root = ROOT / "Runs" / f"BENCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def detect_columns(df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str], config: Dict) -> Tuple[str, str]:
    if text_col is None:
        for candidate in config["default_text_columns"]:
            if candidate in df.columns:
                text_col = candidate
                break
    if label_col is None:
        for candidate in config["default_label_columns"]:
            if candidate in df.columns:
                label_col = candidate
                break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Found columns: {list(df.columns)}")
    return text_col, label_col


def load_dataset(dataset_path: Path, text_col: Optional[str], label_col: Optional[str], config: Dict) -> pd.DataFrame:
    if dataset_path.suffix.lower() == ".jsonl":
        rows = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(dataset_path)
    text_col, label_col = detect_columns(df, text_col, label_col, config)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio = config["train_ratio"]
    dev_ratio = config["dev_ratio"]
    test_ratio = config["test_ratio"]
    if round(train_ratio + dev_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/dev/test ratios must sum to 1.0")
    stratify_full = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=config["random_seed"],
        stratify=stratify_full,
    )
    relative_test = test_ratio / (dev_ratio + test_ratio)
    stratify_temp = temp_df["label"] if temp_df["label"].value_counts().min() >= 2 else None
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=config["random_seed"],
        stratify=stratify_temp,
    )
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_split(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_text_model(kind: str):
    if kind == "logreg":
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ])
    if kind == "nb":
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", MultinomialNB()),
        ])
    if kind == "sgd":
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", SGDClassifier(loss="log_loss", max_iter=2000, random_state=42, class_weight="balanced")),
        ])
    raise ValueError(kind)


def metrics_payload(y_true: List[str], y_pred: List[str], labels: List[str], architecture: str) -> Dict:
    eval_labels = list(labels)
    report = classification_report(y_true, y_pred, labels=eval_labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=eval_labels).tolist()
    abstain_count = sum(1 for x in y_pred if x == "ABSTAIN")
    effective_preds = [x for x in y_pred if x != "ABSTAIN"]
    effective_true = [yt for yt, yp in zip(y_true, y_pred) if yp != "ABSTAIN"]
    return {
        "architecture": architecture,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0),
        "abstain_rate": abstain_count / len(y_pred) if y_pred else 0.0,
        "effective_accuracy": accuracy_score(effective_true, effective_preds) if effective_preds else 0.0,
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": eval_labels,
        "test_label_distribution": dict(Counter(y_true)),
    }


def write_predictions(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["text", "true_label", "predicted_label", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def run_openai_only(test_df: pd.DataFrame, labels: List[str], config: Dict) -> Tuple[List[Dict], Dict]:
    client = OpenAI()
    limit = min(len(test_df), config["openai_test_limit"])
    rows = []
    y_true: List[str] = []
    y_pred: List[str] = []
    label_text = ", ".join(labels)

    for _, row in test_df.head(limit).iterrows():
        prompt = (
            "Classify the text into exactly one label from this set: "
            f"{label_text}. "
            "Return compact JSON only with keys label, confidence, reason. "
            "If uncertain, still choose the closest label."
        )
        response = client.responses.create(
            model=config["openai_model"],
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": row["text"]},
            ],
        )
        output = response.output_text.strip()
        parsed = {"label": "ABSTAIN", "confidence": 0.0, "reason": output[:200]}
        try:
            parsed = json.loads(output)
        except Exception:
            pass
        predicted = parsed.get("label", "ABSTAIN")
        if predicted not in labels:
            predicted = "ABSTAIN"
        y_true.append(row["label"])
        y_pred.append(predicted)
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": predicted,
                "confidence": parsed.get("confidence", 0.0),
                "reason": parsed.get("reason", ""),
            }
        )
    metrics = metrics_payload(y_true, y_pred, labels + ["ABSTAIN"], "openai_only")
    metrics["evaluated_rows"] = limit
    metrics["dataset_test_rows"] = len(test_df)
    return rows, metrics


def run_ensemble_judge(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, labels: List[str]) -> Tuple[List[Dict], Dict]:
    base_models = {
        "logreg": build_text_model("logreg"),
        "nb": build_text_model("nb"),
        "sgd": build_text_model("sgd"),
    }
    for model in base_models.values():
        model.fit(train_df["text"], train_df["label"])

    def stack_features(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for name, model in base_models.items():
            probs = model.predict_proba(df["text"])
            proba_df = pd.DataFrame(probs, columns=[f"{name}__{label}" for label in model.classes_])
            parts.append(proba_df)
        return pd.concat(parts, axis=1)

    if dev_df["label"].nunique() >= 2:
        x_dev = stack_features(dev_df)
        x_test = stack_features(test_df)
        meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        meta.fit(x_dev, dev_df["label"])
        test_probs = meta.predict_proba(x_test)
        preds = meta.classes_[test_probs.argmax(axis=1)]
        confs = test_probs.max(axis=1)
    else:
        prob_mats = [model.predict_proba(test_df["text"]) for model in base_models.values()]
        avg_probs = sum(prob_mats) / len(prob_mats)
        classes = list(base_models.values())[0].classes_
        preds = classes[avg_probs.argmax(axis=1)]
        confs = avg_probs.max(axis=1)

    rows = []
    y_true = test_df["label"].tolist()
    y_pred = preds.tolist()
    for idx, row in test_df.reset_index(drop=True).iterrows():
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": preds[idx],
                "confidence": round(float(confs[idx]), 4),
            }
        )
    return rows, metrics_payload(y_true, y_pred, labels, "ensemble_judge")


def run_chain_review(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: List[str], config: Dict) -> Tuple[List[Dict], Dict]:
    proposer = build_text_model("logreg")
    reviewer = build_text_model("nb")
    resolver = build_text_model("sgd")
    proposer.fit(train_df["text"], train_df["label"])
    reviewer.fit(train_df["text"], train_df["label"])
    resolver.fit(train_df["text"], train_df["label"])

    rows = []
    y_true: List[str] = []
    y_pred: List[str] = []
    for _, row in test_df.iterrows():
        p_probs = proposer.predict_proba([row["text"]])[0]
        r_probs = reviewer.predict_proba([row["text"]])[0]
        z_probs = resolver.predict_proba([row["text"]])[0]
        p_label, p_conf = proposer.classes_[p_probs.argmax()], float(p_probs.max())
        r_label, r_conf = reviewer.classes_[r_probs.argmax()], float(r_probs.max())
        z_label, z_conf = resolver.classes_[z_probs.argmax()], float(z_probs.max())

        if p_label == r_label and p_conf >= config["chain_accept_threshold"]:
            final = p_label
            stage = "proposer+reviewer_agree"
            conf = p_conf
        elif r_label == z_label and r_conf >= config["chain_accept_threshold"]:
            final = r_label
            stage = "reviewer+resolver_agree"
            conf = r_conf
        else:
            best = max([(p_label, p_conf, "proposer"), (r_label, r_conf, "reviewer"), (z_label, z_conf, "resolver")], key=lambda x: x[1])
            final = best[0] if best[1] >= config["chain_final_threshold"] else "ABSTAIN"
            stage = best[2] if final != "ABSTAIN" else "abstain"
            conf = best[1]

        y_true.append(row["label"])
        y_pred.append(final)
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": final,
                "confidence": round(conf, 4),
                "chain_stage": stage,
                "proposer_label": p_label,
                "reviewer_label": r_label,
                "resolver_label": z_label,
            }
        )
    return rows, metrics_payload(y_true, y_pred, labels + ["ABSTAIN"], "chain_review")


def write_run_notes(run_root: Path, dataset_path: Path, config: Dict, labels: List[str]) -> None:
    write_json(
        run_root / "run_notes.json",
        {
            "run_id": run_root.name,
            "dataset_path": str(dataset_path),
            "labels": labels,
            "architectures": [
                "openai_only",
                "ensemble_judge",
                "chain_review"
            ],
            "what_changed": [
                "Added a separate benchmark pipeline for labeled text datasets.",
                "Ground truth is the dataset label column, not rule output from the book pipeline.",
                "OpenAI-only is inference-only; local architectures are train/dev/test supervised.",
                "Comparison is written only after per-architecture metrics are produced."
            ],
            "config": config,
            "created_at": datetime.now().isoformat(),
        },
    )


def run_pipeline(
    dataset_path: Path,
    text_col: Optional[str],
    label_col: Optional[str],
    eval_dataset_path: Optional[Path] = None,
    eval_text_col: Optional[str] = None,
    eval_label_col: Optional[str] = None,
) -> Path:
    config = load_config()
    run_root = make_run_root()
    b1 = run_root / "B1_DataPrep"
    b2 = run_root / "B2_Predictions"
    b3 = run_root / "B3_Evaluation"
    b4 = run_root / "B4_Comparison"
    for p in [b1, b2, b3, b4]:
        p.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, text_col, label_col, config)
    train_df, dev_df, split_test_df = split_dataset(df, config)
    if eval_dataset_path is not None:
        test_df = load_dataset(eval_dataset_path, eval_text_col, eval_label_col, config)
        labels = sorted(set(df["label"].unique().tolist()) | set(test_df["label"].unique().tolist()))
    else:
        test_df = split_test_df
        labels = sorted(df["label"].unique().tolist())
    write_split(b1 / "train.csv", train_df)
    write_split(b1 / "dev.csv", dev_df)
    write_split(b1 / "test.csv", test_df)
    write_json(
        b1 / "dataset_manifest.json",
        {
            "dataset_path": str(dataset_path),
            "eval_dataset_path": str(eval_dataset_path) if eval_dataset_path else str(dataset_path),
            "total_rows": len(df),
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "test_rows": len(test_df),
            "labels": labels,
            "label_distribution": dict(Counter(df["label"])),
        },
    )

    results = {}

    try:
        openai_rows, openai_metrics = run_openai_only(test_df, labels, config)
    except Exception as exc:
        openai_rows = []
        openai_metrics = {
            "architecture": "openai_only",
            "error": str(exc),
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "abstain_rate": 1.0,
            "effective_accuracy": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
            "labels": labels + ["ABSTAIN"],
            "test_label_distribution": dict(Counter(test_df["label"])),
            "evaluated_rows": 0,
            "dataset_test_rows": len(test_df),
        }
    write_predictions(b2 / "openai_only" / "predictions.csv", openai_rows)
    write_json(b3 / "openai_only" / "metrics.json", openai_metrics)
    results["openai_only"] = openai_metrics

    ensemble_rows, ensemble_metrics = run_ensemble_judge(train_df, dev_df, test_df, labels)
    write_predictions(b2 / "ensemble_judge" / "predictions.csv", ensemble_rows)
    write_json(b3 / "ensemble_judge" / "metrics.json", ensemble_metrics)
    results["ensemble_judge"] = ensemble_metrics

    chain_rows, chain_metrics = run_chain_review(train_df, test_df, labels, config)
    write_predictions(b2 / "chain_review" / "predictions.csv", chain_rows)
    write_json(b3 / "chain_review" / "metrics.json", chain_metrics)
    results["chain_review"] = chain_metrics

    comparison_rows = []
    for arch, metrics in results.items():
        comparison_rows.append(
            {
                "architecture": arch,
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "abstain_rate": round(metrics["abstain_rate"], 4),
                "effective_accuracy": round(metrics["effective_accuracy"], 4),
                "test_rows": len(test_df),
                "test_label_distribution": json.dumps(metrics.get("test_label_distribution", dict(Counter(test_df["label"]))), ensure_ascii=False),
                "error": metrics.get("error", ""),
            }
        )
    comparison_rows = sorted(comparison_rows, key=lambda row: row["macro_f1"], reverse=True)
    with (b4 / "comparison_dashboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    md_lines = [
        "# Architecture Comparison",
  
          "",
        "| Architecture | Accuracy | Macro F1 | Weighted F1 | Abstain Rate | Effective Accuracy | Test Rows | Error |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        md_lines.append(
            f"| {row['architecture']} | {row['accuracy']} | {row['macro_f1']} | {row['weighted_f1']} | {row['abstain_rate']} | {row['effective_accuracy']} | {row['test_rows']} | {row['error']} |"
        )
    (b4 / "comparison_dashboard.md").write_text("\n".join(md_lines), encoding="utf-8")

    write_run_notes(run_root, dataset_path, config, labels)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Mental health architecture benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    prun = sub.add_parser("run", help="Run the full benchmark pipeline.")
    prun.add_argument("--dataset", required=True)
    prun.add_argument("--text-col", default=None)
    prun.add_argument("--label-col", default=None)
    prun.add_argument("--eval-dataset", default=None)
    prun.add_argument("--eval-text-col", default=None)
    prun.add_argument("--eval-label-col", default=None)
    args = parser.parse_args()

    if args.command == "run":
        run_root = run_pipeline(
            Path(args.dataset),
            args.text_col,
            args.label_col,
            Path(args.eval_dataset) if args.eval_dataset else None,
            args.eval_text_col,
            args.eval_label_col,
        )
        print(json.dumps({"status": "ok", "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()
