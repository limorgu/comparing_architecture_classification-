import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_config() -> Dict:
    with (ROOT / "benchmark_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced evaluation slice from a labeled dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    dataset_path = Path(args.dataset)
    df = pd.read_csv(dataset_path)
    text_col, label_col = detect_columns(df, args.text_col, args.label_col, config)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    counts = df["label"].value_counts().to_dict()
    missing = [label for label, count in counts.items() if count < args.per_class]
    if missing:
        raise ValueError(f"These labels do not have enough rows for per-class={args.per_class}: {missing}")

    parts = []
    for label in sorted(df["label"].unique()):
        sample = df[df["label"] == label].sample(n=args.per_class, random_state=args.seed)
        parts.append(sample)
    slice_df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_root = ROOT / "Eval_Slices"
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_path.stem}_balanced_{args.per_class * len(parts)}"
    csv_path = out_root / f"{stem}.csv"
    manifest_path = out_root / f"{stem}_manifest.json"

    slice_df.to_csv(csv_path, index=False)
    manifest = {
        "source_dataset": str(dataset_path),
        "output_csv": str(csv_path),
        "rows": len(slice_df),
        "per_class": args.per_class,
        "label_distribution": slice_df["label"].value_counts().to_dict(),
        "seed": args.seed,
        "created_at": datetime.now().isoformat(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"status": "ok", "csv": str(csv_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
