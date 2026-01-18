#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symptoms = cfg.get("symptoms", [])
    conditions = cfg.get("conditions", {})
    prevalence = cfg.get("prevalence", {})
    noise = float(cfg.get("noise_probability", 0.02))

    if not symptoms or not isinstance(symptoms, list):
        raise ValueError("Config error: 'symptoms' must be a non-empty list")
    if not conditions or not isinstance(conditions, dict):
        raise ValueError("Config error: 'conditions' must be a non-empty mapping")
    if not prevalence or not isinstance(prevalence, dict):
        raise ValueError("Config error: 'prevalence' must be a non-empty mapping")

    for name, meta in conditions.items():
        probs = meta.get("probabilities", [])
        if len(probs) != len(symptoms):
            raise ValueError(
                f"Config error: condition '{name}' has {len(probs)} probabilities, "
                f"but there are {len(symptoms)} symptoms."
            )

    total_prev = sum(float(v) for v in prevalence.values())
    if not (0.999 <= total_prev <= 1.001):
        raise ValueError(f"Config error: prevalence must sum to 1.0 (got {total_prev:.4f})")

    return {
        "symptoms": symptoms,
        "conditions": conditions,
        "prevalence": prevalence,
        "noise_probability": noise,
    }

def generate_dataset(num_rows: int, cfg: dict, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)

    symptoms = cfg["symptoms"]
    conditions = list(cfg["conditions"].keys())
    prevalence = np.array([cfg["prevalence"][c] for c in conditions], dtype=float)
    noise_p = float(cfg["noise_probability"])

    cond_prob_arrays = {
        c: np.array(cfg["conditions"][c]["probabilities"], dtype=float) for c in conditions
    }

    chosen_idx = np.random.choice(len(conditions), size=num_rows, p=prevalence)
    chosen_conditions = [conditions[i] for i in chosen_idx]

    num_symptoms = len(symptoms)
    X = np.zeros((num_rows, num_symptoms), dtype=np.int8)

    for ci, cname in enumerate(conditions):
        mask = (chosen_idx == ci)
        n = int(mask.sum())
        if n == 0:
            continue
        probs = cond_prob_arrays[cname]
        U = np.random.rand(n, num_symptoms)
        X[mask, :] = (U < probs).astype(np.int8)

        if noise_p > 0:
            noise_mask = (X[mask, :] == 0) & (np.random.rand(n, num_symptoms) < noise_p)
            X_sub = X[mask, :]
            X_sub[noise_mask] = 1
            X[mask, :] = X_sub

    import pandas as pd
    df = pd.DataFrame(X, columns=symptoms)
    df["Diagnosis"] = chosen_conditions
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic symptom dataset (Plan B)")
    parser.add_argument("--config", type=Path, default=Path("conditions_config.yaml"),
                        help="Path to YAML config with symptoms/conditions/prevalence.")
    parser.add_argument("--rows", type=int, default=25000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=Path, default=Path("nigeria_7_conditions_dataset.csv"),
                        help="Output CSV filename.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = generate_dataset(num_rows=args.rows, cfg=cfg, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Generated {len(df):,} rows -> {args.out}")
    print("Diagnosis distribution:")
    print(df['Diagnosis'].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    main()
