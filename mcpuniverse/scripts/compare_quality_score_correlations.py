#!/usr/bin/env python3
"""Compute Kendall's tau and Spearman's rho between GPT and Sonnet quality scores."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Sequence, Tuple

from scipy.stats import kendalltau, spearmanr


def _read_scores(
    path: Path,
    *,
    gpt_col: str,
    sonnet_col: str,
) -> List[Tuple[float, float]]:
    """Read paired numeric scores from CSV, skipping rows with missing/invalid values."""

    scores: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        missing = [col for col in (gpt_col, sonnet_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {', '.join(missing)}"
            )
        for row in reader:
            gpt_raw = (row.get(gpt_col) or "").strip()
            sonnet_raw = (row.get(sonnet_col) or "").strip()
            if not gpt_raw or not sonnet_raw:
                continue
            try:
                gpt_score = float(gpt_raw)
                sonnet_score = float(sonnet_raw)
            except ValueError:
                continue
            if math.isnan(gpt_score) or math.isnan(sonnet_score):
                continue
            scores.append((gpt_score, sonnet_score))
    return scores


def _write_results(
    path: Path,
    *,
    tau: float,
    tau_p: float,
    rho: float,
    rho_p: float,
) -> None:
    """Write correlation results to CSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "p_value"])
        writer.writeheader()
        writer.writerow({"metric": "kendall_tau", "value": tau, "p_value": tau_p})
        writer.writerow({"metric": "spearman_rho", "value": rho, "p_value": rho_p})


def _compute_correlations(pairs: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (tau, tau_p, rho, rho_p) for the given score pairs."""

    if len(pairs) < 2:
        raise ValueError("At least two paired scores are required to compute correlations.")
    gpt_scores, sonnet_scores = zip(*pairs)
    tau_res = kendalltau(gpt_scores, sonnet_scores, nan_policy="omit")
    rho_res = spearmanr(gpt_scores, sonnet_scores, nan_policy="omit")
    return float(tau_res.statistic), float(tau_res.pvalue), float(rho_res.statistic), float(rho_res.pvalue)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Kendall's tau and Spearman's rho for GPT vs Sonnet quality scores."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path containing GPT and Sonnet quality scores.",
    )
    parser.add_argument(
        "--gpt-col",
        default="description_quality_score_from_gpt",
        help="Column name for GPT quality scores (default: description_quality_score_from_gpt).",
    )
    parser.add_argument(
        "--sonnet-col",
        default="description_quality_score_from_sonnet",
        help="Column name for Sonnet quality scores (default: description_quality_score_from_sonnet).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()

    pairs = _read_scores(input_path, gpt_col=args.gpt_col, sonnet_col=args.sonnet_col)
    tau, tau_p, rho, rho_p = _compute_correlations(pairs)


    print(f"Kendall's tau: {tau:.4f} (p={tau_p:.4g})")
    print(f"Spearman's rho: {rho:.4f} (p={rho_p:.4g})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
