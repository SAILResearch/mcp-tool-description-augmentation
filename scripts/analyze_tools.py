"""Analyze MCP tool description quality CSVs and emit reports and figures (LLM smells)."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- NEW: OpenAI client import (optional provider swap-in) ---
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # lazy guard; we validate at runtime if LLM is enabled

# ------------------------------
# Canonical smell names (6 themes)
# ------------------------------
SMELL_LABELS = [
    "too brief/short",
    "no guidance",
    "no examples",
    "no purpose",
    "missing parameter explanation",
    "no limitations",
]

# For backward compatibility: keep but unused in LLM path
DEFAULT_THEME_PATTERNS = {
    "too brief/short": r"\b(too\s+)?brief\b|\bvery\s+brief\b|\bshort\b",
    "vague/unclear": r"\bvague\b|\bunclear\b|\bgeneric\b|\bambiguous\b",
    "no usage guidance/examples": r"when to use|not use|usage|examples?",
    "no warnings/limitations": r"limitations|warnings?|caveats?|what is not returned",
    "lacks purpose/what it does": r"purpose|what the tool does|objective",
    "missing parameter explanation": r"parameters?|args?|fields?.*(explain|meaning|effect|format)",
}

# Put this near your other helpers (top of file with imports)
_EXAMPLES_REGEX = re.compile(
    r"\bexamples?\b|for example|e\.g\.|such as|sample(?:s)?|usage example|walkthrough|demonstration|code snippet|snippet",
    flags=re.IGNORECASE
)

def _reason_mentions_examples(text: str) -> bool:
    return bool(_EXAMPLES_REGEX.search(text or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MCP tool description quality data (LLM smells).")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV file")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for analysis outputs")
    parser.add_argument("--score-col", default="description_quality_score", help="Column for quality scores")
    parser.add_argument("--server-col", default="mcp_server_name", help="Column for server names")
    parser.add_argument("--tool-col", default="tool_name", help="Column for tool names")
    parser.add_argument("--reason-col", default="description_reason", help="Column for judge reasons")
    parser.add_argument("--missing-col", default="description_missing_points", help="Column for missing items")
    parser.add_argument("--score-threshold", type=float, default=70.0, help="Score threshold for pass rate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top missing items to surface")
    parser.add_argument("--by-server", default="yes", help="Whether to compute per-server stats (yes/no)")
    parser.add_argument("--figure-dpi", type=int, default=200, help="Figure DPI (resolution)")
    parser.add_argument("--figure-fontsize", type=int, default=16, help="Base matplotlib font size")
    parser.add_argument("--max-servers", type=int, default=10, help="Maximum servers to show in boxplot")
    # --- NEW LLM options ---
    parser.add_argument("--use-llm", default="yes", help="Use LLM to detect smells from reasons (yes/no)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI chat model name")
    parser.add_argument("--llm-batch-size", type=int, default=50, help="Batch size for LLM calls")
    parser.add_argument("--llm-max-retries", type=int, default=3, help="Max retries per LLM batch")
    parser.add_argument("--augmented-out-csv", default="augmented_with_smells.csv",
                    help="Filename for the output CSV that includes detected_smells")

    return parser.parse_args()

def write_augmented_csv(outdir: Path, df: pd.DataFrame, filename: str) -> None:
    df.to_csv(outdir / filename, index=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        normalized = df[col].fillna("").astype(str).str.strip()
        normalized = normalized.replace({"None": "", "none": "", "NA": "", "na": "", "N/A": "", "n/a": ""})
        df[col] = normalized
    return df


def load_data(path: Path, colmap: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    missing_cols = [actual for actual in colmap.values() if actual not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")
    score_col = colmap["score"]
    before = len(df)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with non-numeric scores", flush=True)
    return df


def compute_basic_stats(df: pd.DataFrame, score_col: str, threshold: float) -> Dict[str, float]:
    scores = df[score_col]
    stats = {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "std": float(scores.std(ddof=1)) if scores.shape[0] > 1 else 0.0,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "pct_below_threshold": float((scores < threshold).mean() * 100.0),
    }
    return stats


def split_items(cell: object) -> List[str]:
    if not isinstance(cell, str) or not cell:
        return []
    tokens = re.split(r"[;,\n]+", cell.lower())
    cleaned = [re.sub(r"\s+", " ", token).strip(" .;:,") for token in tokens]
    cleaned = [token for token in cleaned if token and token not in {"none", "n/a", "na"}]
    return sorted(set(cleaned))


def count_missing_items(df: pd.DataFrame, missing_col: str) -> Tuple[pd.DataFrame, Counter]:
    df = df.copy()
    item_counts: Counter[str] = Counter()
    per_row_counts: List[int] = []
    for _, row in df.iterrows():
        items = split_items(row.get(missing_col, ""))
        per_row_counts.append(len(items))
        item_counts.update(items)
    df["missing_count"] = per_row_counts
    return df, item_counts


def compute_server_missing(
    df: pd.DataFrame, server_col: str, missing_col: str
) -> Tuple[Dict[str, float], int]:
    per_server_items: Dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        server = row.get(server_col, "")
        if not server:
            continue
        items = split_items(row.get(missing_col, ""))
        per_server_items[server].update(items)
    server_item_counts: Counter[str] = Counter()
    for items in per_server_items.values():
        server_item_counts.update(items)
    server_count = len(per_server_items)
    server_pct_map = {
        item: (count / server_count * 100.0 if server_count else np.nan)
        for item, count in server_item_counts.items()
    }
    return server_pct_map, server_count


def top_k_items(
    tool_item_counts: Counter,
    total_tools: int,
    server_pct_map: Dict[str, float],
    k: int,
) -> List[Tuple[str, int, float, float | None]]:
    rows: List[Tuple[str, int, float, float | None]] = []
    if not total_tools:
        return rows
    for item, count in tool_item_counts.items():
        tool_pct = count / total_tools * 100.0
        server_pct = server_pct_map.get(item)
        rows.append((item, count, tool_pct, server_pct))
    rows.sort(key=lambda entry: (entry[2], entry[1]), reverse=True)
    return rows[:k]


# ------------------------------
# NEW: LLM-based smell detection
# ------------------------------

# --- Replace helpers + classify_reasons_llm with this ---

def _ensure_openai_client(model_name: str):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(), model_name

def _build_llm_prompt_json_array(reasons: List[str]) -> List[Dict[str, str]]:
    system = (
        "You are a strict multi-label classifier for tool description quality smells.\n"
        "Allowed labels ONLY:\n"
        f"{', '.join(SMELL_LABELS)}\n\n"
        "Labeling rules (very important):\n"
        "1) If the rationale states that examples are missing, assign \"no examples\".\n"
        "2) If the rationale states that examples ARE provided, do NOT assign \"no examples\".\n"
        "3) If the rationale does NOT mention examples at all (no words like 'example', 'examples', 'e.g.', "
        "'for example', 'sample', 'usage example', 'walkthrough', 'demonstration', 'code snippet'), "
        "ASSIGN \"no examples\" by default.\n"
        "4) Other labels follow their plain meaning.\n\n"
        "Output format:\n"
        "- Return a JSON array with the same length as the number of inputs; each element is an array of 0..N labels.\n"
        "- Use ONLY the exact allowed label strings.\n"
        "- If none apply, return an empty array for that item."
    )
    user = (
        "Classify the following judge rationales according to the rules above.\n\n"
        "Inputs (JSON array of strings):\n"
        + json.dumps(reasons, ensure_ascii=False)
        + "\nReturn ONLY the JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _build_llm_prompt_jsonl(reasons: List[str]) -> List[Dict[str, str]]:
    system = (
        "You are a strict multi-label classifier for tool description quality smells.\n"
        "Allowed labels ONLY:\n"
        f"{', '.join(SMELL_LABELS)}\n\n"
        "Labeling rules (very important):\n"
        "1) If the rationale states that examples are missing, assign \"no examples\".\n"
        "2) If the rationale states that examples ARE provided, do NOT assign \"no examples\".\n"
        "3) If the rationale does NOT mention examples at all (no words like 'example', 'examples', 'e.g.', "
        "'for example', 'sample', 'usage example', 'walkthrough', 'demonstration', 'code snippet'), "
        "ASSIGN \"no examples\" by default.\n"
        "4) Other labels follow their plain meaning.\n\n"
        "Output format:\n"
        "- Return JSON Lines: one line per input; each line is a JSON array of 0..N labels using ONLY the allowed strings.\n"
        "- No prose, no extra lines."
    )
    user_lines = "\n".join([json.dumps(r, ensure_ascii=False) for r in reasons])
    user = "Inputs (one JSON string per line):\n" + user_lines + "\nReturn JSON Lines, one per input."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _normalize_labels(labels: Iterable[str]) -> List[str]:
    alias_map = {
        "too brief": "too brief/short",
        "brief": "too brief/short",
        "short": "too brief/short",
        "no usage guidance": "no guidance",
        "no guidance/examples": "no guidance",
        "missing examples": "no examples",
        "no example": "no examples",
        "no examples provided": "no examples",
        "no purpose/what it does": "no purpose",
        "missing parameter details": "missing parameter explanation",
        "missing parameter": "missing parameter explanation",
        "missing parameters": "missing parameter explanation",
        "no parameter explanation": "missing parameter explanation",
        "missing limitations": "no limitations",
        "no caveats": "no limitations",
    }
    normed = []
    for lab in labels:
        s = alias_map.get(str(lab).strip().lower(), str(lab).strip().lower())
        if s in SMELL_LABELS and s not in normed:
            normed.append(s)
    return normed

def _parse_json_array_or_raise(text: str, expected_len: int) -> List[List[str]]:
    data = json.loads(text)
    if not isinstance(data, list) or any(not isinstance(x, list) for x in data):
        raise ValueError("Malformed JSON structure (expected list of lists).")
    # length guard (pad/truncate)
    if len(data) < expected_len:
        data = data + [[] for _ in range(expected_len - len(data))]
    elif len(data) > expected_len:
        data = data[:expected_len]
    return data

def _parse_jsonl_or_raise(text: str, expected_len: int) -> List[List[str]]:
    rows = [ln for ln in text.splitlines() if ln.strip()]
    parsed = []
    for ln in rows[:expected_len]:
        try:
            arr = json.loads(ln)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
        parsed.append(arr)
    if len(parsed) < expected_len:
        parsed += [[] for _ in range(expected_len - len(parsed))]
    return parsed

def classify_reasons_llm(
    df: pd.DataFrame,
    reason_col: str,
    model_name: str,
    batch_size: int = 20,        # smaller default
    max_retries: int = 3,
) -> Tuple[pd.DataFrame, Counter]:
    if df.empty:
        df = df.copy()
        df["detected_smells"] = ""
        return df, Counter()

    client, model = _ensure_openai_client(model_name)
    reasons = df[reason_col].fillna("").astype(str).tolist()
    all_labels_per_row: List[List[str]] = []

    i = 0
    while i < len(reasons):
        batch = reasons[i : i + batch_size]
        # First try: strict JSON array using response_format
        attempt = 0
        succeeded = False
        while attempt < max_retries and not succeeded:
            attempt += 1
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=_build_llm_prompt_json_array(batch),
                    response_format={"type": "json_object"},  # forces valid JSON
                    max_tokens=2048,
                )
                content = resp.choices[0].message.content.strip()
                # If response_format=json_object, model may wrap in {"data": [...]}
                try:
                    obj = json.loads(content)
                    if isinstance(obj, dict) and "data" in obj:
                        parsed = obj["data"]
                        if not isinstance(parsed, list):
                            raise ValueError("data is not a list")
                        data = parsed
                    else:
                        data = _parse_json_array_or_raise(content, len(batch))
                except Exception:
                    # If it’s not an object, parse as raw array
                    data = _parse_json_array_or_raise(content, len(batch))
                for labels in data:
                    all_labels_per_row.append(_normalize_labels(labels))
                succeeded = True
            except Exception as e:
                if attempt >= max_retries:
                    print(f"LLM strict-JSON batch failed after {attempt} attempts: {e}", flush=True)

        # Fallback: JSONL (line-per-item), no response_format
        if not succeeded:
            attempt = 0
            while attempt < max_retries and not succeeded:
                attempt += 1
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        messages=_build_llm_prompt_jsonl(batch),
                        max_tokens=2048,
                    )
                    content = resp.choices[0].message.content.strip()
                    data = _parse_jsonl_or_raise(content, len(batch))
                    for labels in data:
                        all_labels_per_row.append(_normalize_labels(labels))
                    succeeded = True
                except Exception as e:
                    if attempt >= max_retries:
                        print(f"LLM JSONL fallback failed after {attempt} attempts: {e}", flush=True)
                        # Graceful degrade: no labels for this batch
                        all_labels_per_row.extend([[] for _ in batch])

        i += batch_size

    df = df.copy()
    df["detected_smells"] = [", ".join(lbls) for lbls in all_labels_per_row]

    smell_counts: Counter = Counter()
    for labels in all_labels_per_row:
        smell_counts.update(labels)

    return df, smell_counts



# ------------------------------
# Figures and reporting (unchanged)
# ------------------------------

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def figure_score_distribution(df: pd.DataFrame, score_col: str, outpath: Path, dpi: int) -> None:
    plt.figure(figsize=(12, 8), dpi=dpi)
    scores = df[score_col].dropna()
    plt.hist(scores, bins=20, color="#4C72B0", edgecolor="white")
    mean = scores.mean()
    median = scores.median()
    plt.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.1f}")
    plt.axvline(median, color="green", linestyle=":", linewidth=2, label=f"Median: {median:.1f}")
    plt.xlabel("Description quality score")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def figure_missing_vs_score(df: pd.DataFrame, score_col: str, outpath: Path, dpi: int) -> None:
    plt.figure(figsize=(12, 8), dpi=dpi)
    rng = np.random.default_rng(42)
    x = df["missing_count"].to_numpy()
    jitter = rng.normal(scale=0.1, size=x.shape)
    plt.scatter(x + jitter, df[score_col], alpha=0.5, color="#55A868")
    if len(df) >= 2 and df["missing_count"].nunique() > 1:
        slope, intercept = np.polyfit(df["missing_count"], df[score_col], 1)
        x_vals = np.linspace(df["missing_count"].min(), df["missing_count"].max(), 100)
        plt.plot(x_vals, slope * x_vals + intercept, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Missing items count")
    plt.ylabel("Description quality score")
    plt.title("Missing items vs. score")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def figure_top_missing(top_df: pd.DataFrame, outpath: Path, dpi: int) -> None:
    plt.figure(figsize=(12, 8), dpi=dpi)
    if top_df.empty:
        plt.text(0.5, 0.5, "No missing items", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        ordered = top_df.sort_values("tool_pct")
        bars = plt.barh(ordered["item"], ordered["tool_pct"], color="#C44E52")
        plt.xlabel("% of tools missing item")
        plt.title("Top missing items")
        for bar, value in zip(bars, ordered["tool_pct"]):
            plt.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
        plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def figure_reason_themes(theme_df: pd.DataFrame, outpath: Path, dpi: int) -> None:
    plt.figure(figsize=(12, 8), dpi=dpi)
    if theme_df.empty:
        plt.text(0.5, 0.5, "No themes detected", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        ordered = theme_df.sort_values("pct")
        bars = plt.barh(ordered["theme"], ordered["pct"], color="#8172B2")
        plt.xlabel("% of tools with theme present")
        plt.title("Judge reason themes")
        for bar, value in zip(bars, ordered["pct"]):
            plt.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
        plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def figure_server_boxplot(
    df: pd.DataFrame,
    server_col: str,
    score_col: str,
    outpath: Path,
    max_servers: int,
    dpi: int,
) -> None:
    if df.empty:
        return
    means = df.groupby(server_col)[score_col].mean().sort_values(ascending=False)
    top_servers = means.head(max_servers).index.tolist()
    subset = df[df[server_col].isin(top_servers)]
    order = subset.groupby(server_col)[score_col].mean().sort_values(ascending=False).index.tolist()
    data = [subset[subset[server_col] == srv][score_col].dropna().to_numpy() for srv in order]
    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.ylabel("Description quality score")
    plt.title("Score distribution by server (top by mean score)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def build_notes(stats: Dict[str, float], top_items: pd.DataFrame, theme_df: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    if stats["pct_below_threshold"] >= 50:
        notes.append("More than half of tool descriptions fall below the quality threshold; prioritize broad documentation improvements.")
    elif stats["pct_below_threshold"] >= 25:
        notes.append("A sizable minority of descriptions are underperforming; focus remediation on the weakest tools first.")
    top_missing = top_items.head(1)
    if not top_missing.empty:
        item = top_missing.iloc[0]
        notes.append(f"The most common gap is '{item['item']}' affecting {item['tool_pct']:.1f}% of tools.")
    top_theme = theme_df.head(1)
    if not top_theme.empty:
        theme = top_theme.iloc[0]
        notes.append(f"Judges most frequently cite '{theme['theme']}' ({theme['pct']:.1f}% of tools).")
    if not notes:
        notes.append("Tool descriptions generally meet expectations with no dominant deficiencies detected.")
    return notes


def write_summary_json(outdir: Path, stats: Dict[str, float], config: Dict[str, object]) -> None:
    payload = {"stats": stats, "config": config}
    (outdir / "summary_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_top_missing_csv(outdir: Path, top_df: pd.DataFrame) -> None:
    top_df.to_csv(outdir / "top_missing_items.csv", index=False)


def write_reason_themes_csv(outdir: Path, theme_df: pd.DataFrame) -> None:
    theme_df.to_csv(outdir / "reason_themes.csv", index=False)


def write_server_stats_csv(outdir: Path, server_stats_df: pd.DataFrame) -> None:
    if server_stats_df.empty:
        return
    server_stats_df.to_csv(outdir / "server_stats.csv")


def write_report(
    outdir: Path,
    stats: Dict[str, float],
    top_df: pd.DataFrame,
    theme_df: pd.DataFrame,
    server_stats_df: pd.DataFrame,
    notes: Iterable[str],
    config: Dict[str, object],
) -> None:
    report_path = outdir / "report.md"
    lines: List[str] = []
    lines.append("# Tool Description Quality Analysis\n")
    lines.append("## Summary statistics\n")
    lines.append(
        """
- Count: {count}
- Mean: {mean:.2f}
- Median: {median:.2f}
- Std: {std:.2f}
- Min: {min:.2f}
- Max: {max:.2f}
- % below threshold ({threshold:.0f}): {pct_below_threshold:.2f}%
""".strip().format(threshold=config["score_threshold"], **stats)
    )
    lines.append("\n## Top missing items\n")
    if top_df.empty:
        lines.append("No missing items reported.\n")
    else:
        for _, row in top_df.iterrows():
            server_pct = (f", servers: {row['server_pct']:.1f}%" if pd.notna(row.get("server_pct")) else "")
            lines.append(f"- {row['item']}: tools {row['tool_pct']:.1f}%{server_pct}\n")
    lines.append("\n## Reason themes (LLM-detected smells)\n")
    if theme_df.empty:
        lines.append("No recurring themes detected.\n")
    else:
        for _, row in theme_df.iterrows():
            lines.append(f"- {row['theme']}: {row['pct']:.1f}%\n")
    lines.append("\n## Notes & insights\n")
    for note in notes:
        lines.append(f"- {note}\n")
    lines.append("\n## Figures\n")
    figures = [
        "score_distribution.png",
        "missing_vs_score.png",
        "top_missing_items.png",
        "reason_themes.png",
    ]
    if not server_stats_df.empty:
        figures.append("server_scores_boxplot.png")
    for figure in figures:
        lines.append(f"![{figure}]({figure})\n")
    lines.append("\n## Recommended remediation checklist\n")
    lines.extend(
        [
            "- Add a concise description that covers what the tool does, why it exists, and when to use it.\n",
            "- Include guidance on when to use the tool and when not to, highlighting expected inputs/outputs.\n",
            "- Explain each parameter: name, type, format, defaults, and their effects.\n",
            "- Document caveats or limitations, including what the tool does not return.\n",
            "- Provide one minimal working example and one edge-case example.\n",
        ]
    )
    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_outdir(args.outdir)
    colmap = {
        "server": args.server_col.strip().lower().replace(" ", "_"),
        "tool": args.tool_col.strip().lower().replace(" ", "_"),
        "score": args.score_col.strip().lower().replace(" ", "_"),
        "reason": args.reason_col.strip().lower().replace(" ", "_"),
        "missing": args.missing_col.strip().lower().replace(" ", "_"),
    }
    df = load_data(args.input, colmap)

    stats = compute_basic_stats(df, colmap["score"], args.score_threshold)
    df, tool_item_counts = count_missing_items(df, colmap["missing"])

    by_server = str(args.by_server).strip().lower() in {"yes", "true", "1"}
    server_pct_map: Dict[str, float] = {}
    server_count = 0
    if by_server:
        server_pct_map, server_count = compute_server_missing(df, colmap["server"], colmap["missing"])
        print(f"Computed missing item coverage across {server_count} servers", flush=True)

    top_rows = top_k_items(tool_item_counts, len(df), server_pct_map, args.top_k)
    top_df = pd.DataFrame(top_rows, columns=["item", "tool_count", "tool_pct", "server_pct"])

    # --- NEW: LLM smells instead of regex ---
    use_llm = str(args.use_llm).strip().lower() in {"yes", "true", "1"}
    if use_llm:
        df, smell_counts = classify_reasons_llm(
            df, colmap["reason"], model_name=args.openai_model,
            batch_size=args.llm_batch_size, max_retries=args.llm_max_retries
        )
    else:
        # Fallback: no smells detected
        df = df.copy()
        df["detected_smells"] = ""
        smell_counts = Counter()

    # Build theme_df (percent of tools with the smell)
    theme_total = len(df)
    theme_data = []
    for smell in SMELL_LABELS:
        count = smell_counts.get(smell, 0)
        pct = (count / theme_total * 100.0) if theme_total else 0.0
        theme_data.append((smell, count, pct))
    theme_df = pd.DataFrame(theme_data, columns=["theme", "count", "pct"]).sort_values("pct", ascending=False)

    # Per-server stats (unchanged)
    server_stats_df = pd.DataFrame()
    if by_server:
        grouped = df.groupby(colmap["server"])[colmap["score"]]
        server_stats_df = pd.DataFrame(
            {
                "count": grouped.count(),
                "mean": grouped.mean(),
                "median": grouped.median(),
                "std": grouped.std(ddof=1),
                "min": grouped.min(),
                "max": grouped.max(),
            }
        ).sort_values("mean", ascending=False)

    # Figures
    plt.rcParams.update({"font.size": args.figure_fontsize})
    figure_score_distribution(df, colmap["score"], args.outdir / "score_distribution.png", args.figure_dpi)
    figure_missing_vs_score(df, colmap["score"], args.outdir / "missing_vs_score.png", args.figure_dpi)
    truncated = top_df.copy()
    if not truncated.empty:
        truncated["item"] = truncated["item"].str.slice(0, 80)
    figure_top_missing(truncated, args.outdir / "top_missing_items.png", args.figure_dpi)
    figure_reason_themes(theme_df, args.outdir / "reason_themes.png", args.figure_dpi)
    if by_server and not server_stats_df.empty:
        figure_server_boxplot(
            df,
            colmap["server"],
            colmap["score"],
            args.outdir / "server_scores_boxplot.png",
            args.max_servers,
            args.figure_dpi,
        )

    config = {
        "input": str(args.input),
        "score_column": args.score_col,
        "server_column": args.server_col,
        "tool_column": args.tool_col,
        "reason_column": args.reason_col,
        "missing_column": args.missing_col,
        "score_threshold": args.score_threshold,
        "top_k": args.top_k,
        "by_server": by_server,
        "figure_dpi": args.figure_dpi,
        "figure_fontsize": args.figure_fontsize,
        "max_servers": args.max_servers,
        "server_count": server_count,
        # NEW: LLM config
        "use_llm": use_llm,
        "openai_model": args.openai_model,
        "llm_batch_size": args.llm_batch_size,
        "llm_max_retries": args.llm_max_retries,
        "smell_labels": SMELL_LABELS,
    }

    write_summary_json(args.outdir, stats, config)
    write_top_missing_csv(args.outdir, top_df)
    write_reason_themes_csv(args.outdir, theme_df)
    write_server_stats_csv(args.outdir, server_stats_df)

    notes = build_notes(stats, top_df, theme_df)
    write_augmented_csv(args.outdir, df, args.augmented_out_csv)
    write_report(args.outdir, stats, top_df, theme_df, server_stats_df, notes, config)

    print(f"Analysis complete. Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
