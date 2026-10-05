#!/usr/bin/env python3
"""Measure how well a prompted local model corrects spelling on data/eval.jsonl.

Sends each example to an OpenAI-compatible chat endpoint (llama.cpp), aligns the
output with the input word by word, and reports exact match, recall, false
positives, rewrites and latency - overall, per language and per error type.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
ROLE_MARKER = re.compile(r"###\s*(system|user|assistant)\s*", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
THINKING_WARNING = (
    "WARNING: the model is thinking before answering. Output was stripped, but "
    "latency and tokens/sec include the thinking. Disable it on the server "
    "(llama-server --reasoning-budget 0)."
)


class ServerError(Exception):
    pass


# --------------------------------------------------------------------------
# Prompt and server
# --------------------------------------------------------------------------


def load_prompt(path: Path) -> list[dict]:
    """Parse a prompt file into chat messages (system + few-shot turns)."""
    messages, role, buffer = [], None, []

    def flush():
        if role:
            messages.append({"role": role, "content": "\n".join(buffer).strip()})

    for line in path.read_text(encoding="utf-8").splitlines():
        marker = ROLE_MARKER.fullmatch(line.strip())
        if marker:
            flush()
            role, buffer = marker.group(1).lower(), []
        else:
            buffer.append(line)
    flush()
    if not messages:
        raise ValueError(f"{path}: no '### system' / '### user' / '### assistant' sections found")
    return messages


def chat(url: str, messages: list[dict], model: str | None, max_tokens: int, timeout: float):
    """POST one chat completion. Returns (response payload, latency in seconds)."""
    body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if model:
        body["model"] = model
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ServerError(f"server returned HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ServerError(f"cannot reach the server ({getattr(e, 'reason', e)})") from e
    return payload, time.perf_counter() - start


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove <think> blocks. Returns (clean text, whether any thinking was found)."""
    found = "<think>" in text or "</think>" in text
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:  # closing tag only: the opening one was in the template
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:  # never closed: cut off by max_tokens
        text = text.split("<think>", 1)[0]
    return text.strip(), found


def correct(text: str, prompt: list[dict], args) -> dict:
    """Ask the model to correct `text`."""
    messages = [*prompt, {"role": "user", "content": text}]
    payload, latency = chat(args.url, messages, args.model, args.max_tokens, args.timeout)
    message = payload["choices"][0]["message"]
    output, thought = strip_thinking(message.get("content") or "")
    return {
        "output": output,
        "latency_s": latency,
        "thinking": thought or bool(message.get("reasoning_content")),
        "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
        "model": payload.get("model"),
        "truncated": payload["choices"][0].get("finish_reason") == "length",
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def classify(source: str, target: str, output: str, errors: list[dict]) -> dict:
    """Align output with input word by word and sort every difference into:

    fixed           an injected error, corrected to the target word
    missed          an injected error, left as is or changed to something else
    false_positives a correct word that was changed
    rewrites        blocks where words were inserted, deleted or re-split
    """
    src, tgt, out = source.split(), target.split(), output.split()
    type_at = {e["word_index"]: e["type"] for e in errors}
    result = {"fixed": [], "missed": [], "false_positives": [], "rewrites": []}

    def judge(i: int, got: str | None) -> None:
        if i in type_at:
            record = {"word_index": i, "type": type_at[i], "input": src[i], "expected": tgt[i], "output": got}
            result["fixed" if got == tgt[i] else "missed"].append(record)
        elif got is not None and got != src[i]:
            result["false_positives"].append({"word_index": i, "input": src[i], "output": got})

    matcher = difflib.SequenceMatcher(a=src, b=out, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                judge(i, out[j])
        else:  # insert, delete, or replace with a different word count
            result["rewrites"].append({"op": op, "input": src[i1:i2], "output": out[j1:j2]})
            for i in range(i1, i2):  # an error inside a rewritten block counts as fixed if its target word shows up
                judge(i, tgt[i] if tgt[i] in out[j1:j2] else None)
    return result


def score_example(example: dict, reply: dict) -> dict:
    """One row of outputs.jsonl."""
    output = reply["output"]
    return {
        "id": example["id"],
        "lang": example["lang"],
        "clean": example["clean"],
        "input": example["input"],
        "target": example["target"],
        "output": output,
        "exact": output == example["target"],
        "words": len(example["input"].split()),
        "errors": example["errors"],
        **classify(example["input"], example["target"], output, example["errors"]),
        "latency_s": round(reply["latency_s"], 3),
        "completion_tokens": reply["completion_tokens"],
        "truncated": reply["truncated"],
    }


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil
    return ordered[int(rank) - 1]


def ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def summarize(rows: list[dict], error_type: str | None = None) -> dict:
    """Aggregate metrics over rows. With `error_type`, keep only the examples
    containing that error type and count recall on errors of that type only."""

    def of_type(records):
        return [r for r in records if error_type is None or r["type"] == error_type]

    if error_type:
        rows = [r for r in rows if of_type(r["errors"])]
    fixed = sum(len(of_type(r["fixed"])) for r in rows)
    missed = sum(len(of_type(r["missed"])) for r in rows)
    false_positives = sum(len(r["false_positives"]) for r in rows)
    clean = [r for r in rows if r["clean"]]
    latencies = [r["latency_s"] for r in rows]
    timed = [r for r in rows if r["completion_tokens"]]
    return {
        "examples": len(rows),
        "words": sum(r["words"] for r in rows),
        "exact_match_rate": ratio(sum(r["exact"] for r in rows), len(rows)),
        "errors": fixed + missed,
        "errors_fixed": fixed,
        "errors_missed": missed,
        "recall": ratio(fixed, fixed + missed),
        "false_positives": false_positives,
        "false_positives_per_100_words": ratio(100 * false_positives, sum(r["words"] for r in rows)),
        "clean_sentences": len(clean),
        "clean_untouched_rate": ratio(sum(r["output"] == r["input"] for r in clean), len(clean)),
        "rewrites": sum(len(r["rewrites"]) for r in rows),
        "latency_mean_s": ratio(sum(latencies), len(latencies)),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "tokens_per_sec": ratio(
            sum(r["completion_tokens"] for r in timed), sum(r["latency_s"] for r in timed)
        ),
    }


def compute_metrics(rows: list[dict]) -> dict:
    langs = sorted({r["lang"] for r in rows})
    types = sorted({e["type"] for r in rows for e in r["errors"]})
    return {
        "overall": summarize(rows),
        "by_lang": {lang: summarize([r for r in rows if r["lang"] == lang]) for lang in langs},
        "by_error_type": {t: summarize(rows, error_type=t) for t in types},
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

COLUMNS = (  # header, metric key, format
    ("n", "examples", "{:d}"),
    ("exact%", "exact_match_rate", "{:.0%}"),
    ("recall%", "recall", "{:.0%}"),
    ("fixed", "errors_fixed", "{:d}"),
    ("missed", "errors_missed", "{:d}"),
    ("FP", "false_positives", "{:d}"),
    ("FP/100w", "false_positives_per_100_words", "{:.2f}"),
    ("clean ok%", "clean_untouched_rate", "{:.0%}"),
    ("rewr", "rewrites", "{:d}"),
    ("lat mean", "latency_mean_s", "{:.2f}s"),
    ("p50", "latency_p50_s", "{:.2f}s"),
    ("p95", "latency_p95_s", "{:.2f}s"),
    ("tok/s", "tokens_per_sec", "{:.1f}"),
)


def format_table(metrics: dict) -> str:
    groups = [("overall", metrics["overall"])]
    groups += [(f"lang:{k}", v) for k, v in metrics["by_lang"].items()]
    groups += [(f"type:{k}", v) for k, v in metrics["by_error_type"].items()]
    table = [["group", *(header for header, _, _ in COLUMNS)]]
    for name, m in groups:
        cells = [fmt.format(m[key]) if m[key] is not None else "-" for _, key, fmt in COLUMNS]
        table.append([name, *cells])
    widths = [max(len(row[c]) for row in table) for c in range(len(table[0]))]
    lines = [
        "  ".join(cell.ljust(w) if c == 0 else cell.rjust(w) for c, (cell, w) in enumerate(zip(row, widths)))
        for row in table
    ]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_results(out_dir: Path, rows: list[dict], metrics: dict, run_info: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"run": run_info, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "outputs.jsonl", rows)
    write_jsonl(out_dir / "failures.jsonl", [r for r in rows if not r["exact"]])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval", type=Path, default=ROOT / "data" / "eval.jsonl")
    p.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    p.add_argument("--prompt", type=Path, default=ROOT / "prompts" / "v1.txt")
    p.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    p.add_argument("--label", default="baseline", help="suffix of the results folder")
    p.add_argument("--model", default=None, help="model name to send (llama.cpp ignores it)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=300, help="seconds per request")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p.parse_args(argv)


def run(examples: list[dict], prompt: list[dict], args) -> tuple[list[dict], dict]:
    """Send every example, sequentially. Returns (scored rows, run info)."""
    rows, info = [], {"model": None, "thinking_detected": False}
    for n, example in enumerate(examples, 1):
        reply = correct(example["input"], prompt, args)
        if reply["thinking"] and not info["thinking_detected"]:
            print(THINKING_WARNING, file=sys.stderr)
        info["thinking_detected"] |= reply["thinking"]
        info["model"] = reply["model"] or info["model"]
        row = score_example(example, reply)
        rows.append(row)
        status = "ok  " if row["exact"] else "CUT " if row["truncated"] else "FAIL"
        print(f"[{n}/{len(examples)}] {status} {row['id']}  {row['latency_s']:.2f}s", file=sys.stderr)
    return rows, info


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.eval.exists():
        print(f"ERROR: {args.eval} not found. Run make_eval.py first.", file=sys.stderr)
        return 2
    examples = read_jsonl(args.eval, args.limit)
    prompt = load_prompt(args.prompt)
    started = datetime.now()
    try:
        rows, info = run(examples, prompt, args)
    except ServerError as e:
        print(f"ERROR: {e}\n  url: {args.url}\n  Is llama-server running? See README.md.", file=sys.stderr)
        return 2
    metrics = compute_metrics(rows)
    label = re.sub(r"[^\w.-]+", "-", args.label)
    out_dir = args.results_dir / f"{started:%Y%m%d-%H%M%S}_{label}"
    run_info = {
        "label": args.label,
        "started": started.isoformat(timespec="seconds"),
        "eval": str(args.eval),
        "prompt": str(args.prompt),
        "url": args.url,
        "limit": args.limit,
        **info,
        "truncated_answers": sum(r["truncated"] for r in rows),
    }
    save_results(out_dir, rows, metrics, run_info)
    print()
    print(format_table(metrics))
    if info["thinking_detected"]:
        print("\n" + THINKING_WARNING)
    truncated = sum(r["truncated"] for r in rows)
    if truncated:
        print(f"\nWARNING: {truncated} answer(s) hit --max-tokens ({args.max_tokens}) and were cut off (marked CUT above).")
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
