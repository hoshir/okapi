#!/usr/bin/env python3
"""
scripts/ab_compare.py - Paired A/B benchmark harness for deep Zebra positions.

Implements ticket BENCH-001:
- Evaluates candidate builds A and B interleaved (A,B,A,B,... alternating order each pair).
- Evaluates on specified FFO positions loaded from scripts/eval_candidate.py.
- Captures stdout + stderr from scrzebra run with PASS_DIAG=1.
- Parses Total time, Total nodes, exact pass line, selective pass lines, and score/move.
- Enforces load average guards before each run.
- Outputs a markdown table and summary, and optionally writes full raw data to JSON.
"""

import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def load_ffo_positions() -> List[Tuple]:
    """
    Load FFO positions from scripts/eval_candidate.py without duplicating the table.
    """
    # Attempt 1: direct import
    try:
        from scripts.eval_candidate import FFO_POSITIONS
        return FFO_POSITIONS
    except ImportError:
        pass

    try:
        from eval_candidate import FFO_POSITIONS
        return FFO_POSITIONS
    except ImportError:
        pass

    # Attempt 2: importlib from file location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_path = os.path.join(script_dir, "eval_candidate.py")
    if os.path.isfile(candidate_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("eval_candidate", candidate_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "FFO_POSITIONS"):
                return mod.FFO_POSITIONS

    # Attempt 3: parse ast literal if import fails
    if os.path.isfile(candidate_path):
        with open(candidate_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"FFO_POSITIONS\s*=\s*(\[.*?\])\n\n", content, re.DOTALL)
        if m:
            return ast.literal_eval(m.group(1))

    raise RuntimeError("Could not load FFO_POSITIONS from eval_candidate.py")


def resolve_position(pos_arg: Any, positions: List[Tuple]) -> Tuple:
    """
    Resolve position argument (e.g. 40, "40", "FFO #40", "ffo 40") to a position tuple.
    Tuple format: (name, board_string, expected_black, expected_white, list_of_valid_first_moves)
    """
    s = str(pos_arg).strip()
    digits = re.search(r"(\d+)", s)
    if digits:
        target_name = f"FFO #{int(digits.group(1))}"
        for item in positions:
            if item[0] == target_name:
                return item
    for item in positions:
        if item[0].lower() == s.lower():
            return item
    available = [item[0] for item in positions]
    raise ValueError(f"Position '{pos_arg}' not found in FFO_POSITIONS. Available: {available}")


def resolve_bin_dir(repo_dir: str) -> str:
    """
    Find build/bin directory containing scrzebra executable.
    """
    abs_path = os.path.abspath(repo_dir)
    build_bin = os.path.join(abs_path, "build", "bin")
    if os.path.isfile(os.path.join(build_bin, "scrzebra")):
        return build_bin
    if os.path.isfile(os.path.join(abs_path, "scrzebra")):
        return abs_path
    raise FileNotFoundError(
        f"scrzebra binary not found in {build_bin} or {abs_path}. Build it first."
    )


def parse_env_args(env_arg_list: List[Any]) -> Dict[str, str]:
    """
    Parse repeatable or multi-argument --env-a / --env-b flags into a dictionary.
    Supports '--env-a K=V', '--env-a K1=V1 K2=V2', '--env-a K1=V1 --env-a K2=V2'.
    """
    res = {}
    if not env_arg_list:
        return res
    for item in env_arg_list:
        if isinstance(item, list):
            for entry in item:
                if "=" in entry:
                    k, v = entry.split("=", 1)
                    res[k] = v
        elif isinstance(item, str):
            if "=" in item:
                k, v = entry.split("=", 1)
                res[k] = v
    return res


def check_load_guard(max_load: float, max_wait: float = 300.0, step: float = 10.0) -> float:
    """
    Before each run, read os.getloadavg()[0].
    If above max_load, wait in 10s steps (max 5 min) and log it to stderr.
    Returns the load average right before running.
    """
    if max_load is None or max_load <= 0:
        return os.getloadavg()[0]

    start_time = time.time()
    logged_wait = False
    while True:
        load = os.getloadavg()[0]
        if load <= max_load:
            if logged_wait:
                sys.stderr.write(f"Load average dropped to {load:.2f} <= {max_load:.2f}. Proceeding.\n")
                sys.stderr.flush()
            return load

        elapsed = time.time() - start_time
        if elapsed >= max_wait:
            sys.stderr.write(
                f"Warning: Load average ({load:.2f}) remained above --max-load ({max_load:.2f}) "
                f"after waiting {int(elapsed)}s. Proceeding.\n"
            )
            sys.stderr.flush()
            return load

        logged_wait = True
        sys.stderr.write(
            f"Load average {load:.2f} > max-load {max_load:.2f}. "
            f"Waiting {int(step)}s (elapsed {int(elapsed)}s/{int(max_wait)}s)...\n"
        )
        sys.stderr.flush()
        time.sleep(step)


def run_scrzebra(
    bin_dir: str,
    board_str: str,
    threads: int,
    hash_bits: int,
    extra_env: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run scrzebra in bin_dir:
    cd <dir>/build/bin && echo "<board>" | PASS_DIAG=1 [env] ./scrzebra -n T -h HB -e 0 -line 1 -script /dev/stdin /dev/stdout
    Parses Total time, Total nodes, exact pass line, selective pass lines, and result line.
    """
    cmd = [
        "./scrzebra",
        "-n", str(threads),
        "-h", str(hash_bits),
        "-e", "0",
        "-line", "1",
        "-script", "/dev/stdin",
        "/dev/stdout",
    ]
    env = os.environ.copy()
    env["PASS_DIAG"] = "1"
    for k, v in extra_env.items():
        env[k] = v

    proc = subprocess.Popen(
        cmd,
        cwd=bin_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout, stderr = proc.communicate(input=board_str + "\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"scrzebra exited with code {proc.returncode} in {bin_dir}.\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
        )

    # 1. Parse Total time: e.g. "Total time:               0.4 s"
    m_time = re.search(r"Total time:\s+([0-9.]+)\s*s", stdout)
    if not m_time:
        raise ValueError(f"Could not parse 'Total time' from scrzebra output:\n{stdout}")
    total_time = float(m_time.group(1))

    # 2. Parse Total nodes: e.g. "Total nodes:              12866259"
    m_nodes = re.search(r"Total nodes:\s+([0-9]+)", stdout)
    if not m_nodes:
        raise ValueError(f"Could not parse 'Total nodes' from scrzebra output:\n{stdout}")
    total_nodes = int(m_nodes.group(1))

    # 3. Parse exact pass line from PASS_DIAG (typically on stderr):
    # PASS_DIAG: empties=20 pass=exact sel=0 initial_center=38 final_eval=38 researches=0 nodes=7080333 time=0.220s
    combined = stdout + "\n" + stderr
    m_exact = re.search(r"PASS_DIAG:.*?pass=exact.*?nodes=([0-9.]+)\s+time=([0-9.]+)s", combined)
    if m_exact:
        exact_nodes = int(float(m_exact.group(1)))
        exact_time = float(m_exact.group(2))
    else:
        exact_nodes = None
        exact_time = None

    # 4. Parse selective pass lines:
    # PASS_DIAG: empties=20 pass=0 sel=6 center_before=18 center_after=38 eval=38 steps=10 nodes=2604387 time=0.086s
    sel_matches = list(re.finditer(r"PASS_DIAG:.*?pass=(\d+).*?nodes=([0-9.]+)\s+time=([0-9.]+)s", combined))
    selective_nodes = sum(int(float(m.group(2))) for m in sel_matches)
    selective_time = round(sum(float(m.group(3)) for m in sel_matches), 4)

    # 5. Parse result line (score and move):
    # e.g.: "51 - 13    a2 b1 c1 -- b6 c7 a7 b7 b8 d7 f8 c6 f7"
    score = None
    move = None
    for line in stdout.splitlines():
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("%") or line_clean.startswith("scrZebra"):
            continue
        m_res = re.match(r"^(\d+)\s*-\s*(\d+)(?:\s+([a-h][1-8]|--|\S+))?", line_clean)
        if m_res:
            score = f"{int(m_res.group(1))}-{int(m_res.group(2))}"
            move = m_res.group(3) if m_res.group(3) else ""
            break

    return {
        "total_time": total_time,
        "total_nodes": total_nodes,
        "exact_time": exact_time,
        "exact_nodes": exact_nodes,
        "selective_time": selective_time,
        "selective_nodes": selective_nodes,
        "score": score,
        "move": move,
        "stdout": stdout,
        "stderr": stderr,
    }


def format_markdown_table(runs: List[Dict[str, Any]]) -> str:
    """
    Format markdown table per run (pair, arm, total time, exact time, selective time, total nodes, load).
    """
    lines = [
        "| Pair | Arm | Total Time (s) | Exact Time (s) | Selective Time (s) | Total Nodes | Load |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in runs:
        tot_t = f"{r['total_time']:.3f}" if r["total_time"] is not None else "N/A"
        ex_t = f"{r['exact_time']:.3f}" if r["exact_time"] is not None else "N/A"
        sel_t = f"{r['selective_time']:.3f}" if r["selective_time"] is not None else "N/A"
        tot_n = f"{r['total_nodes']}" if r["total_nodes"] is not None else "N/A"
        load_s = f"{r['load']:.2f}"
        lines.append(f"| {r['pair']} | {r['arm']} | {tot_t} | {ex_t} | {sel_t} | {tot_n} | {load_s} |")
    return "\n".join(lines)


def format_summary(
    runs: List[Dict[str, Any]],
    pairs_data: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute summary: median and min/max of A and B, median of per-pair ratio B/A - 1 in %,
    and whether scores agree.
    """
    runs_a = [r for r in runs if r["arm"] == "A"]
    runs_b = [r for r in runs if r["arm"] == "B"]

    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    stats_a = {
        "total_time": calc_stats([r["total_time"] for r in runs_a if r["total_time"] is not None]),
        "exact_time": calc_stats([r["exact_time"] for r in runs_a if r["exact_time"] is not None]),
        "selective_time": calc_stats([r["selective_time"] for r in runs_a if r["selective_time"] is not None]),
        "total_nodes": calc_stats([r["total_nodes"] for r in runs_a if r["total_nodes"] is not None]),
    }
    stats_b = {
        "total_time": calc_stats([r["total_time"] for r in runs_b if r["total_time"] is not None]),
        "exact_time": calc_stats([r["exact_time"] for r in runs_b if r["exact_time"] is not None]),
        "selective_time": calc_stats([r["selective_time"] for r in runs_b if r["selective_time"] is not None]),
        "total_nodes": calc_stats([r["total_nodes"] for r in runs_b if r["total_nodes"] is not None]),
    }

    per_pair_ratios = [p["ratio_b_over_a_pct"] for p in pairs_data]
    median_pair_ratio = statistics.median(per_pair_ratios) if per_pair_ratios else 0.0

    all_scores = [r["score"] for r in runs]
    all_moves = [r["move"] for r in runs]
    scores_agree = (len(set(all_scores)) == 1) and (len(set(all_moves)) == 1)

    if scores_agree:
        score_desc = f"YES ({all_scores[0]}, move: {all_moves[0]})"
    else:
        score_desc = f"NO (scores: {all_scores}, moves: {all_moves})"

    summary_md = [
        "### Summary",
        f"- **Arm A Total Time**: median {stats_a['total_time']['median']:.3f}s (min {stats_a['total_time']['min']:.3f}s, max {stats_a['total_time']['max']:.3f}s)",
        f"- **Arm B Total Time**: median {stats_b['total_time']['median']:.3f}s (min {stats_b['total_time']['min']:.3f}s, max {stats_b['total_time']['max']:.3f}s)",
        f"- **Arm A Exact Time**: median {stats_a['exact_time']['median']:.3f}s (min {stats_a['exact_time']['min']:.3f}s, max {stats_a['exact_time']['max']:.3f}s)",
        f"- **Arm B Exact Time**: median {stats_b['exact_time']['median']:.3f}s (min {stats_b['exact_time']['min']:.3f}s, max {stats_b['exact_time']['max']:.3f}s)",
        f"- **Arm A Selective Time**: median {stats_a['selective_time']['median']:.3f}s (min {stats_a['selective_time']['min']:.3f}s, max {stats_a['selective_time']['max']:.3f}s)",
        f"- **Arm B Selective Time**: median {stats_b['selective_time']['median']:.3f}s (min {stats_b['selective_time']['min']:.3f}s, max {stats_b['selective_time']['max']:.3f}s)",
        f"- **Arm A Total Nodes**: median {int(stats_a['total_nodes']['median'])} (min {int(stats_a['total_nodes']['min'])}, max {int(stats_a['total_nodes']['max'])})",
        f"- **Arm B Total Nodes**: median {int(stats_b['total_nodes']['median'])} (min {int(stats_b['total_nodes']['min'])}, max {int(stats_b['total_nodes']['max'])})",
        f"- **Median Per-Pair Ratio (B/A - 1)**: {median_pair_ratio:+.2f}%",
        f"- **Scores Agree**: {score_desc}",
    ]

    if len(per_pair_ratios) > 1:
        pairs_str = ", ".join(f"Pair {i+1}: {r:+.2f}%" for i, r in enumerate(per_pair_ratios))
        summary_md.append(f"- **Per-Pair Ratios**: {pairs_str}")

    summary_dict = {
        "arm_a": stats_a,
        "arm_b": stats_b,
        "per_pair_ratios_pct": per_pair_ratios,
        "median_pair_ratio_pct": median_pair_ratio,
        "scores_agree": scores_agree,
        "score_desc": score_desc,
    }

    return "\n".join(summary_md), summary_dict


def main():
    parser = argparse.ArgumentParser(
        description="Paired A/B benchmark harness for deep Zebra positions."
    )
    parser.add_argument("--a", required=True, help="Directory for arm A (repo root containing build/bin/scrzebra)")
    parser.add_argument("--b", required=True, help="Directory for arm B (repo root containing build/bin/scrzebra)")
    parser.add_argument("--pos", required=True, help="FFO position number or name (e.g. 40, 55, 'FFO #40')")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--pairs", type=int, default=3, help="Number of interleaved pairs (default: 3)")
    parser.add_argument("--hash-bits", type=int, default=22, help="Hash table bits (default: 22)")
    parser.add_argument("--env-a", action="append", nargs="*", default=[], help="Extra environment variables for arm A as KEY=VALUE")
    parser.add_argument("--env-b", action="append", nargs="*", default=[], help="Extra environment variables for arm B as KEY=VALUE")
    parser.add_argument("--json", default=None, help="Path to write full raw data in JSON format")
    parser.add_argument("--max-load", type=float, default=1.5, help="Maximum 1-min load average before running (default: 1.5)")

    args = parser.parse_args()

    # Load positions
    positions = load_ffo_positions()
    pos_entry = resolve_position(args.pos, positions)
    pos_name, board_str = pos_entry[0], pos_entry[1]

    # Resolve directories
    bin_dir_a = resolve_bin_dir(args.a)
    bin_dir_b = resolve_bin_dir(args.b)

    # Parse environments
    env_a = parse_env_args(args.env_a)
    env_b = parse_env_args(args.env_b)

    # Run interleaved runs
    runs = []
    pairs_data = []

    for p in range(1, args.pairs + 1):
        # Alternate which goes first each pair: A,B then B,A
        arm_order = ["A", "B"] if (p % 2 == 1) else ["B", "A"]
        pair_runs = {}

        for arm in arm_order:
            bin_dir = bin_dir_a if arm == "A" else bin_dir_b
            repo_root = args.a if arm == "A" else args.b
            extra_env = env_a if arm == "A" else env_b

            load = check_load_guard(args.max_load)
            res = run_scrzebra(bin_dir, board_str, args.threads, args.hash_bits, extra_env)

            run_record = {
                "pair": p,
                "arm": arm,
                "dir": os.path.abspath(repo_root),
                "load": round(load, 2),
                "total_time": res["total_time"],
                "exact_time": res["exact_time"],
                "selective_time": res["selective_time"],
                "total_nodes": res["total_nodes"],
                "exact_nodes": res["exact_nodes"],
                "selective_nodes": res["selective_nodes"],
                "score": res["score"],
                "move": res["move"],
                "stdout": res["stdout"],
                "stderr": res["stderr"],
            }
            runs.append(run_record)
            pair_runs[arm] = run_record

        # Per-pair ratio: B/A - 1 in %
        time_a = pair_runs["A"]["total_time"]
        time_b = pair_runs["B"]["total_time"]
        ratio_pct = ((time_b / time_a) - 1.0) * 100.0 if (time_a and time_a > 0) else 0.0

        pairs_data.append({
            "pair": p,
            "order": arm_order,
            "time_a": time_a,
            "time_b": time_b,
            "ratio_b_over_a_pct": ratio_pct,
            "scores_agree": (pair_runs["A"]["score"] == pair_runs["B"]["score"]) and (pair_runs["A"]["move"] == pair_runs["B"]["move"]),
        })

    # Generate markdown table and summary
    table_md = format_markdown_table(runs)
    summary_md, summary_dict = format_summary(runs, pairs_data)

    print(f"## A/B Comparison: {pos_name} ({args.threads} threads, {args.pairs} pairs, hash-bits={args.hash_bits})\n")
    print(table_md)
    print()
    print(summary_md)

    if args.json:
        json_path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        out_data = {
            "benchmark": "ab_compare",
            "position": {
                "name": pos_name,
                "board": board_str,
                "expected_black": pos_entry[2] if len(pos_entry) > 2 else None,
                "expected_white": pos_entry[3] if len(pos_entry) > 3 else None,
                "expected_moves": pos_entry[4] if len(pos_entry) > 4 else None,
            },
            "config": {
                "a": os.path.abspath(args.a),
                "b": os.path.abspath(args.b),
                "pos": args.pos,
                "threads": args.threads,
                "pairs": args.pairs,
                "hash_bits": args.hash_bits,
                "max_load": args.max_load,
                "env_a": env_a,
                "env_b": env_b,
            },
            "runs": runs,
            "pairs": pairs_data,
            "summary": summary_dict,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    main()
