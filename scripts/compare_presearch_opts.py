#!/usr/bin/env python3
"""
scripts/compare_presearch_opts.py - Compare Presearch Options 2-A, 2-B, and 2-C on FFO Endgame Suite.

Options compared:
  2-A: Proof Tree Complexity Selection only (PRESEARCH_OPT=2a)
  2-B: Leaf Solver Dispatch at <= 8 empties only (PRESEARCH_OPT=2b)
  2-C: Both combined (PRESEARCH_OPT=2c)
"""

import os
import sys
import time
import subprocess
import re
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SCRZEBRA = os.path.join(REPO_ROOT, "build", "bin", "scrzebra")

# Read positions from tests/ffotest.scr
def load_ffo_positions():
    ffo_file = os.path.join(REPO_ROOT, "tests", "ffotest.scr")
    positions = []
    with open(ffo_file, "r") as f:
        lines = f.readlines()
    
    current_label = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("% #"):
            # e.g., "% #40, 51-13"
            m = re.match(r"%\s*(#\d+)", line)
            if m:
                current_label = f"FFO {m.group(1)}"
        elif not line.startswith("%") and len(line) >= 66:
            # board string line
            board_str = line
            if current_label:
                positions.append((current_label, board_str))
                current_label = None
    return positions

def solve_one(board_str, opt_name, threads=8, hash_bits=22):
    tmp_scr = os.path.join(REPO_ROOT, "build", "bin", f"tmp_{os.getpid()}.scr")
    tmp_out = os.path.join(REPO_ROOT, "build", "bin", f"tmp_{os.getpid()}.out")
    with open(tmp_scr, "w") as f:
        f.write(board_str + "\n")
    
    env = os.environ.copy()
    env["PRESEARCH_OPT"] = opt_name
    env["PRESEARCH_MPC_MARGIN"] = "128"
    
    cmd = [
        SCRZEBRA,
        "-n", str(threads),
        "-h", str(hash_bits),
        "-e", "1",
        "-script", f"tmp_{os.getpid()}.scr",
        f"tmp_{os.getpid()}.out"
    ]
    
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "build", "bin"),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           timeout=400.0)
        dt = time.time() - t0
        stdout = p.stdout
        stderr = p.stderr
    except subprocess.TimeoutExpired as e:
        dt = 400.0
        stdout = e.stdout or ""
        stderr = e.stderr or ""
    
    out_content = ""
    if os.path.exists(tmp_out):
        with open(tmp_out, "r") as f:
            out_content = f.read()
        os.remove(tmp_out)
    if os.path.exists(tmp_scr):
        os.remove(tmp_scr)
    
    # Parse total nodes from stdout/out_content
    # e.g. "Total nodes: 123456"
    nodes = 0
    m_nodes = re.search(r"Total nodes:\s+(\d+)", out_content)
    if m_nodes:
        nodes = int(m_nodes.group(1))
    
    # Parse score from out_content (e.g. "32 - 32" or "51 - 13")
    score_str = ""
    m_score = re.search(r"(\d+\s*-\s*\d+)", out_content)
    if m_score:
        score_str = m_score.group(1).replace(" ", "")
    
    # Parse move from out_content
    # e.g. "PV: g6 ..."
    move_str = ""
    m_pv = re.search(r"PV:\s*([a-h][1-8])", out_content)
    if m_pv:
        move_str = m_pv.group(1)
    
    # Parse telemetry from stderr
    # [POS_TELEMETRY] empties=29 move=g6 eval=0 time=117.90s tt_hits=12.1% (34817373/286649671) tt_cuts=28326772 mpc_cuts=61.5% (206262/335193)
    tt_hit_pct = 0.0
    tt_hits = 0
    tt_probes = 0
    tt_cuts = 0
    mpc_pct = 0.0
    m_tel = re.search(r"\[POS_TELEMETRY\].*?tt_hits=([\d\.]+)%\s*\((\d+)/(\d+)\)\s*tt_cuts=(\d+)\s*mpc_cuts=([\d\.]+)%", stderr)
    if m_tel:
        tt_hit_pct = float(m_tel.group(1))
        tt_hits = int(m_tel.group(2))
        tt_probes = int(m_tel.group(3))
        tt_cuts = int(m_tel.group(4))
        mpc_pct = float(m_tel.group(5))
    
    return {
        "time": dt,
        "nodes": nodes,
        "score": score_str,
        "move": move_str,
        "tt_hit_pct": tt_hit_pct,
        "tt_hits": tt_hits,
        "tt_probes": tt_probes,
        "tt_cuts": tt_cuts,
        "mpc_pct": mpc_pct,
    }

def main():
    positions = load_ffo_positions()
    print(f"Loaded {len(positions)} FFO endgame positions.")
    
    opts = ["2a", "2b", "2c"]
    results = {opt: {} for opt in opts}
    
    header = (
        f"{'Position':<9} | "
        f"{'2-A Time':<9} {'2-A Nodes':<11} {'2-A TT%':<7} | "
        f"{'2-B Time':<9} {'2-B Nodes':<11} {'2-B TT%':<7} | "
        f"{'2-C Time':<9} {'2-C Nodes':<11} {'2-C TT%':<7} | "
        f"{'2-C vs 2-A':<10} {'2-C vs 2-B':<10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    sys.stdout.flush()
    
    tot_time = {opt: 0.0 for opt in opts}
    tot_nodes = {opt: 0 for opt in opts}
    
    for label, board_str in positions:
        pos_res = {}
        for opt in opts:
            res = solve_one(board_str, opt, threads=8, hash_bits=22)
            pos_res[opt] = res
            results[opt][label] = res
            tot_time[opt] += res["time"]
            tot_nodes[opt] += res["nodes"]
        
        # Compute speedups
        t_a = pos_res["2a"]["time"]
        t_b = pos_res["2b"]["time"]
        t_c = pos_res["2c"]["time"]
        
        n_a = pos_res["2a"]["nodes"]
        n_b = pos_res["2b"]["nodes"]
        n_c = pos_res["2c"]["nodes"]
        
        d_ca = f"{(n_c / n_a - 1.0) * 100:+.1f}%" if n_a > 0 else "N/A"
        d_cb = f"{(n_c / n_b - 1.0) * 100:+.1f}%" if n_b > 0 else "N/A"
        
        row = (
            f"{label:<9} | "
            f"{t_a:6.2f}s   {n_a:11,d} {pos_res['2a']['tt_hit_pct']:5.1f}% | "
            f"{t_b:6.2f}s   {n_b:11,d} {pos_res['2b']['tt_hit_pct']:5.1f}% | "
            f"{t_c:6.2f}s   {n_c:11,d} {pos_res['2c']['tt_hit_pct']:5.1f}% | "
            f"{d_ca:<10} {d_cb:<10}"
        )
        print(row)
        sys.stdout.flush()
    
    print("=" * len(header))
    tot_row = (
        f"{'TOTAL':<9} | "
        f"{tot_time['2a']:6.1f}s   {tot_nodes['2a']:11,d}       | "
        f"{tot_time['2b']:6.1f}s   {tot_nodes['2b']:11,d}       | "
        f"{tot_time['2c']:6.1f}s   {tot_nodes['2c']:11,d}       | "
        f"{(tot_nodes['2c'] / tot_nodes['2a'] - 1.0) * 100:+.1f}%     "
        f"{(tot_nodes['2c'] / tot_nodes['2b'] - 1.0) * 100:+.1f}%"
    )
    print(tot_row)
    print("=" * len(header))
    
    # Save json summary
    summary_path = os.path.join(REPO_ROOT, "build", "ffo_comparison_2a_2b_2c.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw detailed comparison to {summary_path}")

if __name__ == "__main__":
    main()
