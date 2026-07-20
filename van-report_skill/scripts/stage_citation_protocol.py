#!/usr/bin/env python3
"""
Stage Citation Protocol: Find function definitions and trace call reachability.

Given a repo path, canonical tree root, and stage keyword, locate the function(s)
implementing that stage, find all call sites, and determine which one is reachable
from the driver's main() entry point.

Returns: (function_name, file, line_start, line_end, verification_cmd, verification_output)
"""

import subprocess
import sys
import json
import re
from pathlib import Path

def run_grep(pattern, search_root, include_pattern="*.py"):
    """Run ripgrep and return (filename, line_number, matched_text) tuples."""
    cmd = [
        "grep", "-rn", pattern,
        "--include", include_pattern,
        str(search_root)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return lines
    except Exception as e:
        print(f"[ERROR] grep failed: {e}", file=sys.stderr)
        return []

def find_function_definitions(repo_path, canonical_root, stage_keyword):
    """Find all function definitions that might implement the stage."""
    # Common naming patterns for each stage
    patterns = {
        "Triangulation": [r"def.*triangulat", r"triangulat"],
        "RANSAC": [r"def.*ransac", r"ransac", r"solvePnP"],
        "PnP trajectory": [r"def.*build_data", r"def.*trajectory"],
        "Database definition": [r"class.*TrackingDB", r"class.*Observation"],
        "Add frame to database": [r"def.*add_feature", r"def.*update_tracks"],
        "Bundle adjustment": [r"def.*bundle", r"def.*solve_bundle"],
        "Relative transformation": [r"def.*relative.*pose", r"def.*compute.*relative"],
        "Pose graph": [r"def.*pose_graph", r"def.*build.*graph", r"def.*optimize_pose"],
        "Loop closure detection": [r"def.*loop_closure", r"def.*detect_loop"],
        "Loop closure factor": [r"def.*loop.*factor", r"def.*add_loop_closure"],
    }

    search_patterns = patterns.get(stage_keyword, [stage_keyword.lower()])
    candidates = {}

    for pattern in search_patterns:
        lines = run_grep(f"def.*{pattern}", canonical_root)
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(":", 2)
            if len(parts) >= 3:
                filename, linenum, code = parts[0], parts[1], parts[2]
                func_match = re.search(r"def\s+(\w+)", code)
                if func_match:
                    func_name = func_match.group(1)
                    key = f"{filename}:{func_name}"
                    if key not in candidates:
                        candidates[key] = {
                            "file": filename,
                            "func_name": func_name,
                            "line_def": int(linenum),
                            "code_snippet": code.strip()
                        }

    return candidates

def find_call_sites(repo_path, func_name, canonical_root):
    """Find all places where func_name is called."""
    pattern = rf"\b{func_name}\s*\("
    lines = run_grep(pattern, canonical_root)
    call_sites = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) >= 3:
            filename, linenum, code = parts[0], parts[1], parts[2]
            call_sites.append({
                "file": filename,
                "line": int(linenum),
                "code": code.strip()
            })
    return call_sites

def trace_from_entry_point(repo_path, canonical_root, target_func):
    """Trace whether target_func is reachable from main()."""
    # This is a heuristic: check if target_func appears in the call chain from main.py
    main_py = Path(canonical_root) / "main.py"
    if not main_py.exists():
        # Try to find main.py-like entry point
        entry_points = list(Path(canonical_root).glob("**/main.py")) + \
                      list(Path(canonical_root).glob("**/run.py")) + \
                      list(Path(canonical_root).glob("**/driver.py"))
        if not entry_points:
            return False, "No main.py entry point found"
        main_py = entry_points[0]

    try:
        with open(main_py, 'r') as f:
            content = f.read()
        # Simple heuristic: check if target_func is mentioned in the file
        # More sophisticated: build call graph, but this is a start
        if re.search(rf"\b{target_func}\s*\(", content):
            return True, f"Found call to {target_func} in {main_py}"
        else:
            return False, f"No call to {target_func} found in {main_py}"
    except Exception as e:
        return False, f"Could not trace: {e}"

def main():
    if len(sys.argv) < 4:
        print("Usage: stage_citation_protocol.py <repo_path> <canonical_root> <stage_keyword>")
        sys.exit(1)

    repo_path = Path(sys.argv[1])
    canonical_root = Path(sys.argv[2])
    stage_keyword = sys.argv[3]

    if not canonical_root.exists():
        print(f"[ERROR] Canonical root not found: {canonical_root}", file=sys.stderr)
        sys.exit(1)

    # Find candidates
    candidates = find_function_definitions(repo_path, canonical_root, stage_keyword)

    if not candidates:
        print(json.dumps({
            "status": "TRUE_GAP",
            "stage": stage_keyword,
            "reason": "No function definition found matching stage keyword",
            "search_patterns": stage_keyword
        }))
        sys.exit(0)

    # For each candidate, find call sites and trace reachability
    results = []
    for key, candidate in candidates.items():
        call_sites = find_call_sites(repo_path, candidate["func_name"], canonical_root)
        reachable, trace_msg = trace_from_entry_point(repo_path, canonical_root, candidate["func_name"])

        results.append({
            "function": candidate["func_name"],
            "file": candidate["file"],
            "line": candidate["line_def"],
            "reachable": reachable,
            "trace_msg": trace_msg,
            "call_count": len(call_sites),
            "call_sites": call_sites[:3]  # First 3 call sites
        })

    # Sort by reachability (reachable first) and call count (more calls = more likely)
    results.sort(key=lambda x: (-x["reachable"], -x["call_count"]))

    # Return the top candidate
    if results:
        top = results[0]
        print(json.dumps({
            "status": "VERIFIED" if top["reachable"] else "PLAUSIBLE",
            "stage": stage_keyword,
            "function": top["function"],
            "file": top["file"],
            "line": top["line"],
            "reachable": top["reachable"],
            "trace_msg": top["trace_msg"],
            "all_candidates": results
        }))
    else:
        print(json.dumps({
            "status": "TRUE_GAP",
            "stage": stage_keyword,
            "reason": "No function candidates found"
        }))

if __name__ == "__main__":
    main()