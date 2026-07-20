#!/usr/bin/env python3
"""
Gap Classifier: Classify required stats/graphs as EXISTS/REPRODUCIBLE/FIXABLE/TRUE_GAP.
"""

import subprocess
import json
import sys
from pathlib import Path

def classify_graph(repo_path, canonical_root, graph_name):
    """Classify a required graph/stat."""

    # Check for existing PNG
    outputs_dirs = list(Path(canonical_root).glob("**/outputs*"))
    existing_png = None
    for d in outputs_dirs:
        for png in d.glob("**/*.png"):
            if graph_name.lower() in png.name.lower():
                existing_png = str(png)
                break

    if existing_png:
        return {
            "status": "EXISTS-ON-DISK",
            "graph": graph_name,
            "artifact": existing_png
        }

    # Check for plotting function
    search_terms = [
        f"plot_{graph_name}",
        f"def.*{graph_name}",
        graph_name.lower()
    ]

    func_found = None
    for term in search_terms:
        try:
            result = subprocess.run(
                ["grep", "-rn", f"def.*{term}", str(canonical_root)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                func_found = result.stdout.strip().split('\n')[0]
                break
        except:
            pass

    if not func_found:
        return {
            "status": "TRUE_GAP",
            "graph": graph_name,
            "reason": "No plotting function found"
        }

    # Check if function calls savefig
    try:
        result = subprocess.run(
            ["grep", "-A5", func_found.split(":")[1], str(canonical_root)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "savefig" in result.stdout:
            return {
                "status": "REPRODUCIBLE-BY-RERUN",
                "graph": graph_name,
                "function": func_found.split("def ")[-1].split("(")[0] if "def" in func_found else "unknown",
                "file": func_found.split(":")[0],
                "line": func_found.split(":")[1]
            }
        else:
            return {
                "status": "FIXABLE-GAP-ADD-SAVEFIG",
                "graph": graph_name,
                "function": func_found.split("def ")[-1].split("(")[0] if "def" in func_found else "unknown",
                "file": func_found.split(":")[0],
                "line": func_found.split(":")[1]
            }
    except:
        return {
            "status": "FIXABLE-GAP-ADD-SAVEFIG",
            "graph": graph_name,
            "function": "unknown",
            "reason": "Could not verify savefig"
        }

def main():
    if len(sys.argv) < 4:
        print("Usage: gap_classifier.py <repo_path> <canonical_root> <graph_name>")
        sys.exit(1)

    repo_path = sys.argv[1]
    canonical_root = sys.argv[2]
    graph_name = sys.argv[3]

    result = classify_graph(repo_path, canonical_root, graph_name)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()