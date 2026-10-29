"""Reproduce the residual-reduction experiment without assuming RH.

Example:
  python refinement_scan.py --gram ../Riemann_Continuation_Verification/nb_gram_check.npy \
      --sizes 16 32 64 125 128 250 --output ../Riemann_Continuation_Verification/refinement.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from block_refinement import chain_refinement, polynomial_control, project, refinement
from nb_gram_np import b_vector, build_gram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gram", type=Path, required=True, help="Existing .npy matrix, or new cache path")
    parser.add_argument("--build", type=int, help="Build a new NumPy Gram matrix of this dimension")
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ("block_refinement.py", "refinement_scan.py", "nb_gram_np.py")}
    if args.gram.suffix != ".npy":
        parser.error("--gram must have a .npy suffix")
    if min(args.sizes) < 2:
        parser.error("all sizes must be at least 2")
    if args.build is not None:
        if args.gram.exists():
            parser.error("--build refuses to overwrite an existing matrix")
        if args.build < 2 * max(args.sizes):
            parser.error("--build must be at least twice the largest requested size")
        matrix = build_gram(args.build)
        args.gram.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.gram, matrix)
    else:
        matrix = np.load(args.gram, allow_pickle=False)
    if 2 * max(args.sizes) > len(matrix):
        parser.error("matrix must be at least twice the largest requested size")
    target = b_vector(len(matrix))
    np.testing.assert_allclose(matrix[0, 0], np.log(2*np.pi) - 0.5772156649015329,
                               rtol=0, atol=2e-11)
    rows = []
    for n in sorted(set(args.sizes)):
        row = chain_refinement(matrix, target, n)
        rows.append(row)
        print(json.dumps(row), flush=True)
    controls = []
    for a in (1.0, 2.0):
        for n in (4, 8, 16, 32):
            gram, rhs, exact = polynomial_control(2*n, a)
            calculation = refinement(gram, rhs, n)
            measured = project(gram, rhs).error
            np.testing.assert_allclose(measured, exact, rtol=1e-11, atol=1e-13)
            controls.append({"a": a, "N": n, "error_2N": measured,
                             "exact_error_2N": exact,
                             "eta_logN": calculation["gain"] /
                             calculation["old_projection"].error * float(np.log(n)),
                             "known_limiting_error": 0.0 if a == 1 else 1 - a**-2})
    root = Path(__file__).resolve().parent.parent
    result = {
        "status": "Finite floating-point evidence; no interval certificate or uniform contraction proof",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": source_hashes,
        "gram_path": str(args.gram.resolve()),
        "gram_sha256": hashlib.sha256(args.gram.read_bytes()).hexdigest(),
        "gram_dimension": len(matrix), "rows": rows, "controls": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
