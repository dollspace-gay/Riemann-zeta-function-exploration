"""Evaluate the finite Vasyunin Gram formula using the companion C++ library.

This is a floating-point implementation of an exact formula, not a certificate.
The alternative piecewise-integral builder remains in nb_gram_np.py.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def build_gram(size, library):
    if not isinstance(size, int) or not 1 <= size <= 8192:
        raise ValueError("Require an integer size between 1 and 8192")
    backend = ctypes.CDLL(str(Path(library).resolve()))
    function = backend.nb_vasyunin_gram
    function.argtypes = [ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
    function.restype = ctypes.c_int
    matrix = np.empty((size, size), dtype=np.float64)
    status = function(size, matrix)
    if status:
        raise RuntimeError(f"Vasyunin builder failed with status {status}")
    if not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
        raise ArithmeticError("Nonfinite or asymmetric Gram matrix")
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path, help="compare a previously built leading principal matrix")
    args = parser.parse_args()
    started = time.monotonic()
    matrix = build_gram(args.size, args.library)
    report = {"status": "Numerical finite Vasyunin formula; not an interval enclosure",
              "size": args.size, "build_seconds": time.monotonic() - started,
              "source": "https://arxiv.org/pdf/2405.06349", "formula": "(17), (30)",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_suffix('.cpp')]},
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}
    if args.compare:
        old = np.load(args.compare, allow_pickle=False)
        if old.ndim != 2 or old.shape[0] != old.shape[1] or len(old) > args.size:
            raise ValueError("Comparison matrix must be square and no larger than the new matrix")
        error = float(np.max(np.abs(matrix[:len(old), :len(old)] - old)))
        report["comparison"] = {"size": len(old), "max_absolute_difference": error,
                                "matrix_sha256": hashlib.sha256(args.compare.read_bytes()).hexdigest()}
        np.testing.assert_allclose(matrix[:len(old), :len(old)], old, atol=2e-11, rtol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix, allow_pickle=False)
    report["matrix_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
