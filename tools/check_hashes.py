"""Check that this repository's files are byte-identical to the originals.

Usage (from the repository root):  python tools/check_hashes.py [repository_root]
Standard library only. Exit code 0 = all checks pass, 1 = at least one failure.

Two checks:
1. Run-time records. During the v0.11 runs, the pipeline itself wrote the SHA-256 of
   its result files into its provenance JSONs. Those recorded values are read from the
   JSONs and compared with the files in results/.
2. Publication manifest. tools/MANIFEST.sha256 lists every file copied from the project
   folder with the SHA-256 of the original. Every copy must match.

Run it on a fresh clone: a line-ending conversion by git would show up as a mismatch.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    res = root / "results"
    failures = 0
    print("repository root:", root, "\n")

    print("1. Hashes recorded by the pipeline at run time")
    prov = json.loads((res / "v0_11_test_provenance.json").read_text(encoding="utf-8"))
    thr = json.loads((res / "v0_11_thresholds_frozen.json").read_text(encoding="utf-8"))
    recorded = [
        ("v0_11_thresholds_frozen.json", prov["thresholds_frozen_sha256"], "v0_11_test_provenance.json"),
        ("v0_11_test_per_patient.csv", prov["per_patient_csv_sha256"], "v0_11_test_provenance.json"),
        ("v0_11_test_report.csv", prov["report_csv_sha256"], "v0_11_test_provenance.json"),
        ("v0_11_val_scores.csv", thr["scores_sha256"], "v0_11_thresholds_frozen.json"),
    ]
    for name, want, source in recorded:
        ok = (res / name).is_file() and sha256(res / name) == want
        failures += 0 if ok else 1
        print("  {}  {:<32} (recorded in {})".format("OK  " if ok else "FAIL", name, source))

    print("\n2. Publication manifest (tools/MANIFEST.sha256)")
    n = 0
    for line in (root / "tools" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        want, rel = line.split(None, 1)
        p = root / rel
        ok = p.is_file() and sha256(p) == want
        n += 1
        if not ok:
            failures += 1
            print("  FAIL  {}".format(rel))
    print("  {} files checked".format(n))

    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else "{} FAILURE(S)".format(failures)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
