#!/usr/bin/env python3
"""Rebuild the exact train/val/test split file the notebooks load.

The notebooks never recompute the split: `create_or_load_split()` loads
`data_split_v1.json` if it exists. The original file stored absolute paths on
the author's machine, so the repository ships only the patient IDs, in their
original order (`data/split_v1_patient_ids.json`). This script turns those IDs
back into the file the notebooks expect, pointing at YOUR copy of BraTS 2020.

Order is preserved on purpose: the training loader's shuffle is seeded, so the
order of the list is part of what makes a run reproducible.

    python tools/make_split_file.py \
        --brats-root  "D:/MedSSL/MICCAI_BraTS2020_TrainingData" \
        --out         "D:/MedSSL/MedSSL_Results/data_split_v1.json"

The script checks that every referenced file exists and that the training-ID
hash matches the one recorded in the paper's provenance records.
"""
import argparse, hashlib, json, os, sys

EXPECTED_TRAIN_SHA256 = "f84bfa95b09c863ff3fb97ece7f901f2370941a78c6d273f4968eb4c36040aa7"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-root", required=True,
                    help="folder that contains BraTS20_Training_001 ... BraTS20_Training_369")
    ap.add_argument("--out", required=True, help="where to write data_split_v1.json")
    ap.add_argument("--ids", default=os.path.join(HERE, "..", "data", "split_v1_patient_ids.json"))
    a = ap.parse_args()

    ids = json.load(open(a.ids))
    digest = hashlib.sha256("\n".join(sorted(set(ids["train"]))).encode("utf-8")).hexdigest()
    if digest != EXPECTED_TRAIN_SHA256:
        sys.exit(f"train-ID hash mismatch: {digest}")

    out = {k: ids[k] for k in ("seed", "train_fraction", "val_fraction", "test_fraction")}
    missing = []
    for part in ("train", "val", "test"):
        pairs = []
        for pid in ids[part]:
            img = os.path.join(a.brats_root, pid, f"{pid}_flair.nii")
            seg = os.path.join(a.brats_root, pid, f"{pid}_seg.nii")
            missing += [p for p in (img, seg) if not os.path.isfile(p)]
            pairs.append([img, seg])
        out[part] = pairs

    if missing:
        print(f"warning: {len(missing)} files not found, e.g. {missing[0]}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: train {len(out['train'])}, val {len(out['val'])}, "
          f"test {len(out['test'])}; train-ID hash OK")


if __name__ == "__main__":
    main()
