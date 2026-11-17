"""Flat, memory-mappable slice bank and organ-balanced training stream."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

ORGANS = (
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
)
PROMPTS = {
    "spleen": "Segment the spleen.",
    "kidney_right": "Segment the right kidney.",
    "kidney_left": "Segment the left kidney.",
    "gallbladder": "Segment the gallbladder.",
    "liver": "Segment the liver.",
    "stomach": "Segment the stomach.",
    "pancreas": "Segment the pancreas.",
}


def pack(cache, output, split):
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((cache / "index.json").read_text())
    if tuple(index["organs"]) != ORGANS:
        raise ValueError("Unexpected organ order in slice cache")
    patients = [p for p in index["patients"] if p["split"] == split]
    total = sum(p["slices"] * p["hw"][0] * p["hw"][1] for p in patients)
    ct = np.lib.format.open_memmap(output / f"{split}_ct.npy", "w+", np.int16, (total,))
    lab = np.lib.format.open_memmap(
        output / f"{split}_lab.npy", "w+", np.uint8, (total,)
    )
    rows, offset = [], 0
    for p in patients:
        data = np.load(cache / p["file"])
        h, w = p["hw"]
        if data["ct"].shape != (p["slices"], h, w):
            raise ValueError(f"{p['subject_id']}: cached shape differs from index")
        n = p["slices"] * h * w
        ct[offset : offset + n] = data["ct"].reshape(-1)
        lab[offset : offset + n] = data["lab"].reshape(-1)
        for z in range(p["slices"]):
            bits = sum(1 << k for k, flag in enumerate(p["present"][z]) if flag)
            rows.append(
                {
                    "subject_id": p["subject_id"],
                    "z": z,
                    "h": h,
                    "w": w,
                    "offset": offset + z * h * w,
                    "present": bits,
                    "spacing_yx_mm": p["spacing_yx_mm"],
                    "spacing_z_mm": p.get("spacing_z_mm"),
                }
            )
        offset += n
    ct.flush()
    lab.flush()
    (output / f"{split}_slices.json").write_text(json.dumps(rows))
    return {
        "split": split,
        "patients": len(patients),
        "slices": len(rows),
        "pixels": total,
    }


class SliceBank:
    """Memory-mappable slice bank with lazy array opening so label-only evaluation never opens _ct.npy."""

    def __init__(self, folder, split):
        self.folder = Path(folder)
        self.split = split
        self._ct = None
        self._lab = None
        self.slices = json.loads((self.folder / f"{split}_slices.json").read_text())

    @property
    def ct(self):
        if self._ct is None:
            self._ct = np.load(self.folder / f"{self.split}_ct.npy", mmap_mode="r")
        return self._ct

    @property
    def lab(self):
        if self._lab is None:
            self._lab = np.load(self.folder / f"{self.split}_lab.npy", mmap_mode="r")
        return self._lab

    def image(self, i):
        r = self.slices[i]
        raw = self.ct[r["offset"] : r["offset"] + r["h"] * r["w"]].reshape(
            r["h"], r["w"]
        )
        return (raw.astype(np.float32) + 160.0) / 400.0

    def image_stack(self, i, deltas=(-2, -1, 0, 1, 2)):
        """Return multi-slice 2.5D stack (len(deltas), H, W) clamped within the same patient."""
        r0 = self.slices[i]
        sid = r0["subject_id"]
        n_total = len(self.slices)
        planes = []
        for dz in deltas:
            j = min(max(i + dz, 0), n_total - 1)
            if self.slices[j]["subject_id"] != sid:
                step = 1 if dz < 0 else -1
                while j != i and self.slices[j]["subject_id"] != sid:
                    j += step
            planes.append(self.image(j))
        return np.stack(planes, axis=0)

    def mask(self, i, organ):
        r = self.slices[i]
        raw = self.lab[r["offset"] : r["offset"] + r["h"] * r["w"]].reshape(
            r["h"], r["w"]
        )
        return ((raw >> ORGANS.index(organ)) & 1) == 1

    def present(self, i, organ):
        return bool((self.slices[i]["present"] >> ORGANS.index(organ)) & 1)


class BalancedRequests(torch.utils.data.IterableDataset):
    """Organ-balanced training stream; half positive, half target-absent (half within `near` slices of organ tips)."""

    def __init__(
        self,
        bank,
        seed,
        rank=0,
        positive_fraction=0.5,
        near=10,
        multi_slice=False,
        hard_organs=False,
    ):
        self.bank = bank
        self.seed = seed
        self.rank = rank
        self.positive_fraction = positive_fraction
        self.multi_slice = multi_slice
        self.organ_pool = (
            ORGANS
            + (
                "pancreas",
                "pancreas",
                "gallbladder",
                "gallbladder",
                "stomach",
                "kidney_left",
                "kidney_right",
            )
            if hard_organs
            else ORGANS
        )
        self.positive = {o: [] for o in ORGANS}
        self.empty = {o: [] for o in ORGANS}
        self.near = {o: [] for o in ORGANS}
        by_subject = {}
        for i, r in enumerate(bank.slices):
            by_subject.setdefault(r["subject_id"], []).append(i)
            for k, organ in enumerate(ORGANS):
                (self.positive if (r["present"] >> k) & 1 else self.empty)[organ].append(i)
        for rows in by_subject.values():
            rows = sorted(rows, key=lambda i: bank.slices[i]["z"])
            for k, organ in enumerate(ORGANS):
                zs = [bank.slices[i]["z"] for i in rows if (bank.slices[i]["present"] >> k) & 1]
                if not zs:
                    continue
                lo, hi = min(zs) - near, max(zs) + near
                self.near[organ].extend(
                    i
                    for i in rows
                    if lo <= bank.slices[i]["z"] <= hi
                    and not ((bank.slices[i]["present"] >> k) & 1)
                )

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        rng = random.Random(self.seed * 1000003 + self.rank * 1009 + worker)
        while True:
            organ = rng.choice(self.organ_pool)
            if rng.random() < self.positive_fraction:
                pool = self.positive
            else:
                pool = self.near if self.near[organ] and rng.random() < 0.5 else self.empty
            i = rng.choice(pool[organ])
            item = {
                "slice": i,
                "organ": organ,
                "image": self.bank.image(i),
                "mask": self.bank.mask(i, organ),
            }
            if self.multi_slice:
                item["stack"] = self.bank.image_stack(i)
            yield item


def main():
    parser = argparse.ArgumentParser(description="Pack per-patient slice cache into flat memmaps.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    for split in args.splits.split(","):
        print(json.dumps(pack(args.cache, args.output, split)), flush=True)


if __name__ == "__main__":
    main()
