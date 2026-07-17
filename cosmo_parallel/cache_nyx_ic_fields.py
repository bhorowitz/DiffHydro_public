#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yt


def main() -> None:
    p = argparse.ArgumentParser(description="Cache Nyx gas IC fields into a compact npz for runtimes without yt.")
    p.add_argument("--plotfile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ds = yt.load(str(args.plotfile.resolve()), hint="NyxDataset")
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    payload = {
        "density": np.asarray(cg[("boxlib", "density")], dtype=np.float32),
        "xmom": np.asarray(cg[("boxlib", "xmom")], dtype=np.float32),
        "ymom": np.asarray(cg[("boxlib", "ymom")], dtype=np.float32),
        "zmom": np.asarray(cg[("boxlib", "zmom")], dtype=np.float32),
        "temp_k": np.asarray(cg[("boxlib", "Temp")], dtype=np.float32),
        "box_size": np.asarray(float(ds.domain_width[0].to_value("code_length")), dtype=np.float32),
        "n_grid": np.asarray(int(ds.domain_dimensions[0]), dtype=np.int32),
        "a_init": np.asarray(float((args.plotfile / "comoving_a").read_text().strip()), dtype=np.float32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
