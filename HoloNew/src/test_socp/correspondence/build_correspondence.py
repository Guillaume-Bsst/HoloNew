"""Build, save and load the fixed human->G1 correspondence.

The .npz embeds the point-cloud cache (tri_idx, bary): the correspondence is valid
only for the exact cache it was built against, so the online consumer must rebuild
HumanBody's PointCloudCache from these arrays rather than resampling.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


@dataclass(frozen=True)
class CorrespondenceTable:
    link_idx: np.ndarray      # (M,)   G1 point's link, index into link_names
    offset_local: np.ndarray  # (M, 3) G1 point in that link's frame
    link_names: list[str]     # (L,)
    human_idx: np.ndarray     # (M,)   human point driving each G1 point (into tri_idx/bary)
    tri_idx: np.ndarray       # (N,)   embedded human cache identity
    bary: np.ndarray          # (N, 3) embedded human cache identity
    density: float
    gender: str
    betas: np.ndarray         # (16,) or (0,)
    g1_rest_cfg: np.ndarray   # (n_actuated,)
    seg: np.ndarray           # (M,)   G1 point segment index (for viz/debug)


def save_correspondence(path, table: CorrespondenceTable) -> None:
    np.savez(
        Path(path),
        link_idx=table.link_idx, offset_local=table.offset_local,
        link_names=np.array(table.link_names, dtype="<U64"),
        human_idx=table.human_idx,
        tri_idx=table.tri_idx, bary=table.bary, seg=table.seg,
        density=np.float64(table.density), gender=np.str_(table.gender),
        betas=table.betas, g1_rest_cfg=table.g1_rest_cfg,
        version=np.int64(FORMAT_VERSION),
    )


def load_correspondence(path) -> CorrespondenceTable:
    d = np.load(Path(path), allow_pickle=False)
    return CorrespondenceTable(
        link_idx=d["link_idx"], offset_local=d["offset_local"],
        link_names=[str(x) for x in d["link_names"]],
        human_idx=d["human_idx"],
        tri_idx=d["tri_idx"], bary=d["bary"], seg=d["seg"],
        density=float(d["density"]), gender=str(d["gender"]),
        betas=d["betas"], g1_rest_cfg=d["g1_rest_cfg"],
    )


def build_table(model_dir: str, gender: str, betas, urdf_path: str,
                human_density: float, g1_density: float, reg: float) -> CorrespondenceTable:
    """Run the full offline pipeline and return the correspondence table."""
    import yourdfpy

    from .human_body import HumanBody

    from .g1_surface import build_rest_cfg, sample_g1_surface
    from .human_source import build_human_source
    from .ot_couple import couple

    body = HumanBody(model_dir, betas, gender)
    src = build_human_source(body, human_density)

    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    tgt = sample_g1_surface(urdf, g1_density)
    human_idx = couple(src, tgt, reg=reg)

    return CorrespondenceTable(
        link_idx=tgt.link_idx, offset_local=tgt.offset_local, link_names=tgt.link_names,
        human_idx=human_idx, tri_idx=src.tri_idx, bary=src.bary, seg=tgt.seg,
        density=human_density, gender=gender,
        betas=np.asarray(betas, np.float32) if betas is not None else np.zeros(0, np.float32),
        g1_rest_cfg=build_rest_cfg(urdf),
    )


def main() -> None:
    from .constants import G1_29DOF_URDF, HUMAN_GRID_DENSITY

    ap = argparse.ArgumentParser(description="Build fixed human->G1 OT correspondence.")
    ap.add_argument("--model-dir", required=True, help="SMPL-X models directory")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--urdf", default=str(G1_29DOF_URDF))
    ap.add_argument("--human-density", type=float, default=HUMAN_GRID_DENSITY)
    ap.add_argument("--g1-density", type=float, default=3000.0)
    ap.add_argument("--reg", type=float, default=0.005)
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    table = build_table(
        args.model_dir, args.gender, None, args.urdf,
        args.human_density, args.g1_density, args.reg,
    )
    save_correspondence(args.out, table)
    print(f"Saved correspondence: {table.link_idx.shape[0]} G1 points -> {args.out}")


if __name__ == "__main__":
    main()
