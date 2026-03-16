"""Nyx-style tabulated cooling/heating support.

This module precomputes and interpolates a table for net volumetric source terms
from Nyx atomic + UV background rates as a function of:
    - redshift z
    - baryon overdensity delta_b = rho_b / <rho_b>(z)
    - temperature T [K]

The underlying rates are evaluated through the Fortran-backed ``eos_t`` module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import jax.numpy as jnp
import numpy as np


# Match Nyx constants used in f_rhs_struct.H and constants_cosmo.H.
_COMPTON_C = 1.01765467e-37
_T_CMB = 2.725
_M_PROTON_CGS = 1.6726231e-24
_G_CGS = 6.6743e-8
_MPC_CM = 3.085677581e24


@dataclass(frozen=True)
class NyxCoolingTableData:
    """Container for Nyx cooling/heating lookup tables."""

    z_nodes: np.ndarray
    logdelta_nodes: np.ndarray
    logT_nodes: np.ndarray
    net_cgs: np.ndarray
    heat_cgs: np.ndarray
    cool_cgs: np.ndarray
    h: float
    omega_b: float
    h_species: float
    uvb_density_a: float
    uvb_density_b: float
    jh: int
    jhe: int

    @classmethod
    def load(cls, path: Path) -> "NyxCoolingTableData":
        with np.load(path) as d:
            return cls(
                z_nodes=np.asarray(d["z_nodes"], dtype=np.float64),
                logdelta_nodes=np.asarray(d["logdelta_nodes"], dtype=np.float64),
                logT_nodes=np.asarray(d["logT_nodes"], dtype=np.float64),
                net_cgs=np.asarray(d["net_cgs"], dtype=np.float32),
                heat_cgs=np.asarray(d["heat_cgs"], dtype=np.float32),
                cool_cgs=np.asarray(d["cool_cgs"], dtype=np.float32),
                h=float(np.asarray(d["h"])),
                omega_b=float(np.asarray(d["omega_b"])),
                h_species=float(np.asarray(d["h_species"])),
                uvb_density_a=float(np.asarray(d["uvb_density_a"])),
                uvb_density_b=float(np.asarray(d["uvb_density_b"])),
                jh=int(np.asarray(d["jh"])),
                jhe=int(np.asarray(d["jhe"])),
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_nodes=np.asarray(self.z_nodes, dtype=np.float64),
            logdelta_nodes=np.asarray(self.logdelta_nodes, dtype=np.float64),
            logT_nodes=np.asarray(self.logT_nodes, dtype=np.float64),
            net_cgs=np.asarray(self.net_cgs, dtype=np.float32),
            heat_cgs=np.asarray(self.heat_cgs, dtype=np.float32),
            cool_cgs=np.asarray(self.cool_cgs, dtype=np.float32),
            h=np.asarray(self.h, dtype=np.float64),
            omega_b=np.asarray(self.omega_b, dtype=np.float64),
            h_species=np.asarray(self.h_species, dtype=np.float64),
            uvb_density_a=np.asarray(self.uvb_density_a, dtype=np.float64),
            uvb_density_b=np.asarray(self.uvb_density_b, dtype=np.float64),
            jh=np.asarray(self.jh, dtype=np.int32),
            jhe=np.asarray(self.jhe, dtype=np.int32),
        )


def rho_crit0_cgs(h: float) -> float:
    h0_s = (100.0 * float(h)) * 1.0e5 / _MPC_CM
    return 3.0 * h0_s * h0_s / (8.0 * np.pi * _G_CGS)


def rho_b_mean_cgs(z: float, *, h: float, omega_b: float) -> float:
    return rho_crit0_cgs(h) * float(omega_b) * (1.0 + float(z)) ** 3


def rho_b_mean_cgs_jax(z: jnp.ndarray, *, h: float, omega_b: float) -> jnp.ndarray:
    h0_s = (100.0 * float(h)) * 1.0e5 / _MPC_CM
    rho_c0 = 3.0 * h0_s * h0_s / (8.0 * np.pi * _G_CGS)
    return jnp.asarray(rho_c0 * float(omega_b), dtype=jnp.float32) * (1.0 + z) ** 3


def parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError(f"No float values parsed from '{s}'")
    return out


@contextmanager
def _pushd(path: Path):
    import os

    prev = Path.cwd()
    try:
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def _import_eos_t(extra_paths: Iterable[Path] | None = None):
    try:
        import eos_t  # type: ignore

        return eos_t
    except Exception:
        pass

    candidates: list[Path] = []
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "nyx_eos")
    if extra_paths is not None:
        for p in extra_paths:
            candidates.append(Path(p))

    for c in candidates:
        c = c.resolve()
        if c.is_file():
            c = c.parent
        if c.is_dir():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)

    import eos_t  # type: ignore

    return eos_t


def _initialize_atomic_rates(eos_t_mod, treecool_file: Path) -> None:
    treecool_file = treecool_file.resolve()
    if not treecool_file.exists():
        raise FileNotFoundError(f"TREECOOL file not found: {treecool_file}")

    link_name = "TREECOOL_middle"
    with _pushd(treecool_file.parent):
        tmp_created = False
        tmp_path = Path(link_name)
        if treecool_file.name != link_name:
            if not tmp_path.exists():
                try:
                    tmp_path.symlink_to(treecool_file)
                except Exception:
                    import shutil

                    shutil.copy2(treecool_file, tmp_path)
                tmp_created = True
        eos_t_mod.atomic_rates.tabulate_rates()
        if tmp_created:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _interp_rate_from_table(logt: float, table: np.ndarray, tcoolmin: float, tcoolmax: float, ncooltab: int) -> float:
    dlogt = (tcoolmax - tcoolmin) / float(ncooltab)
    if logt <= tcoolmin:
        logt_eff = tcoolmin + 0.5 * dlogt
    else:
        logt_eff = min(logt, tcoolmax)
    tmp = (logt_eff - tcoolmin) / dlogt
    j = int(np.floor(tmp))
    fhi = tmp - float(j)
    flo = 1.0 - fhi
    j = max(0, min(j, table.shape[0] - 2))
    return flo * float(table[j]) + fhi * float(table[j + 1])


def _ion_fractions_fixed_t(
    ne: float,
    nh: float,
    logt: float,
    *,
    rates: dict[str, np.ndarray],
    jh: int,
    jhe: int,
    h_species: float,
) -> tuple[float, float, float]:
    ahp = _interp_rate_from_table(logt, rates["alphahp"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    ahep = _interp_rate_from_table(logt, rates["alphahep"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    ahepp = _interp_rate_from_table(logt, rates["alphahepp"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    ad = _interp_rate_from_table(logt, rates["alphad"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    geh0 = _interp_rate_from_table(logt, rates["gammaeh0"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    gehe0 = _interp_rate_from_table(logt, rates["gammaehe0"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    gehep = _interp_rate_from_table(logt, rates["gammaehep"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])

    if ne > 0.0 and nh > 0.0:
        ggh0ne = float(jh) * float(rates["ggh0"]) / (ne * nh)
        gghe0ne = float(jh) * float(rates["gghe0"]) / (ne * nh)
        gghepne = float(jhe) * float(rates["gghep"]) / (ne * nh)
    else:
        ggh0ne = 0.0
        gghe0ne = 0.0
        gghepne = 0.0

    yhelium = (1.0 - float(h_species)) / (4.0 * max(float(h_species), 1.0e-30))
    nhp = 1.0 - ahp / max(ahp + geh0 + ggh0ne, 1.0e-60)
    nhp = min(max(nhp, 0.0), 1.0)

    denom_1 = gehe0 + gghe0ne
    if denom_1 > 1.0e-60 and ahepp > 1.0e-60:
        nhep = yhelium / (1.0 + (ahep + ad) / denom_1 + (gehep + gghepne) / ahepp)
        nhep = max(nhep, 0.0)
    else:
        nhep = 0.0

    if nhep > 0.0 and ahepp > 1.0e-60:
        nhepp = nhep * (gehep + gghepne) / ahepp
        nhepp = max(nhepp, 0.0)
    else:
        nhepp = 0.0
    return nhp, nhep, nhepp


def _solve_species_fixed_t(
    nh: float,
    logt: float,
    *,
    rates: dict[str, np.ndarray],
    jh: int,
    jhe: int,
    h_species: float,
) -> tuple[float, float, float, float, float]:
    """Solve (nh0, nhp, nhe0, nhep, nhepp) with fixed T, robustly."""
    yhelium = (1.0 - float(h_species)) / (4.0 * max(float(h_species), 1.0e-30))
    ne = 1.0
    xacc = 1.0e-8

    converged = False
    nhp = 0.0
    nhep = 0.0
    nhepp = 0.0
    for _ in range(40):
        nhp, nhep, nhepp = _ion_fractions_fixed_t(
            ne, nh, logt, rates=rates, jh=jh, jhe=jhe, h_species=h_species
        )
        eps = max(xacc * max(ne, 1.0), 1.0e-24)
        nhp2, nhep2, nhepp2 = _ion_fractions_fixed_t(
            ne + eps, nh, logt, rates=rates, jh=jh, jhe=jhe, h_species=h_species
        )
        dnhp = (nhp2 - nhp) / eps
        dnhep = (nhep2 - nhep) / eps
        dnhepp = (nhepp2 - nhepp) / eps
        f = ne - nhp - nhep - 2.0 * nhepp
        df = 1.0 - dnhp - dnhep - 2.0 * dnhepp
        if not np.isfinite(df) or abs(df) < 1.0e-16:
            break
        dne = f / df
        ne_new = max(ne - dne, 0.0)
        if not np.isfinite(ne_new):
            break
        ne = min(ne_new, 4.0)
        if abs(dne) < xacc:
            converged = True
            break

    if (not converged) and (not np.isfinite(ne)):
        ne = 1.0
        nhp, nhep, nhepp = _ion_fractions_fixed_t(
            ne, nh, logt, rates=rates, jh=jh, jhe=jhe, h_species=h_species
        )

    nh0 = max(1.0 - nhp, 0.0)
    nhe0 = max(yhelium - (nhep + nhepp), 0.0)
    return nh0, nhp, nhe0, nhep, nhepp


def _nyx_heat_cool_scalar(
    *,
    z: float,
    rho_cgs: float,
    temp_k: float,
    h_species: float,
    uvb_density_a: float,
    uvb_density_b: float,
    jh: int,
    jhe: int,
    delta_b: float,
    rates: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    nh = max(float(rho_cgs), 1.0e-300) * float(h_species) / _M_PROTON_CGS
    # Very low-T points can trigger non-convergence in the Fortran Newton solver.
    # Nyx resets pathological cells to finite temperatures during the source update.
    # For robust table generation we clamp solver temperature to >=10 K.
    t = max(float(temp_k), 10.0)
    logt = float(np.log10(t))

    nh0_f, nhp_f, nhe0_f, nhep_f, nhepp_f = _solve_species_fixed_t(
        nh,
        logt,
        rates=rates,
        jh=jh,
        jhe=jhe,
        h_species=h_species,
    )
    ne_f = nhp_f + nhep_f + 2.0 * nhepp_f
    ne = nh * float(ne_f)
    nh0 = nh * float(nh0_f)
    nhp = nh * float(nhp_f)
    nhe0 = nh * float(nhe0_f)
    nhep = nh * float(nhep_f)
    nhepp = nh * float(nhepp_f)

    z1 = 1.0 + float(z)
    lambda_c = _COMPTON_C * (_T_CMB**4) * ne * (t - _T_CMB * z1) * (z1**4)

    if logt >= float(rates["tcoolmax"]):
        lambda_ff = (
            1.42e-27
            * np.sqrt(t)
            * (1.1 + 0.34 * np.exp(-((5.5 - logt) * (5.5 - logt)) / 3.0))
            * (nhp + 4.0 * nhepp)
            * ne
        )
        cool = lambda_ff + lambda_c
        return 0.0, float(cool), float(-cool)

    bh0 = _interp_rate_from_table(logt, rates["betah0"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    bhe0 = _interp_rate_from_table(logt, rates["betahe0"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    bhep = _interp_rate_from_table(logt, rates["betahep"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    bff1 = _interp_rate_from_table(logt, rates["betaff1"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    bff4 = _interp_rate_from_table(logt, rates["betaff4"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    rhp = _interp_rate_from_table(logt, rates["rechp"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    rhep = _interp_rate_from_table(logt, rates["rechep"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])
    rhepp = _interp_rate_from_table(logt, rates["rechepp"], rates["tcoolmin"], rates["tcoolmax"], rates["ncooltab"])

    lambda_atomic = (
        bh0 * nh0
        + bhe0 * nhe0
        + bhep * nhep
        + rhp * nhp
        + rhep * nhep
        + rhepp * nhepp
        + bff1 * (nhp + nhep)
        + bff4 * nhepp
    ) * ne

    cool = float(lambda_atomic + lambda_c)
    heat = float(jh) * nh0 * float(rates["eh0"]) + float(jh) * nhe0 * float(rates["ehe0"]) + float(jhe) * nhep * float(
        rates["ehep"]
    )
    rho_heat = float(uvb_density_a) * (max(float(delta_b), 1.0e-30) ** float(uvb_density_b))
    heat *= rho_heat
    return float(heat), float(cool), float(heat - cool)


def build_nyx_cooling_table(
    out_npz: Path,
    *,
    treecool_file: Path,
    z_nodes: Iterable[float],
    logdelta_nodes: Iterable[float],
    logT_nodes: Iterable[float],
    h: float,
    omega_b: float,
    h_species: float = 0.76,
    uvb_density_a: float = 1.0,
    uvb_density_b: float = 0.0,
    jh: int = 1,
    jhe: int = 1,
    eos_search_paths: Iterable[Path] | None = None,
) -> NyxCoolingTableData:
    """Precompute Nyx-like net heating/cooling table and save as NPZ."""
    eos_t_mod = _import_eos_t(eos_search_paths)
    _initialize_atomic_rates(eos_t_mod, treecool_file)

    z_arr = np.asarray(sorted(float(v) for v in z_nodes), dtype=np.float64)
    ld_arr = np.asarray(list(logdelta_nodes), dtype=np.float64)
    lt_arr = np.asarray(list(logT_nodes), dtype=np.float64)
    if z_arr.size < 2:
        raise ValueError("Need at least two z nodes for interpolation.")
    if ld_arr.size < 2 or lt_arr.size < 2:
        raise ValueError("Need at least two logdelta/logT nodes.")

    rates = {
        "ncooltab": int(eos_t_mod.atomic_rates.ncooltab),
        "tcoolmin": float(eos_t_mod.atomic_rates.tcoolmin),
        "tcoolmax": float(eos_t_mod.atomic_rates.tcoolmax),
        "alphahp": np.asarray(eos_t_mod.atomic_rates.alphahp, dtype=np.float64),
        "alphahep": np.asarray(eos_t_mod.atomic_rates.alphahep, dtype=np.float64),
        "alphahepp": np.asarray(eos_t_mod.atomic_rates.alphahepp, dtype=np.float64),
        "alphad": np.asarray(eos_t_mod.atomic_rates.alphad, dtype=np.float64),
        "gammaeh0": np.asarray(eos_t_mod.atomic_rates.gammaeh0, dtype=np.float64),
        "gammaehe0": np.asarray(eos_t_mod.atomic_rates.gammaehe0, dtype=np.float64),
        "gammaehep": np.asarray(eos_t_mod.atomic_rates.gammaehep, dtype=np.float64),
        "betah0": np.asarray(eos_t_mod.atomic_rates.betah0, dtype=np.float64),
        "betahe0": np.asarray(eos_t_mod.atomic_rates.betahe0, dtype=np.float64),
        "betahep": np.asarray(eos_t_mod.atomic_rates.betahep, dtype=np.float64),
        "betaff1": np.asarray(eos_t_mod.atomic_rates.betaff1, dtype=np.float64),
        "betaff4": np.asarray(eos_t_mod.atomic_rates.betaff4, dtype=np.float64),
        "rechp": np.asarray(eos_t_mod.atomic_rates.rechp, dtype=np.float64),
        "rechep": np.asarray(eos_t_mod.atomic_rates.rechep, dtype=np.float64),
        "rechepp": np.asarray(eos_t_mod.atomic_rates.rechepp, dtype=np.float64),
        "ggh0": 0.0,
        "gghe0": 0.0,
        "gghep": 0.0,
        "eh0": 0.0,
        "ehe0": 0.0,
        "ehep": 0.0,
    }

    heat = np.zeros((z_arr.size, ld_arr.size, lt_arr.size), dtype=np.float32)
    cool = np.zeros_like(heat)
    net = np.zeros_like(heat)

    for iz, z in enumerate(z_arr):
        eos_t_mod.atomic_rates.interp_to_this_z(float(z))
        rates["ggh0"] = float(eos_t_mod.atomic_rates.ggh0)
        rates["gghe0"] = float(eos_t_mod.atomic_rates.gghe0)
        rates["gghep"] = float(eos_t_mod.atomic_rates.gghep)
        rates["eh0"] = float(eos_t_mod.atomic_rates.eh0)
        rates["ehe0"] = float(eos_t_mod.atomic_rates.ehe0)
        rates["ehep"] = float(eos_t_mod.atomic_rates.ehep)

        rho_mean = rho_b_mean_cgs(float(z), h=h, omega_b=omega_b)
        for idl, ld in enumerate(ld_arr):
            delta_b = 10.0 ** float(ld)
            rho_cgs = rho_mean * delta_b
            for it, lt in enumerate(lt_arr):
                t_k = 10.0 ** float(lt)
                hh, cc, nn = _nyx_heat_cool_scalar(
                    z=float(z),
                    rho_cgs=float(rho_cgs),
                    temp_k=float(t_k),
                    h_species=float(h_species),
                    uvb_density_a=float(uvb_density_a),
                    uvb_density_b=float(uvb_density_b),
                    jh=int(jh),
                    jhe=int(jhe),
                    delta_b=float(delta_b),
                    rates=rates,
                )
                heat[iz, idl, it] = np.float32(hh)
                cool[iz, idl, it] = np.float32(cc)
                net[iz, idl, it] = np.float32(nn)

    table = NyxCoolingTableData(
        z_nodes=z_arr,
        logdelta_nodes=ld_arr,
        logT_nodes=lt_arr,
        net_cgs=net,
        heat_cgs=heat,
        cool_cgs=cool,
        h=float(h),
        omega_b=float(omega_b),
        h_species=float(h_species),
        uvb_density_a=float(uvb_density_a),
        uvb_density_b=float(uvb_density_b),
        jh=int(jh),
        jhe=int(jhe),
    )
    table.save(out_npz)
    return table


class NyxCoolingRateInterpolator:
    """JAX runtime interpolator for Nyx cooling/heating tables."""

    def __init__(self, table: NyxCoolingTableData):
        self.h = float(table.h)
        self.omega_b = float(table.omega_b)
        self.z_nodes = jnp.asarray(table.z_nodes, dtype=jnp.float32)
        self.logdelta_nodes = jnp.asarray(table.logdelta_nodes, dtype=jnp.float32)
        self.logT_nodes = jnp.asarray(table.logT_nodes, dtype=jnp.float32)
        self.net = jnp.asarray(table.net_cgs, dtype=jnp.float32)
        self.heat = jnp.asarray(table.heat_cgs, dtype=jnp.float32)
        self.cool = jnp.asarray(table.cool_cgs, dtype=jnp.float32)

        if self.logdelta_nodes.size < 2 or self.logT_nodes.size < 2:
            raise ValueError("NyxCoolingRateInterpolator needs >=2 nodes along both logdelta/logT axes.")

        self.ld_min = self.logdelta_nodes[0]
        self.ld_max = self.logdelta_nodes[-1]
        self.lt_min = self.logT_nodes[0]
        self.lt_max = self.logT_nodes[-1]
        self.n_ld = int(self.logdelta_nodes.size)
        self.n_lt = int(self.logT_nodes.size)
        self._ld_scale = (self.n_ld - 1) / jnp.maximum(self.ld_max - self.ld_min, 1.0e-12)
        self._lt_scale = (self.n_lt - 1) / jnp.maximum(self.lt_max - self.lt_min, 1.0e-12)

    def _bilinear(self, grid2d: jnp.ndarray, logdelta: jnp.ndarray, logt: jnp.ndarray) -> jnp.ndarray:
        x = (jnp.clip(logdelta, self.ld_min, self.ld_max) - self.ld_min) * self._ld_scale
        y = (jnp.clip(logt, self.lt_min, self.lt_max) - self.lt_min) * self._lt_scale
        i0 = jnp.floor(x).astype(jnp.int32)
        j0 = jnp.floor(y).astype(jnp.int32)
        i0 = jnp.clip(i0, 0, self.n_ld - 2)
        j0 = jnp.clip(j0, 0, self.n_lt - 2)
        fx = x - i0.astype(jnp.float32)
        fy = y - j0.astype(jnp.float32)

        v00 = grid2d[i0, j0]
        v10 = grid2d[i0 + 1, j0]
        v01 = grid2d[i0, j0 + 1]
        v11 = grid2d[i0 + 1, j0 + 1]
        return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

    def evaluate(self, rho_cgs: jnp.ndarray, temp_k: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_scalar = jnp.asarray(z, dtype=jnp.float32)
        rho_mean = rho_b_mean_cgs_jax(z_scalar, h=self.h, omega_b=self.omega_b)
        delta = jnp.maximum(jnp.asarray(rho_cgs, dtype=jnp.float32) / jnp.maximum(rho_mean, 1.0e-30), 1.0e-30)
        logdelta = jnp.log10(delta)
        logt = jnp.log10(jnp.maximum(jnp.asarray(temp_k, dtype=jnp.float32), 1.0e-30))

        zc = jnp.clip(z_scalar, self.z_nodes[0], self.z_nodes[-1])
        hi = jnp.searchsorted(self.z_nodes, zc, side="right")
        hi = jnp.clip(hi, 1, self.z_nodes.shape[0] - 1)
        lo = hi - 1
        z0 = self.z_nodes[lo]
        z1 = self.z_nodes[hi]
        wz = (zc - z0) / jnp.maximum(z1 - z0, 1.0e-30)

        heat0 = self._bilinear(self.heat[lo], logdelta, logt)
        heat1 = self._bilinear(self.heat[hi], logdelta, logt)
        cool0 = self._bilinear(self.cool[lo], logdelta, logt)
        cool1 = self._bilinear(self.cool[hi], logdelta, logt)
        net0 = self._bilinear(self.net[lo], logdelta, logt)
        net1 = self._bilinear(self.net[hi], logdelta, logt)

        heat = (1.0 - wz) * heat0 + wz * heat1
        cool = (1.0 - wz) * cool0 + wz * cool1
        net = (1.0 - wz) * net0 + wz * net1
        return heat, cool, net
