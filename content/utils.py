
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Literal
import numpy as np
import pandas as pd

# ------------------ "Advanced function features" demo ------------------
# We use dataclasses for structured options, keyword-only args, and **kwargs pass-through.

@dataclass
class SynthOpts:
    n_units: int = 400
    ori_step: int = 10                  # degrees between orientations
    seed: int = 1
    layers: Tuple[str, ...] = ("SG", "G", "IG")
    effects: Tuple[str, ...] = ("MUL", "MXH")   # multiplicative/divisive vs "Mexican-hat"-like
    amp_range: Tuple[float, float] = (10.0, 50.0)
    base_range: Tuple[float, float] = (0.0, 8.0)
    kappa_range: Tuple[float, float] = (1.0, 6.0)  # von Mises concentration (higher = narrower)


def degwrap(x: float) -> float:
    # Wrap degrees to [0, 360)
    return (x % 360.0 + 360.0) % 360.0


def vm_curve(thetas_deg: np.ndarray, *, amp: float, kappa: float, mu_deg: float, base: float) -> np.ndarray:
    """Von Mises orientation tuning (orientation, not direction). Keyword-only for clarity."""
    th = np.deg2rad(thetas_deg)
    mu = np.deg2rad(mu_deg)
    y = amp * np.exp(kappa * np.cos(2*(th - mu))) / np.exp(kappa) + base
    return y


def mexican_hat_mod(thetas_deg: np.ndarray, *, mu_deg: float, center_supp: float = 0.25, flank_gain: float = 0.20) -> np.ndarray:
    """Return a multiplicative modifier ~1 + flanks - center (very rough)."""
    th = np.deg2rad(thetas_deg)
    mu = np.deg2rad(mu_deg)
    # Narrow bump (center) and broad bump (flanks) using cos(2θ) shapes
    narrow = np.exp(6.0 * np.cos(2*(th - mu))) / np.exp(6.0)
    broad  = np.exp(1.5 * np.cos(2*(th - mu))) / np.exp(1.5)
    mod = 1.0 - center_supp * narrow + flank_gain * (broad - narrow * 0.3)
    return np.clip(mod, 0.1, 2.0)


def compute_osi(rates: np.ndarray, thetas_deg: np.ndarray) -> float:
    """Orientation Selectivity Index via vector sum on doubled angles."""
    th = np.deg2rad(thetas_deg * 2.0)
    vec = np.sum(rates * np.exp(1j * th))
    denom = np.sum(rates) + 1e-12
    return float(abs(vec) / denom)


def _interp_circular(x_deg: np.ndarray, y: np.ndarray, xq_deg: float) -> float:
    # Linear interpolation on circular domain [0,360)
    x = np.radians(x_deg)
    xq = np.radians(xq_deg % 360)
    # unwrap to start at x[0]
    dx = np.unwrap(np.radians(x_deg) - np.radians(x_deg[0]))
    x0 = dx
    y0 = y
    xq0 = (xq - np.radians(x_deg[0]))
    xq0 = (xq0 + 2*np.pi) % (2*np.pi)
    return float(np.interp(xq0, x0.tolist() + [x0[-1] + 2*np.pi], y0.tolist() + [y0[0]]))


def compute_hbw(rates: np.ndarray, thetas_deg: np.ndarray) -> float:
    """Half bandwidth (°) around the peak at half max above baseline (rough)."""
    idx_max = int(np.argmax(rates))
    pref = thetas_deg[idx_max]
    rmax = float(rates[idx_max])
    baseline = float(np.min(rates))
    half = baseline + 0.5 * (rmax - baseline)

    # Search angles going left/right until cross half
    def cross_dir(direction: int) -> float:
        # direction: +1 (increasing angle), -1 (decreasing)
        th = pref
        step = (thetas_deg[1] - thetas_deg[0]) * direction
        for _ in range(3600):
            th_next = degwrap(th + step)
            y_cur = _interp_circular(thetas_deg, rates, th)
            y_next = _interp_circular(thetas_deg, rates, th_next)
            if (direction == 1 and y_next <= half <= y_cur) or (direction == -1 and y_next <= half <= y_cur):
                # linear interpolate angle at half
                if y_cur == y_next:
                    return th_next
                frac = (y_cur - half) / (y_cur - y_next)
                return degwrap(th_next + (1 - frac) * (-step))
            th = th_next
        return degwrap(pref + 90 * direction)

    th_left = cross_dir(-1)
    th_right = cross_dir(+1)
    # angular distance on circle from left to right around pref
    def angdist(a, b):
        d = (b - a + 540) % 360 - 180
        return abs(d)

    return angdist(th_left, th_right)


def simulate_unit(*, layer: str, effect: Literal["MUL", "MXH"], rng: np.random.Generator,
                  amp: float, base: float, kappa: float, mu_deg: float,
                  mul_factor: float = 0.7, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single unit's control and laser curves.
    Uses keyword-only args (MATLAB-like name/value) and **kwargs for future options.
    """
    # Thetas are deferred; provided by outer scope in synth_dataset
    raise RuntimeError("simulate_unit requires synth_dataset to pass thetas via closure.")


def synth_dataset(n_units: int = 400, /, *, seed: int = 1, ori_step: int = 10,
                  layers: Tuple[str, ...] = ("SG","G","IG"), effects: Tuple[str, ...] = ("MUL","MXH"),
                  **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Synthesize a dataset. 'n_units' is positional-only (/) for clarity;
    other options are keyword-only to mimic MATLAB 'arguments' style.
    Returns: units_df, control_df, laser_df, thetas_deg
    """
    opts = SynthOpts(n_units=n_units, seed=seed, ori_step=ori_step, layers=layers, effects=effects)
    rng = np.random.default_rng(opts.seed)
    thetas = np.arange(0, 360, opts.ori_step)

    # Define simulate_unit here to capture 'thetas' via closure safely.
    def _simulate_unit(*, layer: str, effect: str, amp: float, base: float, kappa: float, mu_deg: float,
                       mul_factor: float = 0.7, **kw) -> Tuple[np.ndarray, np.ndarray]:
        y_ctrl = vm_curve(thetas, amp=amp, kappa=kappa, mu_deg=mu_deg, base=base)
        if effect == "MUL":
            y_laser = y_ctrl * mul_factor + base * (1 - mul_factor) * 0.5
        else:
            mod = mexican_hat_mod(thetas, mu_deg=mu_deg)
            y_laser = np.clip(y_ctrl * mod, 0.0, None)
        return y_ctrl, y_laser

    rows_units = []
    rows_ctrl = []
    rows_laser = []

    for uid in range(1, opts.n_units + 1):
        layer = rng.choice(opts.layers, p=[0.35, 0.30, 0.35])
        effect = rng.choice(opts.effects, p=[0.4, 0.6])
        amp = rng.uniform(*opts.amp_range)
        base = rng.uniform(*opts.base_range)
        kappa = rng.uniform(*opts.kappa_range)
        mu_deg = rng.choice(thetas)

        y_ctrl, y_laser = _simulate_unit(layer=layer, effect=effect, amp=amp, base=base, kappa=kappa, mu_deg=mu_deg)

        osi = compute_osi(y_ctrl, thetas)
        hbw = compute_hbw(y_ctrl, thetas)
        fr_mean = float(np.mean(y_ctrl))

        rows_units.append(dict(unit_id=uid, layer=layer, effect=effect, pref_deg=mu_deg,
                               amp=amp, base=base, kappa=kappa, OSI=osi, HBW=hbw, FR_mean=fr_mean))

        for th, yc, yl in zip(thetas, y_ctrl, y_laser):
            rows_ctrl.append(dict(unit_id=uid, ori_deg=float(th), rate=float(yc)))
            rows_laser.append(dict(unit_id=uid, ori_deg=float(th), rate=float(yl)))

    units_df = pd.DataFrame(rows_units)
    ctrl_df = pd.DataFrame(rows_ctrl)
    laser_df = pd.DataFrame(rows_laser)
    return units_df, ctrl_df, laser_df, thetas


def filter_units(units: pd.DataFrame, *, layers: Iterable[str] | None = None,
                 osi_range: Tuple[float, float] = (0.0, 1.0),
                 hbw_range: Tuple[float, float] = (0.0, 180.0),
                 effect: str | None = None) -> pd.DataFrame:
    """Filter helper with keyword-only args (again, MATLAB-like)."""
    df = units.copy()
    if layers:
        df = df[df["layer"].isin(list(layers))]
    df = df[df["OSI"].between(*osi_range)]
    df = df[df["HBW"].between(*hbw_range)]
    if effect and effect != "Any":
        df = df[df["effect"] == effect]
    return df


def overlay_plot(ax, *, ctrl: pd.DataFrame, laser: pd.DataFrame, thetas: np.ndarray,
                 unit_ids: Iterable[int], alpha: float = 0.25, lw: float = 1.0, show_mean: bool = True):
    """Overlay tuning curves; keyword-only; includes simple mean curve option."""
    unit_ids = list(unit_ids)
    if not unit_ids:
        ax.text(0.5, 0.5, "No units match filter.", ha="center", va="center", transform=ax.transAxes)
        return

    # Pre-allocate means
    mat_c = []
    mat_l = []

    for uid in unit_ids:
        yc = ctrl.loc[ctrl.unit_id == uid, "rate"].values
        yl = laser.loc[laser.unit_id == uid, "rate"].values
        mat_c.append(yc)
        mat_l.append(yl)
        ax.plot(thetas, yc, alpha=alpha, lw=lw)
        ax.plot(thetas, yl, alpha=alpha, lw=lw, linestyle="--")

    if show_mean:
        mc = np.mean(np.vstack(mat_c), axis=0)
        ml = np.mean(np.vstack(mat_l), axis=0)
        ax.plot(thetas, mc, lw=2.5)
        ax.plot(thetas, ml, lw=2.5, linestyle="--")

    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Firing rate (a.u.)")
    ax.set_title(f"Overlay tuning curves (N={len(unit_ids)})")
    ax.set_xlim(min(thetas), max(thetas))
    ax.grid(True, alpha=0.3)
