# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH vs MPM cylindrical column-collapse benchmark.

Runs ``newton.examples.sph_granular`` and ``newton.examples.mpm_granular2``
as subprocesses on harmonized physical parameters, parses each frame's
runout / max_h / KE diagnostics, and writes:

- ``analysis/results/runs.csv``     (one row per (solver, config, frame))
- ``analysis/results/summary.csv``  (one row per run)
- ``analysis/results/runout_vs_time.png``
- ``analysis/results/final_runout_bar.png``
- ``analysis/results/kinetic_energy.png``

Usage::

    uv run python analysis/sph_vs_mpm_column_collapse.py
"""

from __future__ import annotations

import csv
import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# matplotlib is intentionally imported lazily inside _setup_matplotlib() and
# the plot functions: TID253 disallows module-level matplotlib imports in
# this repo. The plotting helpers below pull `plt` from sys.modules after
# `_setup_matplotlib()` has been called.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository root inferred from this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Harmonized scenario parameters (passed as CLI args to both examples).
SCENARIO = {
    "cylinder_radius": 0.1,  # [m] L_0
    "cylinder_height": 0.2,  # [m] H
    "particle_spacing": 0.005,  # [m] dx; gives ~50k particles
    "duration": 0.5,  # [s]
    "fps": 60.0,
    "density": 2500.0,  # [kg/m^3]
    "young_modulus": 1.0e6,  # [Pa]
    "poisson_ratio": 0.3,
    "friction": 0.5,  # [-] tan(phi); phi ≈ 26.6 deg
    "viscosity": 0.0,
    "gravity": (0.0, 0.0, -9.81),
}


# Lube et al. 2004 J. Fluid Mech. column-collapse scaling.
# For aspect ratio a = H / L_0:
#   a < ~1.7   :  (L_f - L_0) / L_0 ≈ 1.2 a
#   a > ~3     :  (L_f - L_0) / L_0 ≈ 1.6 a^(2/3)
# We provide both estimates and report whichever bracket dominates the user's setup.
def lube_runout(aspect: float) -> tuple[float, float]:
    """Return (lower-bracket prediction, upper-bracket prediction) for L_f / L_0.

    Args:
        aspect: H / L_0.

    Returns:
        Pair (low, high) of L_f / L_0 from the two Lube regimes.
    """
    short_r = 1.0 + 1.2 * aspect
    tall_r = 1.0 + 1.6 * aspect ** (2.0 / 3.0)
    return short_r, tall_r


# ---------------------------------------------------------------------------
# Subprocess invocation + parsing
# ---------------------------------------------------------------------------

# Regex to parse frame diagnostics emitted by both examples after this commit.
# Matches lines like:
#   "sim step:    428, time: 0.016667s, runout: 0.100312m, max_h: 0.196293m, KE: 0.213014J"
FRAME_RE = re.compile(
    r"sim step:\s*(?P<step>\d+),\s*time:\s*(?P<t>[\d.eE+-]+)s,\s*"
    r"runout:\s*(?P<runout>[\d.eE+-]+)m,\s*"
    r"max_h:\s*(?P<max_h>[\d.eE+-]+)m,\s*"
    r"KE:\s*(?P<ke>[\d.eE+-]+)J"
)
WALL_RE = re.compile(r"Total wall time \(compile \+ run\):\s*(?P<sec>[\d.eE+-]+)s")


@dataclass
class RunResult:
    solver: str
    config: str
    cmd: list[str]
    frames: list[dict[str, float]] = field(default_factory=list)
    wall_time_s: float = float("nan")
    particle_count: int = -1
    crashed: bool = False
    error_tail: str = ""

    @property
    def final_runout(self) -> float:
        return self.frames[-1]["runout"] if self.frames else float("nan")

    @property
    def final_ke(self) -> float:
        return self.frames[-1]["ke"] if self.frames else float("nan")


def _scenario_args() -> list[str]:
    """Build CLI args common to both examples from SCENARIO."""
    g = SCENARIO["gravity"]
    return [
        "--cylinder-radius",
        str(SCENARIO["cylinder_radius"]),
        "--cylinder-height",
        str(SCENARIO["cylinder_height"]),
        "--particle-spacing",
        str(SCENARIO["particle_spacing"]),
        "--duration",
        str(SCENARIO["duration"]),
        "--fps",
        str(SCENARIO["fps"]),
        "--density",
        str(SCENARIO["density"]),
        "--young-modulus",
        str(SCENARIO["young_modulus"]),
        "--poisson-ratio",
        str(SCENARIO["poisson_ratio"]),
        "--friction",
        str(SCENARIO["friction"]),
        "--viscosity",
        str(SCENARIO["viscosity"]),
        "--gravity",
        str(g[0]),
        str(g[1]),
        str(g[2]),
    ]


def run_subprocess(label: str, example: str, extra_args: list[str], device: str, timeout_s: float) -> RunResult:
    """Run a single Newton example as a subprocess and parse diagnostics.

    Args:
        label: Human-readable config label, e.g. ``"sph"`` or ``"mpm-Q1-pic-apic"``.
        example: Newton example module name, e.g. ``"sph_granular"``.
        extra_args: Solver-specific CLI flags appended to scenario args.
        device: ``"cuda:0"`` or ``"cpu"``.
        timeout_s: Per-run wall timeout in seconds.

    Returns:
        RunResult with parsed frames, wall time, and crash diagnostics.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "newton.examples",
        example,
        "--device",
        device,
        "--viewer",
        "null",
        "--quiet",
        # Cap num-frames generously above the duration*fps target so the runner
        # doesn't terminate before --duration triggers SystemExit.
        "--num-frames",
        str(int(SCENARIO["duration"] * SCENARIO["fps"]) + 10),
    ]
    cmd.extend(_scenario_args())
    cmd.extend(extra_args)

    solver = "sph" if example.startswith("sph") else "mpm"
    result = RunResult(solver=solver, config=label, cmd=cmd)

    print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result.crashed = True
        result.error_tail = f"TimeoutExpired after {timeout_s:.0f}s\n{(exc.stderr or '')[-1000:]}"
        result.wall_time_s = time.perf_counter() - t0
        print(f"[{label}] TIMEOUT after {result.wall_time_s:.1f}s", flush=True)
        return result

    elapsed = time.perf_counter() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Examples raise SystemExit("Reached ...") on completion -> non-zero rc.
    # Treat that as success if we parsed at least one frame.
    for line in stdout.splitlines():
        m = FRAME_RE.search(line)
        if m:
            result.frames.append(
                {
                    "step": int(m.group("step")),
                    "t": float(m.group("t")),
                    "runout": float(m.group("runout")),
                    "max_h": float(m.group("max_h")),
                    "ke": float(m.group("ke")),
                }
            )

    wm = WALL_RE.search(stdout) or WALL_RE.search(stderr)
    result.wall_time_s = float(wm.group("sec")) if wm else elapsed

    # Crude particle count: count "add_particles" / log lines if present;
    # otherwise leave at -1 and infer separately.
    pc_match = re.search(r"particle[_ ]?count[:\s]+(\d+)", stdout, flags=re.IGNORECASE)
    if pc_match:
        result.particle_count = int(pc_match.group(1))

    # Decide pass/fail.
    if not result.frames:
        result.crashed = True
        # Capture last 1000 chars of stderr/stdout for the report.
        tail = (stderr or stdout).strip().splitlines()[-30:]
        result.error_tail = "\n".join(tail)
        print(f"[{label}] CRASHED (rc={proc.returncode}). tail:\n{result.error_tail}", flush=True)
    else:
        print(
            f"[{label}] done: {len(result.frames)} frames, "
            f"final runout={result.final_runout:.4f}m, wall={result.wall_time_s:.1f}s",
            flush=True,
        )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_matplotlib() -> None:
    """Apply publication-quality rcParams. Call once before plotting."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9,
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _color_for(label: str, idx: int, total: int) -> tuple[float, float, float]:
    import matplotlib.pyplot as plt

    if label == "sph":
        return (0.0, 0.0, 0.0)
    cmap = plt.cm.viridis
    return cmap(idx / max(1, total - 1))


def plot_runout_vs_time(results: list[RunResult], l0: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mpm_results = [r for r in results if r.solver == "mpm" and r.frames]
    n_mpm = len(mpm_results)

    for r in results:
        if not r.frames:
            continue
        t = np.array([f["t"] for f in r.frames])
        runout = np.array([f["runout"] for f in r.frames])
        if r.solver == "sph":
            ax.plot(t, runout / l0, color="black", linewidth=2.0, label="SPH", zorder=10)
        else:
            idx = mpm_results.index(r)
            ax.plot(t, runout / l0, color=_color_for("mpm", idx, n_mpm), linewidth=1.2, label=r.config)

    ax.set_xlabel("time t [s]")
    ax.set_ylabel(r"runout $L(t) / L_0$ [-]")
    ax.set_title("Cylindrical column collapse: SPH vs MPM configurations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "runout_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_runout_bar(results: list[RunResult], l0: float, aspect: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    vals = []
    colors = []
    mpm_idx = 0
    n_mpm = sum(1 for r in results if r.solver == "mpm" and r.frames)
    for r in results:
        if not r.frames:
            continue
        labels.append(r.config)
        vals.append(r.final_runout / l0)
        if r.solver == "sph":
            colors.append("black")
        else:
            colors.append(_color_for("mpm", mpm_idx, n_mpm))
            mpm_idx += 1

    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"final runout $L_f / L_0$ [-]")
    ax.set_title(f"Final runout per solver/config (aspect a = H / L_0 = {aspect:.2f})")
    ax.grid(True, axis="y", alpha=0.3)

    # Lube reference lines.
    short_r, tall_r = lube_runout(aspect)
    if aspect < 1.7:
        ax.axhline(short_r, color="red", linestyle="--", linewidth=1.2, label=f"Lube short: {short_r:.2f}")
    elif aspect > 3.0:
        ax.axhline(tall_r, color="red", linestyle="--", linewidth=1.2, label=f"Lube tall: {tall_r:.2f}")
    else:
        ax.axhline(short_r, color="red", linestyle="--", linewidth=1.2, label=f"Lube short ext.: {short_r:.2f}")
        ax.axhline(tall_r, color="red", linestyle=":", linewidth=1.2, label=f"Lube tall ext.: {tall_r:.2f}")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_runout_bar.png", bbox_inches="tight")
    plt.close(fig)


def plot_kinetic_energy(results: list[RunResult]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    sph = next((r for r in results if r.solver == "sph" and r.frames), None)
    mpms = [r for r in results if r.solver == "mpm" and r.frames]
    if not mpms:
        plt.close(fig)
        return

    # Pick representative MPM configs by final runout (max and min).
    mpms_sorted = sorted(mpms, key=lambda r: r.final_runout)
    rep_min = mpms_sorted[0]
    rep_max = mpms_sorted[-1]

    plotted = []
    if sph:
        plotted.append(("SPH", sph, "black", 2.0))
    plotted.append((f"MPM (max runout): {rep_max.config}", rep_max, "tab:red", 1.5))
    if rep_min is not rep_max:
        plotted.append((f"MPM (min runout): {rep_min.config}", rep_min, "tab:blue", 1.5))

    for label, r, color, lw in plotted:
        t = np.array([f["t"] for f in r.frames])
        ke = np.array([f["ke"] for f in r.frames])
        ax.plot(t, ke, color=color, linewidth=lw, label=label)

    ax.set_xlabel("time t [s]")
    ax.set_ylabel(r"total kinetic energy KE [J]")
    ax.set_title("KE(t): dissipation comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kinetic_energy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_runs_csv(results: list[RunResult]) -> None:
    path = RESULTS_DIR / "runs.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "config", "frame", "t", "runout", "max_h", "ke"])
        for r in results:
            for i, f in enumerate(r.frames):
                w.writerow(
                    [
                        r.solver,
                        r.config,
                        i,
                        f"{f['t']:.6f}",
                        f"{f['runout']:.6f}",
                        f"{f['max_h']:.6f}",
                        f"{f['ke']:.6f}",
                    ]
                )


def write_summary_csv(results: list[RunResult]) -> None:
    path = RESULTS_DIR / "summary.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "solver",
                "config",
                "frames_emitted",
                "final_t_s",
                "final_runout_m",
                "final_max_h_m",
                "final_ke_J",
                "wall_time_s",
                "particle_count",
                "crashed",
                "error_tail",
            ]
        )
        for r in results:
            last = (
                r.frames[-1]
                if r.frames
                else {"t": float("nan"), "runout": float("nan"), "max_h": float("nan"), "ke": float("nan")}
            )
            w.writerow(
                [
                    r.solver,
                    r.config,
                    len(r.frames),
                    f"{last['t']:.6f}",
                    f"{last['runout']:.6f}",
                    f"{last['max_h']:.6f}",
                    f"{last['ke']:.6f}",
                    f"{r.wall_time_s:.2f}",
                    r.particle_count,
                    int(r.crashed),
                    r.error_tail.replace("\n", " | ")[:500],
                ]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """Return ``"cuda:0"`` if CUDA is available, else ``"cpu"``."""
    try:
        import warp as wp  # noqa: PLC0415

        wp.init()
        d = wp.get_device()
        if d.is_cuda:
            return "cuda:0"
    except Exception as exc:
        print(f"Warp device probe failed: {exc}", flush=True)
    return "cpu"


def main() -> int:
    device = detect_device()
    # Generous per-run timeout: each run is expected to finish in < 120s on CUDA.
    # If on CPU the harness still tries but with a much longer budget.
    timeout_s = 360.0 if device.startswith("cuda") else 1800.0
    print(f"== device = {device}, per-run timeout = {timeout_s}s ==", flush=True)

    results: list[RunResult] = []

    # ---- SPH baseline ---------------------------------------------------
    sph_run = run_subprocess(
        label="sph",
        example="sph_granular",
        extra_args=[],  # uses example defaults (penalty boundary, kh=1.3, c_s=50)
        device=device,
        timeout_s=timeout_s,
    )
    results.append(sph_run)

    # ---- MPM 8-config sweep --------------------------------------------
    velocity_bases = ["Q1", "B2"]
    integration_schemes = ["pic", "gimp"]
    transfer_schemes = ["pic", "apic"]
    for vb, isch, tsch in itertools.product(velocity_bases, integration_schemes, transfer_schemes):
        label = f"mpm-{vb}-{isch}-{tsch}"
        extra = [
            "--velocity-basis",
            vb,
            "--integration-scheme",
            isch,
            "--transfer-scheme",
            tsch,
        ]
        results.append(
            run_subprocess(
                label=label,
                example="mpm_granular2",
                extra_args=extra,
                device=device,
                timeout_s=timeout_s,
            )
        )

    # ---- Persist + plot -------------------------------------------------
    write_runs_csv(results)
    write_summary_csv(results)

    l0 = SCENARIO["cylinder_radius"]
    aspect = SCENARIO["cylinder_height"] / l0

    _setup_matplotlib()
    plot_runout_vs_time(results, l0)
    plot_final_runout_bar(results, l0, aspect)
    plot_kinetic_energy(results)

    # ---- Console summary ------------------------------------------------
    print("\n=== Final L_f / L_0 ===")
    short_r, tall_r = lube_runout(aspect)
    for r in results:
        if r.frames:
            print(f"  {r.config:<24s} {r.final_runout / l0:8.3f}  (wall={r.wall_time_s:6.1f}s)")
        else:
            print(f"  {r.config:<24s} CRASHED ({r.error_tail.splitlines()[-1] if r.error_tail else 'no output'})")
    print(f"  {'Lube short ref':<24s} {short_r:8.3f}")
    print(f"  {'Lube tall ref':<24s} {tall_r:8.3f}")

    print(f"\nWrote {RESULTS_DIR}/runs.csv, summary.csv, *.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
