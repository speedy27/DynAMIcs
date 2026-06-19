"""
Build the M3 planning figure(s) from MEASURED JSONs (no recomputation; reads results/*.json).

Panel A — CONTROLLABILITY: oracle (perfect-model) success + final distance vs action-panel size K
          (results/oracle_K_sweep.json). The task is uncontrollable until ~all species are dose-able.
Panel B — READOUT FIDELITY -> PLANNING: for each trained K=24 encoder regime, decoded-state MPPI final
          distance vs the linear/MLP state-readout R^2 (results/planning_decoded_*.json). Higher
          readout fidelity => better planning (the isolated bottleneck), with baselines + tol marked.

INTEGRITY: plots only numbers present in the committed result JSONs. Missing inputs are skipped with a
note, never fabricated.

Run: .venv-cpu/bin/python -m examples.microbiome_jepa.make_planning_figure
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("examples/microbiome_jepa/results")
# (json tag, label) for each K=24 encoder regime, in increasing readout fidelity
REGIMES = [("default", "default reg\n(cov=25)"), ("lowreg", "weak reg\n(cov=1)"), ("big", "weak reg, big\n(d256/512traj)")]


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- Panel A: controllability curve ----
    osw = RES / "oracle_K_sweep.json"
    if osw.exists():
        d = json.loads(osw.read_text())
        rows = d["rows"]
        Ks = [r["K"] for r in rows]
        sr = [r["success_rate_mean"] for r in rows]
        fd = [r["mean_final_dist_mean"] for r in rows]
        tol = rows[0]["tol"]
        start = rows[0]["mean_start_dist"]
        c1, c2 = "#2a7", "#c84"
        axA.plot(Ks, sr, "o-", color=c1, lw=2, label="oracle success")
        axA.set_xlabel("action panel size K (dose-able species; S=24)")
        axA.set_ylabel("oracle success rate", color=c1)
        axA.set_ylim(-0.03, 1.05)
        axA.tick_params(axis="y", labelcolor=c1)
        axA2 = axA.twinx()
        axA2.plot(Ks, fd, "s--", color=c2, lw=2, label="oracle final dist")
        axA2.axhline(tol, ls=":", color=c2, lw=1, label=f"tol={tol:.2f}")
        axA2.set_ylabel(f"mean final dist (start {start:.1f})", color=c2)
        axA2.tick_params(axis="y", labelcolor=c2)
        axA.set_xticks(Ks)
        axA.set_title("A. Controllability (perfect model): task solvable only at K=24")
    else:
        axA.text(0.5, 0.5, "oracle_K_sweep.json missing", ha="center")

    # ---- Panel B: readout fidelity -> decoded planning ----
    xs, ys, ss, labs = [], [], [], []
    base = None
    for tag, lab in REGIMES:
        p = RES / f"planning_decoded_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        r2 = d["decoder_r2"]["mlp"]
        s = d["summary"]["mppi_decoded_mlp"]
        xs.append(r2)
        ys.append(s["mean_final_dist"])
        ss.append(s["success_rate_mean"])
        labs.append(lab)
        base = d  # keep last for baselines/tol
    if xs:
        sc = axB.scatter(xs, ys, c=ss, s=140, cmap="viridis", vmin=0, vmax=max(0.1, max(ss)),
                         edgecolor="k", zorder=3)
        for x, y, lab, succ in zip(xs, ys, labs, ss):
            axB.annotate(f"{lab}\n{succ*100:.0f}% succ", (x, y), textcoords="offset points",
                         xytext=(8, 6), fontsize=8)
        axB.plot(xs, ys, "-", color="#888", lw=1, zorder=2)
        tol = base["tol"]
        rnd = base["summary"]["random"]["mean_final_dist"]
        grd = base["summary"]["greedy"]["mean_final_dist"]
        lat = base["summary"]["mppi_latent"]["mean_final_dist"]
        axB.axhline(tol, ls=":", color="green", label=f"reach tol={tol:.2f}")
        axB.axhline(rnd, ls="--", color="#999", lw=1, label=f"random ({rnd:.1f})")
        axB.axhline(grd, ls="--", color="#c84", lw=1, label=f"greedy ({grd:.1f})")
        axB.axhline(lat, ls="--", color="#a44", lw=1, label=f"latent-MPPI ({lat:.1f})")
        axB.set_xlabel("state-readout fidelity  R²(z → x)  [MLP probe]")
        axB.set_ylabel("decoded-state MPPI: mean final dist (lower = better)")
        axB.set_title("B. Better readout → better planning (the isolated bottleneck)")
        axB.legend(fontsize=7, loc="upper right")
        cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
        cb.set_label("decoded-MPPI success rate", fontsize=8)
    else:
        axB.text(0.5, 0.5, "planning_decoded_*.json missing", ha="center")

    fig.suptitle("gLV intervention planning: controllability (A) and the representation bottleneck (B)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RES / "planning_diagnosis.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved -> {out}  (regimes plotted: {labs})")


if __name__ == "__main__":
    main()
