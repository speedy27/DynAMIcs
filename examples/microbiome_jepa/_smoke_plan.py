"""WS3 smoke test — prove the gLV intervention-planning HARNESS runs end to end on CPU.

This builds a TINY, RANDOM-INIT world model (NO checkpoint) + a tiny real gLV simulator and runs the
full MPC loop (encode -> latent MPPI plan -> execute in the env -> re-encode -> replan) for the
``mppi`` and ``random`` methods over 2 episodes, then asserts:
  * the gLV-state -> obs encoding has the right shapes ([1,1,S,F], [1,1,S]);
  * a single latent MPPI plan returns a FINITE action of shape [K] within the action box;
  * the latent rollout produces [N, H, D] finite latents;
  * full episodes run (env steps), each producing a finite success flag + final distance;
  * ``run(...)`` writes the results JSON (and figure) and returns per-method success summaries.

INTEGRITY: the model is RANDOM and UNTRAINED. A random model is NOT expected to plan successfully, so
success rates here will be ~0 and are MEANINGLESS as a planning result — this script ONLY checks that
the machinery executes without error. Real planning quality requires a trained checkpoint via
``plan_glv.run(checkpoint=...)``.

Run:
    /Users/bnz/DynaAMIcs/.venv-cpu/bin/python examples/microbiome_jepa/_smoke_plan.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Make `examples...` importable when run as a script (mirrors _smoke_probe.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.microbiome_jepa.plan_glv import (  # noqa: E402
    MPPIConfig,
    build_glv_and_encoder,
    build_world_model,
    mppi_plan,
    rollout_latent,
    run,
    run_episode,
)
from eb_jepa.training_utils import load_config  # noqa: E402


# Tiny overrides: small encoder, few species/candidates, so everything runs in seconds on CPU.
TINY = {
    "model.d_model": 16,
    "model.n_heads": 2,
    "model.n_layers": 1,
    "model.dim_feedforward": 32,
    "model.dropout": 0.0,
    "data.n_species": 6,
    "data.n_candidate": 3,
    "data.n_traj": 8,
    "data.T": 6,
    "data.num_frames": 4,
    "data.noise_std": 0.0,
}

FNAME = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml"


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cpu")

    # ---- build a tiny RANDOM world model (no checkpoint) + tiny gLV env + encoder ----
    cfg = load_config(FNAME, TINY, quiet=True)
    jepa, cfg, K = build_world_model(FNAME, checkpoint=None, device=dev, overrides=TINY)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    S = sim.n_species
    D = jepa.predictor.rnn.hidden_size
    n_attr = int(sim.attractors.shape[0])
    print(f"[smoke] tiny model: D={D} K={K} | gLV: S={S} n_attr={n_attr} "
          f"stub_glv={state_enc.ds.used_stub}")
    assert K == 3 and S == 6, f"unexpected tiny dims K={K} S={S}"
    assert n_attr >= 2, "need >=2 attractors for a planning task"

    # ---- 1) gLV-state -> obs encoding shapes ----
    x0 = sim.reset(attractor=0)
    obs = state_enc.obs(x0)
    assert obs["otu"].shape == (1, 1, S, int(cfg.model.token_dim)), obs["otu"].shape
    assert obs["mask"].shape == (1, 1, S), obs["mask"].shape
    assert torch.isfinite(obs["otu"]).all(), "non-finite tokens in obs"
    z0 = state_enc.encode(jepa, x0)
    assert z0.shape == (1, D, 1, 1, 1), z0.shape
    assert torch.isfinite(z0).all(), "non-finite start latent"
    print(f"[smoke] OK encoding: obs otu={tuple(obs['otu'].shape)} mask={tuple(obs['mask'].shape)} "
          f"z0={tuple(z0.shape)}")

    # ---- 2) latent rollout shape + finiteness ----
    H, N = 4, 16
    actions = torch.zeros(N, K, H)  # zero-action rollout (relaxation) must be finite
    zs = rollout_latent(jepa.predictor, z0, actions)
    assert zs.shape == (N, H, D), zs.shape
    assert torch.isfinite(zs).all(), "non-finite latent rollout"
    print(f"[smoke] OK rollout_latent: zs={tuple(zs.shape)} finite=True")

    # ---- 3) one MPPI plan returns a finite, in-box action of shape [K] ----
    z_tgt = state_enc.encode(jepa, sim.attractors[1]).flatten(1)  # [1, D]
    mcfg = MPPIConfig(horizon=H, n_samples=32, n_elites=8, n_iters=2, init_std=0.2)
    torch_gen = torch.Generator(device=dev).manual_seed(0)
    a, mean_plan = mppi_plan(jepa.predictor, z0, z_tgt, K, float(sim.config.action_max), mcfg,
                             generator=torch_gen)
    assert a.shape == (K,), a.shape
    assert mean_plan.shape == (H, K), mean_plan.shape
    assert torch.isfinite(a).all(), "MPPI returned non-finite action"
    assert (a.abs() <= sim.config.action_max + 1e-6).all(), "MPPI action outside the action box"
    print(f"[smoke] OK mppi_plan: first_action={a.numpy().round(3).tolist()} "
          f"(|a|<=action_max={sim.config.action_max})")

    # ---- 4) full MPC episodes for mppi + random; success flag computed; env steps happen ----
    attr = sim.attractors
    inter = [float(np.linalg.norm(attr[i] - attr[j]))
             for i in range(n_attr) for j in range(n_attr) if i != j]
    tol = 0.15 * (float(np.mean(inter)) if inter else 1.0)
    rng = np.random.default_rng(0)
    for method in ("mppi", "random"):
        for ep in range(2):
            src, tgt = 0, 1
            res = run_episode(method, sim, jepa, state_enc, src, tgt, tol, mpc_steps=5,
                              mppi_cfg=mcfg, rng=rng, torch_gen=torch_gen)
            assert isinstance(res.success, bool), "success flag not computed"
            assert np.isfinite(res.final_dist), "final_dist not finite"
            assert np.isfinite(res.best_dist), "best_dist not finite"
            print(f"[smoke] OK episode method={method} ep={ep}: success={res.success} "
                  f"start={res.start_dist:.3f} final={res.final_dist:.3f} best={res.best_dist:.3f}")

    # ---- 5) end-to-end run(...) writes JSON (+figure) and returns per-method summaries ----
    with tempfile.TemporaryDirectory() as tmp:
        out = run(
            fname=FNAME, checkpoint=None, seeds="0", n_episodes=2, mpc_steps=5,
            horizon=H, n_samples=32, n_elites=8, n_iters=2, init_std=0.2,
            methods="mppi,random", device="cpu", out=tmp,
            overrides=TINY,  # shrink model + env so run() is a quick CPU pass (same path as build_world_model)
        )
        # We assert the harness produced outputs, NOT any particular success value (random model).
        res_json = Path(tmp) / "planning_results.json"
        assert res_json.exists(), "run() did not write planning_results.json"
        with open(res_json) as f:
            payload = json.load(f)
        assert "summary" in payload and set(payload["summary"]) == {"mppi", "random"}, payload.keys()
        for m in ("mppi", "random"):
            sr = payload["summary"][m]["success_rate_mean"]
            assert np.isfinite(sr), f"{m} success_rate not finite"
        assert isinstance(out["summary"], dict)
        print(f"[smoke] OK run(): wrote {res_json.name}; "
              f"success_rate mppi={payload['summary']['mppi']['success_rate_mean']:.3f} "
              f"random={payload['summary']['random']['success_rate_mean']:.3f} "
              f"(RANDOM model -> ~0 expected)")

    print("\n[smoke] PASS — planning harness runs end to end (random model; not a planning result).")


if __name__ == "__main__":
    main()
