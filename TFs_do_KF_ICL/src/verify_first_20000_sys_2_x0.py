"""Thorough structural check for the first_20000_sys_2_x0 training dataset.

Run from any cwd:
    python src/verify_first_20000_sys_2_x0.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/media/volume/ICL_Kalman_Experiments/train_and_test_data/ortho_haar")
NEW_TRAIN = ROOT / "train_ortho_haar_ident_C_state_dim_5_first_20000_sys_2_x0.pkl"
NEW_SIMOBJS = ROOT / "train_ortho_haar_ident_C_state_dim_5_first_20000_sys_2_x0_sim_objs.pkl"
VANILLA_SIMOBJS = ROOT / "train_ortho_haar_ident_C_state_dim_5_sim_objs.pkl"
VANILLA_TRAIN = ROOT / "train_ortho_haar_ident_C_state_dim_5.pkl"

N_SYS_EXPECTED, K_EXPECTED = 20000, 2
N_ENTRIES_EXPECTED = N_SYS_EXPECTED * K_EXPECTED


failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS " if cond else "FAIL "
    msg = tag + name + (f"  --  {detail}" if detail else "")
    print(msg)
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- load
print(f"Loading {NEW_TRAIN.name} ...")
with open(NEW_TRAIN, "rb") as f:
    new_samples = pickle.load(f)
print(f"Loading {NEW_SIMOBJS.name} ...")
with open(NEW_SIMOBJS, "rb") as f:
    new_simobjs = pickle.load(f)
print(f"Loading vanilla sim_objs ...")
with open(VANILLA_SIMOBJS, "rb") as f:
    vanilla_simobjs = pickle.load(f)
print(f"Loading vanilla train (for sample-format reference) ...")
with open(VANILLA_TRAIN, "rb") as f:
    vanilla_samples = pickle.load(f)
print()


# ---------------------------------------------------------------- 1. lengths
check("len(samples) == 40000", len(new_samples) == N_ENTRIES_EXPECTED,
      f"got {len(new_samples)}")
check("len(sim_objs) == 40000", len(new_simobjs) == N_ENTRIES_EXPECTED,
      f"got {len(new_simobjs)}")


# ---------------------------------------------------------------- 2. is-aliasing
bad_pairs = [k for k in range(N_SYS_EXPECTED) if new_simobjs[2 * k] is not new_simobjs[2 * k + 1]]
check("sim_objs[2k] is sim_objs[2k+1] for all k",
      not bad_pairs, f"violations at k={bad_pairs[:5]}" if bad_pairs else "")
unique_ids = {id(o) for o in new_simobjs}
check("exactly 20000 unique FilterSim objects in sim_objs",
      len(unique_ids) == N_SYS_EXPECTED, f"got {len(unique_ids)} unique")


# ---------------------------------------------------------------- 3. attr match vs vanilla[:20000]
attrs = ("A", "C", "sigma_w", "sigma_v", "S_state_inf", "S_observation_inf", "n_noise")
mismatched: list[tuple[int, str]] = []
for k in range(N_SYS_EXPECTED):
    new_obj = new_simobjs[2 * k]
    van_obj = vanilla_simobjs[k]
    for attr in attrs:
        a = getattr(new_obj, attr)
        b = getattr(van_obj, attr)
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                mismatched.append((k, attr))
                break
        else:
            if a != b:
                mismatched.append((k, attr))
                break
check("all 7 attrs of new_simobjs[2k] match vanilla[k] for k in 0..19999",
      not mismatched,
      f"first mismatches: {mismatched[:5]}" if mismatched else "")


# ---------------------------------------------------------------- 4. sample dict shape
v_keys = set(vanilla_samples[0].keys())
n_keys = set(new_samples[0].keys())
check("sample keys match vanilla", v_keys == n_keys,
      f"new={sorted(n_keys)} vanilla={sorted(v_keys)}")
for key in v_keys & n_keys:
    va, na = vanilla_samples[0][key], new_samples[0][key]
    if isinstance(va, np.ndarray):
        check(f"  '{key}': ndim/dtype/last-axis match",
              va.ndim == na.ndim and va.dtype == na.dtype
              and (va.shape[-1] == na.shape[-1] if va.ndim >= 1 else True),
              f"vanilla {va.shape}/{va.dtype} vs new {na.shape}/{na.dtype}")


# ---------------------------------------------------------------- 5. independence + dynamics
obs_key = "obs" if "obs" in new_samples[0] else next(iter(new_samples[0]))
rng = np.random.default_rng(0)
sample_ks = rng.choice(N_SYS_EXPECTED, size=8, replace=False)
for k in sample_ks:
    t0, t1 = new_samples[2 * k], new_samples[2 * k + 1]
    y0, y1 = np.asarray(t0[obs_key]), np.asarray(t1[obs_key])
    check(f"  k={k}: trace0 != trace1",
          not np.allclose(y0, y1),
          f"max|diff|={float(np.max(np.abs(y0 - y1))):.4f}")
    check(f"  k={k}: x0 differs",
          not np.allclose(y0[0], y1[0]),
          f"||y0[0]-y1[0]||={float(np.linalg.norm(y0[0] - y1[0])):.4f}")
    A = new_simobjs[2 * k].A
    C = new_simobjs[2 * k].C
    sigma_w = new_simobjs[2 * k].sigma_w
    if np.allclose(C, np.eye(C.shape[0])):
        resid = np.linalg.norm(y0[1:] - y0[:-1] @ A.T, axis=1)
        # Per-step residual is x noise (state) + 2 readout noises -> std ~ sqrt(sigma_w^2 + 2 sigma_v^2).
        # Allow generous 6x margin.
        threshold = 6 * sigma_w
        check(f"  k={k}: trace0 follows A within ~6*sigma_w",
              float(resid.mean()) < threshold,
              f"mean residual {float(resid.mean()):.4f}, sigma_w {sigma_w}, thr {threshold}")


# ---------------------------------------------------------------- statistical x0 ~ steady-state
all_y0 = np.stack([np.asarray(s[obs_key])[0] for s in new_samples[:2000]])
mean_y0 = all_y0.mean(axis=0)
cov_y0 = np.cov(all_y0.T)
expected_cov = np.mean([o.S_observation_inf for o in new_simobjs[:2000]], axis=0)
check("mean obs[0] ~ 0 over 2000 samples",
      float(np.linalg.norm(mean_y0)) < 0.2,
      f"||mean||={float(np.linalg.norm(mean_y0)):.4f}")
check("cov(obs[0]) ~ mean(S_observation_inf) over 2000 samples",
      float(np.linalg.norm(cov_y0 - expected_cov, ord="fro")) < 0.5,
      f"||delta||_F={float(np.linalg.norm(cov_y0 - expected_cov, ord='fro')):.4f}")


# ---------------------------------------------------------------- summary
print()
if failures:
    print(f"FAILURES ({len(failures)}):")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("ALL PASS")
    sys.exit(0)
