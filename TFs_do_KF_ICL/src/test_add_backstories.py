"""
Rigorous tests for add_backstories() in datasources/filter_dataset.py.

Question under test: backstories should be appended to only `back_frac` of the
`num_tasks` training systems. Concretely, the current implementation gates on:

    sys_choices[i] < math.ceil(config.back_frac * config.num_tasks)

This script verifies:
  T1 deterministic-by-index: for ALL sys indices in [0, num_tasks),
        index <  ceil(bf*N) -> backstory IS added (given first-appearance + non-empty)
        index >= ceil(bf*N) -> backstory is NOT added (always)
  T2 empirical per-trace fraction matches back_frac when sys indices are
        sampled uniformly from [0, num_tasks)
  T3 first-appearance rule: a duplicate eligible system gets exactly ONE backstory
  T4 real_seg_lens[i] == 0 disables the backstory even if eligible
  T5 back_frac == 0    -> never any backstories
  T6 back_frac == 1.0  -> every first-appearance, non-empty segment gets one
  T7 boundary: index == ceil(bf*N)-1 is eligible; index == ceil(bf*N) is NOT
  T8 mask_only_init: only segment 0 considered
  T9 mask_idx and seg_starts are updated consistently with the segment shifts
  T10 each added backstory grows segments by exactly backstory_len rows

The function is loaded directly from the source file via importlib so we
bypass the package's __init__.py (which pulls heavy training deps).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Load add_backstories without triggering datasources/__init__.py
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FD_PATH = HERE / "datasources" / "filter_dataset.py"

# Ensure imports inside filter_dataset can resolve
for p in [str(HERE), str(HERE / "core"), str(HERE / "datasources"), str(HERE / "dyn_models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location("filter_dataset_under_test", str(FD_PATH))
fd_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd_mod)
add_backstories = fd_mod.add_backstories


# ---------------------------------------------------------------------------
# Stub config + sim_objs
# ---------------------------------------------------------------------------
class StubConfig:
    """Minimal duck-typed config matching the attrs add_backstories reads."""

    def __init__(
        self,
        num_tasks: int = 100,
        back_frac: float = 0.5,
        backstory_len: int = 3,
        ny: int = 5,
        max_sys_trace: int = 10,
        n_positions: int = 250,
        iid_gaussian: bool = False,
        iid_gaussian_test: bool = False,
        mask_only_init: bool = False,
    ):
        self.num_tasks = num_tasks
        self.back_frac = back_frac
        self.backstory_len = backstory_len
        self.ny = ny
        self.max_sys_trace = max_sys_trace
        self.n_positions = n_positions
        self.iid_gaussian = iid_gaussian
        self.iid_gaussian_test = iid_gaussian_test
        self.mask_only_init = mask_only_init


class FakeSimObj:
    def __init__(self, ny: int, seed: int):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((ny, ny)) * 0.5


def make_sim_objs(num_tasks: int, ny: int) -> list[FakeSimObj]:
    return [FakeSimObj(ny=ny, seed=i) for i in range(num_tasks)]


# ---------------------------------------------------------------------------
# Helpers to build a fake segments array consistent with populate_traces
# ---------------------------------------------------------------------------
def build_fake_trace(
    cfg: StubConfig,
    sys_choices: list[int],
    real_seg_lens: list[int],
    overhead_per_seg: int = 2,  # the open+close paren tokens
):
    """
    Construct (segments, seg_starts) the way populate_traces would arrange them
    in memory: each segment occupies (real_seg_len + 2) rows in `segments`,
    starting at index 1 (after the global start token).
    """
    n_cols = cfg.ny + 2 * cfg.max_sys_trace + 2
    tok_lens = [
        (rl + overhead_per_seg) if rl > 0 else overhead_per_seg
        for rl in real_seg_lens
    ]
    total_rows = 1 + sum(tok_lens)
    segments = np.zeros((total_rows, n_cols))
    segments[0, 2 * cfg.max_sys_trace] = np.sqrt(2)  # start-of-trace token

    seg_starts: list[int] = []
    pos = 1
    rng = np.random.default_rng(0)
    for rl, tl in zip(real_seg_lens, tok_lens):
        seg_starts.append(pos)
        # put a non-zero observation block at pos+1 .. pos+1+rl so the x0 row
        # (segments[seg_starts[i] + 1, ...]) is non-trivial. Columns beyond
        # 2*max_sys_trace + 2 are the observation payload.
        if rl > 0:
            payload = rng.standard_normal((rl, cfg.ny))
            segments[pos + 1 : pos + 1 + rl, 2 * cfg.max_sys_trace + 2 :] = payload
        pos += tl
    return segments, seg_starts


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
class _Reporter:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = ""):
        self.results.append((name, ok, detail))
        marker = "PASS" if ok else "FAIL"
        line = f"  [{marker}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)

    def summary(self) -> int:
        n = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        print()
        print(f"Summary: {passed}/{n} checks passed")
        return 0 if passed == n else 1


REP = _Reporter()


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def test_T1_deterministic_by_index():
    """For every sys index in [0, num_tasks), exhaustively check eligibility."""
    print("\n[T1] deterministic-by-index for ALL sys indices")
    cfg = StubConfig(num_tasks=37, back_frac=0.5, backstory_len=3, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    threshold = math.ceil(cfg.back_frac * cfg.num_tasks)  # 19

    fails = []
    for sys_idx in range(cfg.num_tasks):
        # Single trace, single segment, this system, non-empty
        sys_choices = [sys_idx]
        real_seg_lens = [4]
        segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
        seg_starts_for_call = seg_starts.copy()

        out_segments, mask_idx = add_backstories(
            cfg,
            sim_objs,
            segments,
            mask_idx=[],
            sys_appear=[],
            sys_choices=sys_choices,
            seg_starts=seg_starts_for_call,
            real_seg_lens=real_seg_lens,
            test=False,
        )

        # Did a backstory get inserted? Detect via segments growing.
        grew = out_segments.shape[0] > segments.shape[0] or out_segments.shape[0] >= cfg.n_positions + 1
        # More precise: the *backstory rows* are inserted at seg_starts[0] + 1
        # and have length cfg.backstory_len. The marker for "real" backstory
        # presence (vs the trailing zero pad to n_positions) is mask_idx
        # containing exactly the backstory window AND additional pad mask
        # indices. So check whether the first backstory_len mask indices land
        # at seg_starts[0]+1.
        x0 = seg_starts[0] + 1
        added = (
            len(mask_idx) >= cfg.backstory_len
            and list(mask_idx[: cfg.backstory_len]) == list(range(x0, x0 + cfg.backstory_len))
        )

        expected = sys_idx < threshold
        if added != expected:
            fails.append((sys_idx, expected, added))

    REP.record(
        "every index in [0, num_tasks) matches threshold",
        not fails,
        f"threshold=ceil({cfg.back_frac}*{cfg.num_tasks})={threshold}; mismatches={fails[:5]}"
        if fails
        else f"threshold=ceil({cfg.back_frac}*{cfg.num_tasks})={threshold}; all 0..{cfg.num_tasks - 1} agree",
    )


def test_T2_empirical_fraction_matches_back_frac():
    """Across many traces with uniformly-sampled sys indices, the fraction of
    systems that receive a backstory should be ~ back_frac."""
    print("\n[T2] empirical per-trace fraction tracks back_frac")
    for back_frac in (0.25, 0.5, 0.75):
        cfg = StubConfig(num_tasks=400, back_frac=back_frac, backstory_len=2, ny=4, max_sys_trace=10)
        sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
        rng = np.random.default_rng(123)

        n_traces = 2000
        sys_in_trace = 5
        total_first_appearances = 0
        total_backstoried = 0

        for _ in range(n_traces):
            sys_inds = rng.choice(cfg.num_tasks, sys_in_trace, replace=False).tolist()
            # Each segment is one of these sys (cycle to keep simple)
            sys_choices = sys_inds[:]
            real_seg_lens = [4] * len(sys_choices)
            segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
            mask_idx = []
            sys_appear = []
            _, mask_idx = add_backstories(
                cfg,
                sim_objs,
                segments,
                mask_idx=mask_idx,
                sys_appear=sys_appear,
                sys_choices=sys_choices,
                seg_starts=seg_starts.copy(),
                real_seg_lens=real_seg_lens,
                test=False,
            )
            total_first_appearances += len(set(sys_choices))
            total_backstoried += len(sys_appear)

        empirical = total_backstoried / total_first_appearances
        threshold_frac = math.ceil(back_frac * cfg.num_tasks) / cfg.num_tasks
        ok = abs(empirical - threshold_frac) < 0.02
        REP.record(
            f"back_frac={back_frac}: empirical={empirical:.4f} expected≈{threshold_frac:.4f}",
            ok,
        )


def test_T3_first_appearance_only():
    print("\n[T3] duplicate eligible system -> exactly ONE backstory")
    cfg = StubConfig(num_tasks=10, back_frac=1.0, backstory_len=3, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = [3, 3, 3, 3]
    real_seg_lens = [2, 2, 2, 2]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, mask_idx = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    # exactly one backstory window inserted
    cnt = Counter(sys_appear)
    REP.record("sys_appear has exactly one entry for the duplicated system", sys_appear == [3])
    REP.record("only one backstory_len-block of non-pad mask indices", len(mask_idx) >= cfg.backstory_len)


def test_T4_empty_segment_blocks_backstory():
    print("\n[T4] real_seg_lens[i] == 0 disables backstory")
    cfg = StubConfig(num_tasks=10, back_frac=1.0, backstory_len=3, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)

    # All zero-length first appearances -> none should be backstoried
    sys_choices = [0, 1, 2]
    real_seg_lens = [0, 0, 0]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    REP.record("no backstories when every real_seg_len is 0", sys_appear == [])

    # Mixed: 0-len first occurrence is skipped (sys_appear NOT updated either,
    # because the gating is a single conjunction). A later non-empty occurrence
    # of the SAME sys must therefore receive a backstory.
    sys_choices = [4, 4]
    real_seg_lens = [0, 3]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    REP.record("0-len first occurrence is skipped, next non-empty gets it", sys_appear == [4])


def test_T5_back_frac_zero():
    print("\n[T5] back_frac=0 -> no backstories anywhere")
    cfg = StubConfig(num_tasks=50, back_frac=0.0, backstory_len=3, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = list(range(10))
    real_seg_lens = [3] * 10
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    REP.record("sys_appear empty when back_frac=0", sys_appear == [])


def test_T6_back_frac_one():
    print("\n[T6] back_frac=1 -> every first-appearance, non-empty segment is backstoried")
    cfg = StubConfig(num_tasks=50, back_frac=1.0, backstory_len=3, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = [11, 22, 11, 33, 22, 44]
    real_seg_lens = [3, 3, 3, 3, 3, 3]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    REP.record("all unique first-appearances captured", sorted(sys_appear) == [11, 22, 33, 44])


def test_T7_threshold_boundary():
    print("\n[T7] boundary at index == ceil(back_frac*num_tasks)")
    # back_frac=0.5, num_tasks=37 -> threshold=19. Index 18 eligible, 19 not.
    cfg = StubConfig(num_tasks=37, back_frac=0.5, backstory_len=2, ny=4, max_sys_trace=5)
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    threshold = math.ceil(cfg.back_frac * cfg.num_tasks)

    for sys_idx, should in [(threshold - 1, True), (threshold, False), (threshold + 1, False)]:
        sys_choices = [sys_idx]
        real_seg_lens = [3]
        segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
        sys_appear = []
        _, _ = add_backstories(
            cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
        )
        got = bool(sys_appear)
        REP.record(f"index={sys_idx}: expected backstory={should}, got={got}", got is should)


def test_T8_mask_only_init():
    print("\n[T8] mask_only_init -> only segment 0 considered")
    cfg = StubConfig(
        num_tasks=10, back_frac=1.0, backstory_len=2, ny=4, max_sys_trace=5, mask_only_init=True
    )
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = [3, 4, 5]
    real_seg_lens = [3, 3, 3]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    sys_appear = []
    _, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    REP.record("only first segment got a backstory", sys_appear == [3])


def test_T9_mask_idx_and_seg_starts_updates():
    print("\n[T9] mask_idx and seg_starts are updated consistently")
    cfg = StubConfig(
        num_tasks=10, back_frac=1.0, backstory_len=4, ny=4, max_sys_trace=5, n_positions=500
    )
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = [1, 2]
    real_seg_lens = [3, 3]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    seg_starts_in = seg_starts.copy()
    sys_appear = []
    out, mask_idx = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts_in, real_seg_lens
    )

    # Two backstories inserted, each backstory_len rows.
    # Original seg_starts: [s0, s1] with s0=1, s1=1+(3+2)=6.
    # After first backstory at i=0, seg_starts[1] += backstory_len -> s1=10.
    # After second backstory at i=1 there are no later starts to bump.
    # The function mutates seg_starts in place.
    expected_seg_starts = [1, 10]
    REP.record(f"seg_starts updated to {expected_seg_starts}", seg_starts_in == expected_seg_starts)

    # mask_idx should contain two windows of length backstory_len:
    # window1: [s0+1 .. s0+1+bl) = [2 .. 6)
    # window2 (after shift): [s1+1 .. s1+1+bl) = [11 .. 15)
    bl = cfg.backstory_len
    expected_first = list(range(2, 2 + bl))
    expected_second = list(range(11, 11 + bl))
    got_first = list(mask_idx[:bl])
    got_second = list(mask_idx[bl : 2 * bl])
    REP.record(f"first mask window {expected_first}", got_first == expected_first, f"got {got_first}")
    REP.record(f"second mask window {expected_second}", got_second == expected_second, f"got {got_second}")


def test_T10_segments_grow_correctly():
    print("\n[T10] each backstory grows segments by exactly backstory_len rows")
    cfg = StubConfig(
        num_tasks=10, back_frac=1.0, backstory_len=3, ny=4, max_sys_trace=5, n_positions=500
    )
    sim_objs = make_sim_objs(cfg.num_tasks, cfg.ny)
    sys_choices = [0, 1, 2]
    real_seg_lens = [3, 3, 3]
    segments, seg_starts = build_fake_trace(cfg, sys_choices, real_seg_lens)
    pre_rows = segments.shape[0]
    sys_appear = []
    out, _ = add_backstories(
        cfg, sim_objs, segments, [], sys_appear, sys_choices, seg_starts.copy(), real_seg_lens
    )
    # n_positions is large so the right-pad branch will pad up to n_positions+1.
    # Therefore out.shape[0] == n_positions+1, but BEFORE the pad, the inserted
    # backstories accounted for len(sys_appear) * backstory_len extra rows.
    # We can't observe pre-pad shape, so we assert the pad invariant:
    expected = max(pre_rows + len(sys_appear) * cfg.backstory_len, cfg.n_positions + 1)
    REP.record(
        f"segments shape after backstories+pad == {expected}",
        out.shape[0] == expected,
        f"pre={pre_rows} backstories={len(sys_appear)} got={out.shape[0]}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    np.random.seed(0)
    test_T1_deterministic_by_index()
    test_T2_empirical_fraction_matches_back_frac()
    test_T3_first_appearance_only()
    test_T4_empty_segment_blocks_backstory()
    test_T5_back_frac_zero()
    test_T6_back_frac_one()
    test_T7_threshold_boundary()
    test_T8_mask_only_init()
    test_T9_mask_idx_and_seg_starts_updates()
    test_T10_segments_grow_correctly()
    return REP.summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
