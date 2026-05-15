"""
Unit checks for the eval-time `eval_sys_subset` override.

Two formulas are under test:
 1) In `populate_traces` (filter_dataset.py:187 branch), needle-in-haystack mode
    selects system indices as `np.arange(base + example, base + example + sys_in_trace)`
    with `base = 0` for masked/None and `base = ceil(back_frac * num_tasks)` for unmasked.
 2) In `data_train.py` (--multi_haystack branch), `num_haystack_examples` is capped to
    keep every emitted trace strictly inside the chosen subset:
        masked   -> max(0, threshold - num_sys_haystack)
        unmasked -> max(0, (num_tasks - threshold) - num_sys_haystack)

Both pieces live in long, deeply-nested call paths; this test exercises the formulas in
isolation so a regression in the index math is caught without spinning up the full
training/eval harness.
"""
from __future__ import annotations

import math

import numpy as np


def enumerate_sys_inds(eval_sys_subset, example, sys_in_trace, back_frac, num_tasks):
    """Mirror of the populate_traces needle-in-haystack enumeration (filter_dataset.py)."""
    if eval_sys_subset == "unmasked":
        base = math.ceil(back_frac * num_tasks)
    else:
        base = 0
    return np.arange(base + example, base + example + sys_in_trace)


def cap_num_haystack_examples(eval_sys_subset, back_frac, num_tasks, num_sys_haystack, max_sys_trace, new_hay_insert=False):
    """Mirror of the data_train.py multi_haystack cap (datasource in {train, backstory_train}).

    `new_hay_insert=True` accounts for the populate_traces branch that accesses
    `entries[sys_inds[-1] + 1]` on the final segment, which needs one extra
    in-bounds slot.
    """
    new_hay_extra = 1 if new_hay_insert else 0
    num_haystack_examples = num_tasks - max_sys_trace
    if eval_sys_subset in ("masked", "unmasked"):
        threshold = math.ceil(back_frac * num_tasks)
        if eval_sys_subset == "masked":
            num_haystack_examples = max(0, threshold - num_sys_haystack + 1 - new_hay_extra)
        else:
            num_haystack_examples = max(0, (num_tasks - threshold) - num_sys_haystack + 1 - new_hay_extra)
        if num_haystack_examples == 0:
            raise ValueError("zero valid examples")
    return num_haystack_examples


def should_insert_backstory(sys_choice, sys_appear, real_seg_len, test, eval_sys_subset, back_frac, num_tasks):
    """Mirror of the add_backstories per-system gate (filter_dataset.py).

    Training-time calls (test=False) keep the back_frac filter: only systems
    in [0, ceil(back_frac*num_tasks)) get backstories. Eval-time calls with
    `eval_sys_subset` set bypass that filter, because populate_traces has
    already restricted sys_inds to the chosen subset.
    """
    subset_active_at_test = test and eval_sys_subset in ("masked", "unmasked")
    threshold = math.ceil(back_frac * num_tasks)
    return (
        sys_choice not in sys_appear
        and real_seg_len > 0
        and (subset_active_at_test or sys_choice < threshold)
    )


def assert_subset(actual_range, lo, hi, label):
    arr = np.asarray(actual_range)
    assert arr.min() >= lo, f"{label}: min={arr.min()} below {lo}"
    assert arr.max() < hi, f"{label}: max={arr.max()} not below {hi}"


def test_masked_enumeration_stays_below_threshold():
    back_frac, num_tasks, num_sys_haystack = 0.5, 100, 5
    threshold = math.ceil(back_frac * num_tasks)  # 50
    cap = cap_num_haystack_examples("masked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    assert cap == threshold - num_sys_haystack + 1 == 46

    for ex in range(cap):
        sys_inds = enumerate_sys_inds("masked", ex, num_sys_haystack, back_frac, num_tasks)
        assert_subset(sys_inds, 0, threshold, f"masked ex={ex}")
    # last valid example's last system is exactly threshold-1
    last = enumerate_sys_inds("masked", cap - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last[-1]) == threshold - 1
    # one beyond the cap would cross the boundary
    over = enumerate_sys_inds("masked", cap, num_sys_haystack, back_frac, num_tasks)
    assert int(over[-1]) >= threshold
    print("PASS masked stays in [0, threshold)")


def test_unmasked_enumeration_stays_above_threshold_and_below_num_tasks():
    back_frac, num_tasks, num_sys_haystack = 0.5, 100, 5
    threshold = math.ceil(back_frac * num_tasks)
    cap = cap_num_haystack_examples("unmasked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    assert cap == (num_tasks - threshold) - num_sys_haystack + 1 == 46

    for ex in range(cap):
        sys_inds = enumerate_sys_inds("unmasked", ex, num_sys_haystack, back_frac, num_tasks)
        assert_subset(sys_inds, threshold, num_tasks, f"unmasked ex={ex}")
    last = enumerate_sys_inds("unmasked", cap - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last[-1]) == num_tasks - 1
    print("PASS unmasked stays in [threshold, num_tasks)")


def test_none_preserves_existing_behavior():
    """eval_sys_subset=None must produce the same enumeration and cap as before the change."""
    back_frac, num_tasks, num_sys_haystack, max_sys_trace = 0.5, 100, 5, 25
    cap = cap_num_haystack_examples(None, back_frac, num_tasks, num_sys_haystack, max_sys_trace)
    assert cap == num_tasks - max_sys_trace == 75, f"cap regressed: {cap}"
    for ex in (0, 1, 42, cap - 1):
        sys_inds = enumerate_sys_inds(None, ex, num_sys_haystack, back_frac, num_tasks)
        assert list(sys_inds) == list(range(ex, ex + num_sys_haystack))
    print("PASS None preserves prior behavior")


def test_zero_valid_examples_raises():
    """back_frac=0.01, num_sys_haystack=5 -> masked threshold=1, cap=max(0, 1-5)=0 -> raise."""
    try:
        cap_num_haystack_examples("masked", 0.01, 100, num_sys_haystack=5, max_sys_trace=25)
    except ValueError:
        print("PASS masked with too-small back_frac raises ValueError")
        return
    raise AssertionError("expected ValueError when subset has zero valid examples")


def test_back_frac_075_asymmetric():
    """back_frac=0.75 with num_sys_haystack=5: masked cap=71, unmasked cap=21."""
    back_frac, num_tasks, num_sys_haystack = 0.75, 100, 5
    cap_m = cap_num_haystack_examples("masked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    cap_u = cap_num_haystack_examples("unmasked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    assert cap_m == 71 and cap_u == 21, f"caps wrong: masked={cap_m} unmasked={cap_u}"
    # last masked trace at ex=70 -> [70, 71, 72, 73, 74] (all < 75 = threshold)
    last_m = enumerate_sys_inds("masked", cap_m - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last_m[-1]) == 74
    # last unmasked trace at ex=20 -> [95, 96, 97, 98, 99] (top is 99 < 100)
    last_u = enumerate_sys_inds("unmasked", cap_u - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last_u[-1]) == 99
    print("PASS asymmetric back_frac=0.75 caps")


def test_new_hay_insert_reduces_cap_by_one_and_stays_in_bounds():
    """new_hay_insert accesses entries[sys_inds[-1] + 1]; cap must shrink by 1."""
    back_frac, num_tasks, num_sys_haystack = 0.5, 100, 5
    threshold = math.ceil(back_frac * num_tasks)

    cap_m_off = cap_num_haystack_examples("masked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    cap_m_on = cap_num_haystack_examples("masked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25, new_hay_insert=True)
    assert cap_m_on == cap_m_off - 1, f"masked: off={cap_m_off} on={cap_m_on}"
    last_m = enumerate_sys_inds("masked", cap_m_on - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last_m[-1]) + 1 < threshold, f"masked+new_hay_insert crosses threshold: {last_m}"

    cap_u_off = cap_num_haystack_examples("unmasked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25)
    cap_u_on = cap_num_haystack_examples("unmasked", back_frac, num_tasks, num_sys_haystack, max_sys_trace=25, new_hay_insert=True)
    assert cap_u_on == cap_u_off - 1, f"unmasked: off={cap_u_off} on={cap_u_on}"
    last_u = enumerate_sys_inds("unmasked", cap_u_on - 1, num_sys_haystack, back_frac, num_tasks)
    assert int(last_u[-1]) + 1 < num_tasks, f"unmasked+new_hay_insert crosses num_tasks: {last_u}"
    print("PASS new_hay_insert shrinks cap by 1 and keeps sys_inds[-1]+1 in bounds")


def test_add_backstories_gate_training_keeps_back_frac_filter():
    """Training path (test=False) must only backstory systems below the threshold."""
    back_frac, num_tasks = 0.25, 40000
    threshold = math.ceil(back_frac * num_tasks)  # 10000

    # below threshold -> insert
    assert should_insert_backstory(
        sys_choice=42, sys_appear=[], real_seg_len=5,
        test=False, eval_sys_subset=None,
        back_frac=back_frac, num_tasks=num_tasks,
    ) is True
    # at/above threshold -> skip (training policy preserved)
    assert should_insert_backstory(
        sys_choice=threshold, sys_appear=[], real_seg_len=5,
        test=False, eval_sys_subset=None,
        back_frac=back_frac, num_tasks=num_tasks,
    ) is False
    assert should_insert_backstory(
        sys_choice=threshold + 100, sys_appear=[], real_seg_len=5,
        test=False, eval_sys_subset="unmasked",  # eval_sys_subset must be ignored when test=False
        back_frac=back_frac, num_tasks=num_tasks,
    ) is False
    print("PASS training path preserves back_frac filter")


def test_add_backstories_gate_eval_unmasked_inserts_above_threshold():
    """Eval with eval_sys_subset=unmasked must backstory systems >= threshold (the bug fix)."""
    back_frac, num_tasks = 0.25, 40000
    threshold = math.ceil(back_frac * num_tasks)  # 10000

    for sys_choice in (threshold, threshold + 1, threshold + 5000, num_tasks - 1):
        assert should_insert_backstory(
            sys_choice=sys_choice, sys_appear=[], real_seg_len=10,
            test=True, eval_sys_subset="unmasked",
            back_frac=back_frac, num_tasks=num_tasks,
        ) is True, f"unmasked eval should insert backstory at sys_choice={sys_choice}"
    print("PASS eval+unmasked inserts backstories above threshold (bug fixed)")


def test_add_backstories_gate_eval_masked_unchanged():
    """Eval with eval_sys_subset=masked is unchanged: every selected (in-range) sys gets a backstory."""
    back_frac, num_tasks = 0.25, 40000
    threshold = math.ceil(back_frac * num_tasks)
    for sys_choice in (0, 1, threshold // 2, threshold - 1):
        assert should_insert_backstory(
            sys_choice=sys_choice, sys_appear=[], real_seg_len=10,
            test=True, eval_sys_subset="masked",
            back_frac=back_frac, num_tasks=num_tasks,
        ) is True
    print("PASS eval+masked still inserts backstories")


def test_add_backstories_gate_eval_no_subset_preserves_old_behavior():
    """Eval with eval_sys_subset=None must match training-path filtering."""
    back_frac, num_tasks = 0.25, 40000
    threshold = math.ceil(back_frac * num_tasks)
    assert should_insert_backstory(
        sys_choice=threshold - 1, sys_appear=[], real_seg_len=5,
        test=True, eval_sys_subset=None,
        back_frac=back_frac, num_tasks=num_tasks,
    ) is True
    assert should_insert_backstory(
        sys_choice=threshold, sys_appear=[], real_seg_len=5,
        test=True, eval_sys_subset=None,
        back_frac=back_frac, num_tasks=num_tasks,
    ) is False
    print("PASS eval with no subset preserves prior behavior")


def test_add_backstories_gate_sys_appear_and_zero_seg_still_block():
    """The subset bypass must not disable the other two guards (already-seen / empty segment)."""
    back_frac, num_tasks = 0.25, 40000
    threshold = math.ceil(back_frac * num_tasks)
    sys_choice = threshold + 5
    # already in sys_appear -> skip
    assert should_insert_backstory(
        sys_choice=sys_choice, sys_appear=[sys_choice], real_seg_len=5,
        test=True, eval_sys_subset="unmasked",
        back_frac=back_frac, num_tasks=num_tasks,
    ) is False
    # zero-length real segment -> skip
    assert should_insert_backstory(
        sys_choice=sys_choice, sys_appear=[], real_seg_len=0,
        test=True, eval_sys_subset="unmasked",
        back_frac=back_frac, num_tasks=num_tasks,
    ) is False
    print("PASS sys_appear / real_seg_len guards still block insertion")


if __name__ == "__main__":
    test_masked_enumeration_stays_below_threshold()
    test_unmasked_enumeration_stays_above_threshold_and_below_num_tasks()
    test_none_preserves_existing_behavior()
    test_zero_valid_examples_raises()
    test_back_frac_075_asymmetric()
    test_new_hay_insert_reduces_cap_by_one_and_stays_in_bounds()
    test_add_backstories_gate_training_keeps_back_frac_filter()
    test_add_backstories_gate_eval_unmasked_inserts_above_threshold()
    test_add_backstories_gate_eval_masked_unchanged()
    test_add_backstories_gate_eval_no_subset_preserves_old_behavior()
    test_add_backstories_gate_sys_appear_and_zero_seg_still_block()
    print("\nAll eval_sys_subset checks passed.")
