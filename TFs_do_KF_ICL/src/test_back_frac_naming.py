"""
Smoke test: assemble the experiment_name prefix block (training.py:79) and the
interleave_traces_dict_path tag block (data_train.py / conv_plots_funcs.py /
create_plots_with_zero_pred.py) under several configs and assert that
back_frac propagates correctly.

We do NOT invoke pytorch_lightning or the real Config singleton -- we just
mirror the literal expression used by those files using a plain stub config.
That keeps the test cheap and free of training-stack deps, while still
catching off-by-one ordering or missing tag mistakes in the propagation.
"""
from __future__ import annotations


class StubCfg:
    def __init__(self, **kw):
        # Defaults match config.py
        self.masking = True
        self.backstory = True
        self.iid_gaussian = False
        self.iid_gaussian_test = False
        self.backstory_test = False
        self.mask_only_init = False
        self.multi_sys_trace = True
        self.zero_cut = False
        self.dataset_typ = "ortho_haar"
        self.val_dataset_typ = "ortho_haar"
        self.C_dist = "_ident_C"
        self.nx = 5
        self.ny = 5
        self.backstory_len = 2
        self.back_frac = 0.5
        self.num_tasks = 40000
        self.learning_rate = 1.4766370475008905e-05
        self.datasource = "backstory_train"
        self.fix_needle = False
        self.opposite_ortho = False
        self.irrelevant_tokens = False
        self.same_tokens = False
        self.new_hay_insert = False
        self.paren_swap = False
        self.identical_haystack = False
        self.repeat_haystack = False
        for k, v in kw.items():
            setattr(self, k, v)


def build_experiment_name(config, train_mix_dist=False, train_mix_state_dim=False, timestamp="260502_120000.aaaaaa"):
    # Mirror of TFs_do_KF_ICL/src/core/training.py:79 after the back_frac edit.
    return (
        (f"back_frac_{config.back_frac}_" if config.masking and config.back_frac != 1.0 else "")
        + (f"back_len_{config.backstory_len}_" if config.masking and config.backstory_len != config.ny + 2 else "")
        + ("iid_gaussian_" if config.masking and config.iid_gaussian else "")
        + ("init_" if config.masking and config.mask_only_init else "")
        + timestamp
        + ("_multi_sys_trace" if config.multi_sys_trace else "")
        + ("_zero_cut" if config.zero_cut else "")
        + f"_{config.dataset_typ}_state_dim_{config.nx}{config.C_dist}"
        + ("_dist_mix" if train_mix_dist else "")
        + ("_state_dim_mix" if train_mix_state_dim else "")
        + "_lr_" + str(config.learning_rate)
        + "_num_train_sys_" + str(config.num_tasks)
    )


def build_interleave_path_tag_block(config, base_path="/BP/", interleaving="multi_cut"):
    # Mirror of data_train.py:1030-1031 / conv_plots_funcs.py:62-63 /
    # create_plots_with_zero_pred.py:1924-1925 after the back_frac edit.
    adds_backstories = (
        config.datasource == "backstory_train"
        or config.iid_gaussian_test
        or config.backstory_test
    )
    datasource_prefix = (
        config.datasource + "_init"
        if (adds_backstories and config.mask_only_init)
        else config.datasource
    )
    backstory_len_tag = (
        f"backlen_{config.backstory_len}_"
        if (config.backstory_len != config.ny + 2 and adds_backstories)
        else ""
    )
    back_frac_tag = (
        f"backfrac_{config.back_frac}_"
        if (config.back_frac != 1.0 and adds_backstories)
        else ""
    )
    return (
        f"{base_path}train_and_test_data/{config.dataset_typ}/"
        + f"{datasource_prefix}_"
        + back_frac_tag
        + backstory_len_tag
        + f"interleaved_traces_{config.dataset_typ}{config.C_dist}_{interleaving}_state_dim_{config.nx}.pkl"
    )


def chk(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    # 1. Default scenario: masking + back_frac=0.5 + backstory_len=2.
    cfg = StubCfg()
    name = build_experiment_name(cfg)
    print("\nexperiment_name (back_frac=0.5, back_len=2):")
    print("  " + name)
    chk(name.startswith("back_frac_0.5_back_len_2_"),
        "back_frac tag is the leading prefix, back_len second")
    chk("_lr_" in name and "_num_train_sys_40000" in name, "trailing structure preserved")

    # 2. back_frac=1.0 -> tag absent (back-compat with old naming).
    cfg = StubCfg(back_frac=1.0)
    name = build_experiment_name(cfg)
    print("\nexperiment_name (back_frac=1.0): " + name)
    chk("back_frac_" not in name, "back_frac tag suppressed at 1.0")

    # 3. masking=False -> tag absent (consistent with back_len gating).
    cfg = StubCfg(masking=False)
    name = build_experiment_name(cfg)
    chk("back_frac_" not in name, "back_frac tag suppressed when masking off")

    # 4. back_frac=0.25 + backstory_len at default (ny+2=7) -> only back_frac.
    cfg = StubCfg(back_frac=0.25, backstory_len=7)
    name = build_experiment_name(cfg)
    print("\nexperiment_name (back_frac=0.25, back_len default): " + name)
    chk(name.startswith("back_frac_0.25_") and "back_len_" not in name,
        "back_frac present, back_len suppressed at default")

    # 5. Data path: backstory_train datasource -> backfrac_X_ tag present.
    cfg = StubCfg()
    p = build_interleave_path_tag_block(cfg)
    print("\ndata path (backstory_train, back_frac=0.5):\n  " + p)
    chk("backfrac_0.5_" in p and "backlen_2_" in p,
        "data path carries both backfrac and backlen tags")
    chk(p.index("backstory_train_") < p.index("backfrac_0.5_") < p.index("backlen_2_"),
        "tag order: datasource_prefix, backfrac, backlen")

    # 6. Data path: datasource not eligible -> tag absent.
    cfg = StubCfg(datasource="val")
    p = build_interleave_path_tag_block(cfg)
    print("\ndata path (datasource=val):\n  " + p)
    chk("backfrac_" not in p, "no backfrac tag when datasource not adds_backstories")

    # 7. Data path: back_frac=1.0 -> tag absent.
    cfg = StubCfg(back_frac=1.0)
    p = build_interleave_path_tag_block(cfg)
    chk("backfrac_" not in p, "no backfrac tag at back_frac=1.0")

    # 8. Data path: iid_gaussian_test=True (different adds_backstories trigger).
    cfg = StubCfg(datasource="val", iid_gaussian_test=True)
    p = build_interleave_path_tag_block(cfg)
    chk("backfrac_0.5_" in p, "backfrac tag present for iid_gaussian_test path")

    # 9. Data path: backstory_test=True.
    cfg = StubCfg(datasource="val", backstory_test=True)
    p = build_interleave_path_tag_block(cfg)
    chk("backfrac_0.5_" in p, "backfrac tag present for backstory_test path")

    # 10. mask_only_init=True -> datasource_prefix gets _init suffix; backfrac still present.
    cfg = StubCfg(mask_only_init=True)
    p = build_interleave_path_tag_block(cfg)
    print("\ndata path (mask_only_init=True):\n  " + p)
    chk("backstory_train_init_backfrac_0.5_backlen_2_" in p,
        "mask_only_init combines with backfrac/backlen in correct order")

    print("\nAll naming checks passed.")


if __name__ == "__main__":
    main()
