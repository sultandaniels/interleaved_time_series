"""Output-path tag helpers for eval-time subset evaluation.

`eval_sys_subset` (defined in src/core/config.py) restricts needle-in-haystack
evaluation to the masked or unmasked half of the training systems. To keep the
resulting pickles and figures from clobbering each other, every output-path
construction site appends a filename tag and/or inserts a subdirectory. These
helpers centralize that logic so call sites stay short.
"""


def sys_subset_filename_tag(config) -> str:
    s = getattr(config, "eval_sys_subset", None)
    return f"sys_subset_{s}_" if s else ""


def sys_subset_figure_subdir(config) -> str:
    s = getattr(config, "eval_sys_subset", None)
    return f"sys_subset_{s}/" if s else ""
