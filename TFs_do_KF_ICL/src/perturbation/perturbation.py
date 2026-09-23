"""Quick inference script: run a trained GPT2 checkpoint on pre-built ortho_haar
needle-in-haystack val traces.

Loads a checkpoint and a pre-built interleaved-traces pickle (the same format
consumed by create_plots_with_zero_pred.compute_errors_needle_or_multi_cut),
runs the model over it via the existing tf_preds() helper, and writes out
model predictions, targets, and a per-timestep ("index by index") MSE curve.
"""
import argparse
import os
import pickle
import re
import sys
import time
import matplotlib.pyplot as plt
import numpy.linalg as lin

BASE_PATH = "/work/hdd/benv/sdaniels2/ICL_Kalman_Experiments/"
os.environ.setdefault("BASE_PATH", BASE_PATH)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch

from core import Config
from models import GPT2
from create_plots_with_zero_pred import tf_preds
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

DEFAULT_CKPT_PATH = (
    BASE_PATH
    + "model_checkpoints/GPT2/250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5"
      "_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000/checkpoints/step=135000.ckpt"
)
DEFAULT_TRACES_PATH = os.path.join(
    BASE_PATH, "train_and_test_data", "ortho_haar",
    "val_interleaved_traces_ortho_haar_ident_C_haystack_len_1_state_dim_5.pkl",
)
TRAIN_TRACES_PATH = os.path.join(
    BASE_PATH, "train_and_test_data", "ortho_haar",
    "train_interleaved_traces_ortho_haar_ident_C_haystack_len_1_state_dim_5.pkl",
)


def scale_observations(multi_sys_ys, alpha):
    """Scale the last 5 columns of axis -1 (the observation dims) by alpha."""
    scaled = multi_sys_ys.copy()
    scaled[..., -5:] *= alpha
    return scaled


def default_output_dir(ckpt_path):
    step_match = re.search(r"step=(\d+)", os.path.basename(ckpt_path))
    step = step_match.group(1) if step_match else "unknown"
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    return os.path.join(BASE_PATH, "model_outputs", f"{run_dir}_step={step}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--dataset", type=str, choices=["val", "train"], default="val",
                         help="Which pre-built traces set to run on; picks traces_path unless --traces_path is explicitly given.")
    parser.add_argument("--traces_path", type=str, default=None,
                         help="Pickle of pre-built interleaved traces (multi_sys_ys, ...). "
                              "Defaults to the val or train file for --dataset.")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Limit number of traces for a quick test run. Default: all traces.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(config, ckpt_path, device):
    config.override("ckpt_path", ckpt_path)
    return GPT2.load_from_checkpoint(
        config.ckpt_path,
        n_dims_in=config.n_dims_in, n_positions=config.n_positions,
        n_dims_out=config.n_dims_out, n_embd=config.n_embd,
        n_layer=config.n_layer, n_head=config.n_head,
        use_pos_emb=config.use_pos_emb, map_location=device, strict=True,
    ).eval().to(device)


def load_multi_sys_ys(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    multi_sys_ys = data["multi_sys_ys"]
    print(f"Loaded multi_sys_ys from {path} with shape {multi_sys_ys.shape}")
    # flatten (num_exs, num_trace_configs, num_trials, ...) into one leading
    # per-trace batch axis so max_examples slices individual traces
    return multi_sys_ys


def run_inference(model, multi_sys_ys, device, config, max_examples=None):
    if max_examples is not None:
        multi_sys_ys = multi_sys_ys[:max_examples]

    preds = tf_preds(multi_sys_ys, model, device, config)

    return preds

def compute_quartiles_median_squared_error(config, preds, multi_sys_ys):
    """Compute the median of the median squared error."""
    #take the last config.ny columns of axis=-1 as the true test observations
    multi_sys_ys_true = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-1] - config.ny, multi_sys_ys.shape[-1]), axis=-1) #get the true test observations

    print(f"multi_sys_ys_true shape: {multi_sys_ys_true.shape}")

    errs = np.linalg.norm((multi_sys_ys_true - preds), axis=-1) ** 2  # get the errors of transformer predictions

    print(f"errs shape: {errs.shape}")

    med_se = np.median(errs, axis=-2)  # get the median squared error over traces
    print(f"med_se shape: {med_se.shape}")
    quart_med_se = np.quantile(med_se,[0.25, 0.50, 0.75], axis=0)  # get the median over systems
    #remove axis 1 from quart_med_se
    quart_med_se = np.squeeze(quart_med_se, axis=1)
    print(f"quart_med_se shape: {quart_med_se.shape}")
    return quart_med_se

def plot_median_squared_error_vs_index(quart_med_se , quart_med_se_31000, dataset=""):
    """Plot the median of the median squared error vs index."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    max_ind = 7
    indices = np.arange(max_ind)
    ax.plot(indices, quart_med_se[1,:max_ind], marker="o", color="black", label="step=135000")
    ax.fill_between(indices, quart_med_se[0,:max_ind], quart_med_se[2,:max_ind], color="gray", alpha=0.35)
    ax.plot(indices, quart_med_se_31000[1, :max_ind], marker="o", color="blue", label="step=31000")
    ax.fill_between(indices, quart_med_se_31000[0, :max_ind], quart_med_se_31000[2, :max_ind], color="blue", alpha=0.35)
    ax.set_xlabel("Index")
    ax.set_ylabel("Squared Error")
    ax.legend()
    fig.tight_layout()

    os.makedirs("perturbation/medse_vs_index_plots", exist_ok=True)
    plt.savefig(f"perturbation/medse_vs_index_plots/{dataset}_medse_vs_index.pdf", format="pdf", bbox_inches="tight")


def plot_median_squared_error_vs_index_alphas(quart_med_se_by_alpha, quart_med_se_31000, dataset=""):
    """Plot median squared error vs index for multiple alpha-scaled observation sets."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    max_ind = 7
    indices = np.arange(max_ind)
    alphas = sorted(quart_med_se_by_alpha.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for alpha, color in zip(alphas, colors):

        if alpha == 1.0:
            color = "red"  # Highlight the original alpha=1.0 case
        quart_med_se = quart_med_se_by_alpha[alpha]
        ax.plot(indices, quart_med_se[1, :max_ind], marker="o", color=color, label=f"a={alpha}")
        ax.fill_between(indices, quart_med_se[0, :max_ind], quart_med_se[2, :max_ind], color=color, alpha=0.2)

    ax.plot(indices, quart_med_se_31000[1, :max_ind], marker="o", color="blue", label="step=31000")
    ax.fill_between(indices, quart_med_se_31000[0, :max_ind], quart_med_se_31000[2, :max_ind], color="blue", alpha=0.2)

    ax.set_xlabel("Index")
    ax.set_ylabel("Squared Error")
    ax.legend()
    fig.tight_layout()

    os.makedirs("perturbation/medse_vs_index_plots", exist_ok=True)
    plt.savefig(f"perturbation/medse_vs_index_plots/{dataset}_medse_vs_index_alphas.pdf", format="pdf", bbox_inches="tight")

def construct_Q(x, c):
    """Deterministically construct a real orthogonal Q such that x^T Q x = c."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    norm_x = lin.norm(x)
    assert norm_x > 0, "x must be nonzero"
    assert abs(c) <= norm_x**2 + 1e-8, "c must satisfy |c| <= ||x||^2 (Cauchy-Schwarz)"

    u = x/norm_x # unit vector of initial state
    t = c/norm_x**2 # scaling factor for the perturbation along the direction of x

    if n == 1:
        assert abs(abs(t) - 1) < 1e-8, "in 1D, c must equal +-||x||^2"
        return np.array([[t]])

    # deterministically pick a unit vector v orthogonal to u: Gram-Schmidt the
    # first standard basis vector that isn't parallel to u
    v = None
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        candidate = e_j - (u @ e_j) * u
        candidate_norm = lin.norm(candidate)
        if candidate_norm > 1e-8:
            v = candidate/candidate_norm
            break

    s = np.sqrt(max(1 - t**2, 0.0)) # clamp for float error when |t| == 1
    Q = np.eye(n) + (t - 1)*(np.outer(u, u) + np.outer(v, v)) + s*(np.outer(v, u) - np.outer(u, v))
    return Q


def main():
    start = time.time()
    args = parse_args()
    end = time.time()
    print(f"Parsed args in {end - start:.2f} seconds: {args}\n")
    output_dir = args.output_dir or default_output_dir(args.ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    config = Config()

    start = time.time()
    model = load_model(config, args.ckpt_path, args.device)
    end = time.time()
    print(f"Loaded model from {args.ckpt_path} onto {args.device} in {end - start:.2f} seconds\n")

    traces_path = args.traces_path or (DEFAULT_TRACES_PATH if args.dataset == "val" else TRAIN_TRACES_PATH)

    start = time.time()
    multi_sys_ys = load_multi_sys_ys(traces_path)
    end = time.time()
    print(f"Loaded {multi_sys_ys.shape[0]*multi_sys_ys.shape[2]} traces from {traces_path} in {end - start:.2f} seconds\n")

    start = time.time()
    preds = run_inference(model, multi_sys_ys, args.device, config, args.max_examples)
    end = time.time()
    print(f"Ran inference on {preds.shape[0]*preds.shape[2]} traces in {end - start:.2f} seconds\n")

    np.save(os.path.join(output_dir, "preds.npy"), preds)

    quart_med_se = compute_quartiles_median_squared_error(config, preds, multi_sys_ys)

    print(f"preds shape: {preds.shape}\n")
    print(f"Wrote outputs to {output_dir}\n")

    alpha_step = 0.1
    alpha_scale = round(1 / alpha_step)
    alphas = np.arange(round(0.5 * alpha_scale), round(1.5 * alpha_scale) + 1) / alpha_scale
    print(f"\nalphas: {alphas}\n")

    quart_med_se_by_alpha = {1.0: quart_med_se}

    for alpha in alphas:
        scaled_multi_sys_ys = scale_observations(multi_sys_ys, alpha)
        scaled_preds = run_inference(model, scaled_multi_sys_ys, args.device, config, args.max_examples)
        quart_med_se_by_alpha[alpha] = compute_quartiles_median_squared_error(config, scaled_preds, scaled_multi_sys_ys)

    os.makedirs("perturbation/medse_vs_index_plots", exist_ok=True)
    #save quart_med_se_by_alpha
    np.savez("perturbation/medse_vs_index_plots/quart_med_se_by_alpha.npz", **quart_med_se_by_alpha)
    #save alphas
    np.save("perturbation/medse_vs_index_plots/alphas.npy", alphas)

    #ckpt_path currently for step=135000. change the step number to 31000
    ckpt_path = args.ckpt_path.replace("step=135000", "step=31000")
    output_dir = default_output_dir(ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    #load model
    model_31000 = load_model(config, ckpt_path, args.device)

    start = time.time()
    preds_31000 = run_inference(model_31000, multi_sys_ys, args.device, config, args.max_examples)
    end = time.time()
    print(f"Ran inference on {preds_31000.shape[0]*preds_31000.shape[2]} traces in {end - start:.2f} seconds\n")

    quart_med_se_31000 = compute_quartiles_median_squared_error(config, preds_31000, multi_sys_ys)

    plot_median_squared_error_vs_index(quart_med_se, quart_med_se_31000, dataset=args.dataset)
    plot_median_squared_error_vs_index_alphas(quart_med_se_by_alpha, quart_med_se_31000, dataset=args.dataset)


if __name__ == "__main__":
    main()
