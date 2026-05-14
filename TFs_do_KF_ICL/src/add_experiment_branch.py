"""Generate a new model_name elif branch in data_train.py from a saved configs/config.py.

Usage:
    python src/add_experiment_branch.py \\
        --experiment-dir outputs/GPT2/<dir-name> \\
        --model-name <name> \\
        [--data-train src/data_train.py] \\
        [--num-val-tasks 100] \\
        [--skip-fields ckpt_path,seed,...] \\
        [--dry-run] [--force]

Parses the experiment's saved Config class with `ast`, rewrites bare attribute
references inside value expressions to `config.<attr>`, and emits a fresh elif
block in the style used elsewhere in data_train.py. Also appends the new
model_name to the multi-name conditional that selects the checkpoint
prediction schedule (the line containing "ortho_haar_big_mask_backstory_no_leak").
"""
from __future__ import annotations

import argparse
import ast
import os
import sys


DEFAULT_SKIP_FIELDS = (
    "ckpt_path",
    "seed",
    "fully_reproducible",
    "eval_mask_only_init",
    "eval_backstory_len",
    "datasource",
)

ELIF_INDENT = "    "
BODY_INDENT = "        "
CKPT_ANCHOR = '"ortho_haar_big_mask_backstory_no_leak" or model_name =='
ELSE_ANCHOR = 'raise ValueError("Model name not recognized'


def parse_config_class(config_path: str):
    """Return [(attr_name, value_ast_node), ...] in source order from the Config class."""
    with open(config_path, "r") as f:
        source = f.read()
    tree = ast.parse(source)

    config_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            config_cls = node
            break
    if config_cls is None:
        raise ValueError(f"No `class Config` found in {config_path}")

    fields = []
    for stmt in config_cls.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                fields.append((stmt.targets[0].id, stmt.value))
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.value is not None:
                fields.append((stmt.target.id, stmt.value))
    return fields


class AttrRewriter(ast.NodeTransformer):
    """Rewrite `Name(id=x)` -> `config.x` when x is in the set of Config attrs."""

    def __init__(self, attr_names):
        self.attr_names = attr_names

    def visit_Name(self, node: ast.Name):
        if node.id in self.attr_names and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="config", ctx=ast.Load()),
                    attr=node.id,
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


def render_override(attr: str, value_node: ast.AST, attr_names) -> str:
    rewritten = AttrRewriter(attr_names).visit(ast.parse(ast.unparse(value_node), mode="eval").body)
    ast.fix_missing_locations(rewritten)
    rendered = ast.unparse(rewritten)
    return f'{BODY_INDENT}config.override("{attr}", {rendered})'


def build_elif_block(model_name: str, experiment_name: str, fields, num_val_tasks: int) -> str:
    attr_names = {name for name, _ in fields}
    attr_names.add("num_val_tasks")

    forced = {"num_val_tasks": ast.Constant(value=num_val_tasks)}

    seen = set()
    ordered_fields = []
    for name, value in fields:
        if name in forced:
            ordered_fields.append((name, forced[name]))
        else:
            ordered_fields.append((name, value))
        seen.add(name)
    for name, value in forced.items():
        if name not in seen:
            ordered_fields.append((name, value))

    lines = [
        f'{ELIF_INDENT}elif model_name == "{model_name}":',
        f'{BODY_INDENT}experiment_name = "{experiment_name}"',
        "",
        f'{BODY_INDENT}print("\\n\\n{model_name.upper()}\\n\\n")',
        "",
        f"{BODY_INDENT}# Overrides generated from configs/config.py for {experiment_name}",
    ]
    for name, value in ordered_fields:
        lines.append(render_override(name, value, attr_names))
    lines.append("")
    return "\n".join(lines) + "\n"


def insert_elif(data_train_source: str, elif_block: str) -> str:
    lines = data_train_source.splitlines(keepends=True)
    target_idx = None
    for i, line in enumerate(lines):
        if ELSE_ANCHOR in line:
            for j in range(i - 1, -1, -1):
                if lines[j].rstrip("\n").rstrip() == f"{ELIF_INDENT}else:":
                    target_idx = j
                    break
            break
    if target_idx is None:
        raise RuntimeError(f"Could not locate `{ELIF_INDENT}else:` preceding `{ELSE_ANCHOR}` in data_train.py")

    block_with_trailing_blank = elif_block if elif_block.endswith("\n\n") else elif_block + "\n"
    new_lines = lines[:target_idx] + [block_with_trailing_blank] + lines[target_idx:]
    return "".join(new_lines)


def update_ckpt_chain(data_train_source: str, model_name: str) -> str:
    needle = f'model_name == "{model_name}"'
    lines = data_train_source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if CKPT_ANCHOR in line:
            if needle in line:
                return data_train_source
            stripped = line.rstrip("\n")
            trailing_nl = line[len(stripped):]
            if not stripped.rstrip().endswith(":"):
                raise RuntimeError(f"Expected ckpt-chain line to end with ':', got: {stripped!r}")
            colon_pos = stripped.rfind(":")
            new_line = stripped[:colon_pos] + f' or model_name == "{model_name}"' + stripped[colon_pos:] + trailing_nl
            lines[i] = new_line
            return "".join(lines)
    raise RuntimeError(f"Could not locate ckpt-chain anchor ({CKPT_ANCHOR!r}) in data_train.py")


def already_has_branch(data_train_source: str, model_name: str) -> bool:
    return f'elif model_name == "{model_name}":' in data_train_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to outputs/<model_type>/<experiment-name> directory.")
    parser.add_argument("--model-name", required=True,
                        help="The --model_name value to register for this experiment.")
    parser.add_argument("--data-train",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_train.py"),
                        help="Path to src/data_train.py.")
    parser.add_argument("--num-val-tasks", type=int, default=100,
                        help="Forced override for num_val_tasks (default: 100).")
    parser.add_argument("--skip-fields", default=",".join(DEFAULT_SKIP_FIELDS),
                        help="Comma-separated Config attributes to omit from overrides.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the generated elif block and updated chain line; don't write.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing branch with the same model_name.")
    args = parser.parse_args(argv)

    config_path = os.path.join(args.experiment_dir, "configs", "config.py")
    if not os.path.isfile(config_path):
        sys.exit(f"error: config not found at {config_path}")

    experiment_name = os.path.basename(os.path.normpath(args.experiment_dir))
    skip_fields = {s.strip() for s in args.skip_fields.split(",") if s.strip()}

    fields = parse_config_class(config_path)
    fields = [(name, value) for name, value in fields if name not in skip_fields]

    elif_block = build_elif_block(args.model_name, experiment_name, fields, args.num_val_tasks)

    with open(args.data_train, "r") as f:
        data_train_source = f.read()

    branch_exists = already_has_branch(data_train_source, args.model_name)
    wrote_elif = False
    if branch_exists and not args.force:
        print(f"warning: branch for model_name={args.model_name!r} already exists; skipping (use --force to overwrite).",
              file=sys.stderr)
    else:
        if branch_exists and args.force:
            print(f"warning: --force is set but in-place branch replacement is not implemented; "
                  f"please remove the existing branch first.", file=sys.stderr)
            sys.exit(2)
        data_train_source = insert_elif(data_train_source, elif_block)
        wrote_elif = True

    pre_chain_source = data_train_source
    data_train_source = update_ckpt_chain(data_train_source, args.model_name)
    chain_updated = data_train_source != pre_chain_source

    if args.dry_run:
        print("---- elif block ----")
        print(elif_block)
        print("---- ckpt chain line (after) ----")
        for line in data_train_source.splitlines():
            if CKPT_ANCHOR in line:
                print(line)
                break
        return

    if not wrote_elif and not chain_updated:
        print(f"no changes: {args.model_name!r} already present in both elif chain and ckpt chain")
        return

    with open(args.data_train, "w") as f:
        f.write(data_train_source)
    parts = []
    if wrote_elif:
        parts.append("inserted elif branch")
    if chain_updated:
        parts.append("appended to ckpt chain")
    print(f"{' and '.join(parts)} for {args.model_name!r} in {args.data_train}")


if __name__ == "__main__":
    main()
