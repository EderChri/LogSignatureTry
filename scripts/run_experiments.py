"""run_experiments.py — run a full pretrain + finetune pipeline from a JSON config.

Sequential (default):
    python scripts/run_experiments.py experiments.json

Parallel across GPUs, seed-by-seed:
    python scripts/run_experiments.py experiments.json --gpus 6,7,8

  For each seed the runner:
    1. Starts all pretrain jobs in parallel (one per free GPU).
    2. As each pretrain finishes its GPU immediately picks up that dataset's
       finetune jobs — no waiting for other pretrains to finish.
    3. Moves on to the next seed only when every job for this seed is done.

  GPU pool acts as the throttle: at most N jobs run at once (N = len(gpus)).

Dry-run (print what would run, don't execute):
    python scripts/run_experiments.py experiments.json --gpus 6,7,8 --dry-run

Config format (experiments.json):
    {
      "experiments": [
        {
          "name": "optional label",
          "enabled": true,           # set false to keep but skip
          "pretrain": {
            "data_name": "_DA_HARTH_256_00",
            "num_feature": 6,
            "num_target": 12
          },
          "finetune_targets": [
            {"data_name": "_DA_HAR70plus_256_00", "num_feature": 6, "num_target": 7}
          ],
          "fixed": {
            "view2": "logsig",
            "epochs_pretrain": 200,
            "epochs_finetune": 100
          },
          "grid": {
            "logsig_mode": ["stream", "window"],
            "seed": [0, 1, 2]
          }
        }
      ]
    }

Rules:
  - "fixed" params go to both pretrain and finetune unchanged.
  - "grid" params are expanded as a Cartesian product.
  - Pretrain is deduplicated: same config+seed only trains once, even if
    multiple finetune targets share that checkpoint.
  - Both scripts skip runs whose output file already exists, so the whole
    config is safe to re-run after a crash.
"""

import argparse
import itertools
import json
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of all grid values (scalars treated as length-1 lists)."""
    if not grid:
        return [{}]
    keys   = list(grid.keys())
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def args_to_flags(params: dict) -> list[str]:
    """Convert a param dict to a flat CLI flag list."""
    flags = []
    for k, v in params.items():
        if v is None or v is False:
            continue
        flags.append(f'--{k}')
        if v is not True:
            flags.append(str(v))
    return flags


def pretrain_key(params: dict) -> frozenset:
    return frozenset(params.items())


def build_pipelines(cfg: dict, python: str,
                    seed_filter=None) -> list[tuple[str, list[str]]]:
    """Return [(pt_cmd, [ft_cmd, ...]), ...] for enabled experiments.

    seed_filter: if set, only include combos where grid['seed'] == seed_filter.
    Pretrain commands are deduplicated: one pt_cmd entry per unique pretrain config,
    with all its finetune commands collected under it.
    """
    seen: dict[frozenset, list[str]] = {}
    order: list[tuple[frozenset, str]] = []   # keeps insertion order

    for exp in cfg['experiments']:
        if not exp.get('enabled', True):
            continue
        pretrain_data    = exp['pretrain']
        finetune_targets = exp.get('finetune_targets', [])
        fixed            = exp.get('fixed', {})

        for combo in expand_grid(exp.get('grid', {})):
            if seed_filter is not None and combo.get('seed') != seed_filter:
                continue

            params    = {**fixed, **combo}
            pt_params = {**pretrain_data, **params}
            pk        = pretrain_key(pt_params)
            pt_cmd    = ' '.join([python, 'scripts/run_pretrain.py']
                                 + args_to_flags(pt_params))

            if pk not in seen:
                seen[pk] = []
                order.append((pk, pt_cmd))

            for target in finetune_targets:
                ft_params = {
                    **target,
                    'pretrain_data_name': pretrain_data['data_name'],
                    **params,
                }
                seen[pk].append(
                    ' '.join([python, 'scripts/run_finetune.py']
                             + args_to_flags(ft_params))
                )

    return [(pt_cmd, seen[pk]) for pk, pt_cmd in order]


# ── Execution ─────────────────────────────────────────────────────────────────

def run_parallel_pipelines(pipelines: list[tuple[str, list[str]]],
                           gpus: list[str],
                           dry_run: bool = False):
    """Run pipelines using a shared GPU pool.

    A pipeline is (pretrain_cmd, [finetune_cmds]).
    Each pipeline holds its assigned GPU for the full pretrain → finetune(s)
    chain, only releasing when all jobs in the pipeline are done.
    This guarantees finetune always runs immediately after its pretrain on
    the same GPU, never blocked by another pipeline's pretrain grabbing the GPU.

    At most len(gpus) pipelines run concurrently.
    """
    gpu_pool: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    def _exec(cmd: str, gpu: str):
        if dry_run:
            print(f'[dry-run GPU {gpu}] {cmd}', flush=True)
        else:
            print(f'[GPU {gpu}] {cmd}', flush=True)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            subprocess.run(cmd, shell=True, env=env, check=True)

    def run_pipeline(pt_cmd: str, ft_cmds: list[str]):
        gpu = gpu_pool.get()
        try:
            _exec(pt_cmd, gpu)
            for ft_cmd in ft_cmds:
                _exec(ft_cmd, gpu)
        finally:
            gpu_pool.put(gpu)

    # max_workers = len(gpus): at most one pipeline per GPU at a time.
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        pipeline_futures = [
            executor.submit(run_pipeline, pt, fts)
            for pt, fts in pipelines
        ]
        for f in as_completed(pipeline_futures):
            f.result()


def run_sequential(cmd: list[str], dry_run: bool = False):
    label = ' '.join(cmd)
    if dry_run:
        print(f'[dry-run] {label}')
    else:
        print(f'\n>>> {label}', flush=True)
        subprocess.run(cmd, check=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Ensure child processes can import src/ even without `pip install -e .`
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _existing_pp = os.environ.get('PYTHONPATH', '')
    if _root not in _existing_pp.split(':'):
        os.environ['PYTHONPATH'] = f"{_root}:{_existing_pp}".strip(':')

    # Fix scipy CXXABI_1.3.15 error: conda env provides a newer libstdc++
    _conda_lib = '/opt/conda/envs/myenv/lib'
    _existing_ldp = os.environ.get('LD_LIBRARY_PATH', '')
    if _conda_lib not in _existing_ldp.split(':'):
        os.environ['LD_LIBRARY_PATH'] = f"{_conda_lib}:{_existing_ldp}".strip(':')

    parser = argparse.ArgumentParser(
        description='Run pretrain+finetune experiments from a JSON config.')
    parser.add_argument('config', help='Path to experiments JSON file')
    parser.add_argument('--gpus', default=None,
                        help='Comma-separated GPU IDs for parallel mode, e.g. --gpus 6,7,8')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    parser.add_argument('--python', default='python',
                        help='Python interpreter (default: python)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # ── Parallel mode ─────────────────────────────────────────────────────────
    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(',')]
        pipelines = build_pipelines(cfg, args.python)
        run_parallel_pipelines(pipelines, gpus, args.dry_run)
        return

    # ── Sequential mode ───────────────────────────────────────────────────────
    seen_pretrain: set[frozenset] = set()

    for exp in cfg['experiments']:
        name = exp.get('name', '(unnamed)')
        if not exp.get('enabled', True):
            print(f'\n[skipped] {name}')
            continue
        print(f'\n{"="*60}\nExperiment: {name}\n{"="*60}', flush=True)

        pretrain_data    = exp['pretrain']
        finetune_targets = exp.get('finetune_targets', [])
        fixed            = exp.get('fixed', {})

        for combo in expand_grid(exp.get('grid', {})):
            params    = {**fixed, **combo}
            pt_params = {**pretrain_data, **params}
            pk        = pretrain_key(pt_params)

            if pk not in seen_pretrain:
                seen_pretrain.add(pk)
                run_sequential(
                    [args.python, 'scripts/run_pretrain.py'] + args_to_flags(pt_params),
                    args.dry_run,
                )

            for target in finetune_targets:
                ft_params = {
                    **target,
                    'pretrain_data_name': pretrain_data['data_name'],
                    **params,
                }
                run_sequential(
                    [args.python, 'scripts/run_finetune.py'] + args_to_flags(ft_params),
                    args.dry_run,
                )


if __name__ == '__main__':
    main()
