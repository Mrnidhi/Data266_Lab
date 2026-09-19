"""Run a measured first segment or resume CycleGAN on an allocated Colab GPU.

Invoke from the complete repository with its Python environment. No allocation,
billing, account login, or competition submission is performed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from lab1.run import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until-step', type=int, help='Absolute update limit; omit to complete the configured schedule.')
    args = parser.parse_args()
    if args.until_step is not None and args.until_step < 1:
        parser.error('--until-step must be positive')
    overrides = [f'max_steps={args.until_step}'] if args.until_step else []
    started = time.perf_counter()
    result = execute('cyclegan', 'full', args.output, 'cuda', overrides=overrides, resume=args.resume)
    rows = [json.loads(line) for line in (args.output / 'training_log.jsonl').read_text().splitlines() if line.strip()]
    stable = rows[-min(100, max(1, len(rows) - 10)):]
    summary = result['summary']
    total_updates = result['config']['epochs'] * summary['steps_per_epoch']
    mean_step_seconds = statistics.mean(row['step_seconds'] for row in stable)
    receipt = {'completed_updates': summary['completed_updates'], 'target_updates': total_updates,
               'full_schedule_completed': summary['completed_updates'] == total_updates,
               'segment_wall_seconds': time.perf_counter() - started,
               'recent_mean_training_step_seconds': mean_step_seconds,
               'remaining_training_hours_estimate': (total_updates - summary['completed_updates']) * mean_step_seconds / 3600,
               'estimate_note': 'Measured training-step extrapolation only; data loading, validation, checkpoint transfers and export add time.',
               'peak_gpu_memory_gib': summary['peak_gpu_memory_bytes'] / 1024**3,
               'nan_events': summary['nan_events']}
    (args.output / 'segment_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
