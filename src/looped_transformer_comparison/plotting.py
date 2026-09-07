"""Headless live plots for flushed training metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _records(path: Path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get('step'), int):
            records.append(record)
    return records


def _metadata(run: Path):
    path = run / 'metadata.json'
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _plot_run(run: Path, records, dpi=130):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not records:
        return False
    meta = _metadata(run)
    config = meta.get('config', {})
    training = config.get('training', {})
    model = config.get('model', {})
    tokens_per_step = (training.get('batch_size', 0) * training.get('grad_accum', 0)
                       * model.get('seq_len', 0))
    steps = [r['step'] for r in records]
    train_loss = [r.get('train_loss') for r in records]
    learning_rate = [r.get('lr') for r in records]
    val_records = [r for r in records if isinstance(r.get('validation'), dict)]
    val_steps = [r['step'] for r in val_records]
    val_loss = [r['validation'].get('loss') for r in val_records]
    val_ppl = [r['validation'].get('perplexity') for r in val_records]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    title = meta.get('architecture', run.name)
    fig.suptitle(f'{title} training stability — through step {steps[-1]:,}', fontsize=14)
    ax = axes[0, 0]
    ax.plot(steps, train_loss, color='#2563eb', linewidth=1, label='training loss')
    if val_steps:
        ax.plot(val_steps, val_loss, 'o-', color='#dc2626', markersize=3, label='validation loss')
    ax.set(xlabel='optimizer step', ylabel='cross-entropy loss', title='Loss')
    ax.legend()
    ax.grid(alpha=.25)
    ax = axes[0, 1]
    if val_steps:
        ax.plot(val_steps, val_ppl, 'o-', color='#7c3aed', markersize=3)
    ax.set(xlabel='optimizer step', ylabel='BPE perplexity', title='Validation perplexity')
    ax.grid(alpha=.25)
    ax = axes[1, 0]
    ax.plot(steps, learning_rate, color='#059669', linewidth=1)
    ax.set(xlabel='optimizer step', ylabel='learning rate', title='Learning-rate schedule')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    ax.grid(alpha=.25)
    ax = axes[1, 1]
    if tokens_per_step > 0:
        ax.plot(steps, [s * tokens_per_step for s in steps], color='#ea580c', linewidth=1)
        ax.set_ylabel('training tokens')
    else:
        ax.plot(steps, steps, color='#ea580c', linewidth=1)
        ax.set_ylabel('optimizer steps')
    ax.set(xlabel='optimizer step', title='Training progress')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(alpha=.25)
    output = run / 'training.png'
    temp = output.with_suffix('.tmp.png')
    fig.savefig(temp, dpi=dpi)
    plt.close(fig)
    temp.replace(output)
    return True


def _plot_comparison(root: Path, runs, dpi=130):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not any(records for records in runs.values()):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle('Standard vs looped training stability', fontsize=14)
    colors = {'standard': '#2563eb', 'looped': '#dc2626'}
    for arch, records in runs.items():
        if not records:
            continue
        steps = [r['step'] for r in records]
        vals = [r.get('train_loss') for r in records]
        validation = [r for r in records if isinstance(r.get('validation'), dict)]
        vsteps = [r['step'] for r in validation]
        axes[0, 0].plot(steps, vals, color=colors[arch], linewidth=1, label=f'{arch} training')
        if vsteps:
            axes[0, 0].plot(vsteps, [r['validation'].get('loss') for r in validation], 'o',
                            color=colors[arch], markersize=3, label=f'{arch} validation')
            axes[0, 1].plot(vsteps, [r['validation'].get('perplexity') for r in validation],
                            'o-', color=colors[arch], markersize=3, label=arch)
        axes[1, 0].plot(steps, [r.get('lr') for r in records], color=colors[arch], label=arch)
        meta = _metadata(root / arch)
        cfg = meta.get('config', {})
        train = cfg.get('training', {})
        model = cfg.get('model', {})
        tokens = train.get('batch_size', 0) * train.get('grad_accum', 0) * model.get('seq_len', 0)
        axes[1, 1].plot(steps, [s * tokens for s in steps] if tokens else steps,
                        color=colors[arch], label=arch)
    axes[0, 0].set(xlabel='optimizer step', ylabel='cross-entropy loss', title='Loss')
    axes[0, 1].set(xlabel='optimizer step', ylabel='BPE perplexity', title='Validation perplexity')
    axes[1, 0].set(xlabel='optimizer step', ylabel='learning rate', title='Learning-rate schedule')
    axes[1, 1].set(xlabel='optimizer step', ylabel='training tokens', title='Training progress')
    axes[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    axes[1, 1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    output = root / 'training-comparison.png'
    temp = output.with_suffix('.tmp.png')
    fig.savefig(temp, dpi=dpi)
    plt.close(fig)
    temp.replace(output)
    return True


def plot_runs(root, watch=False, interval=30.0, once=False, dpi=130):
    """Render per-architecture plots; with watch, update from flushed JSONL."""
    root = Path(root)
    if interval <= 0:
        raise ValueError('interval must be positive')
    try:
        while True:
            changed = False
            runs = {}
            for arch in ('standard', 'looped'):
                run = root / arch
                records = _records(run / 'metrics.jsonl')
                runs[arch] = records
                changed = _plot_run(run, records, dpi) or changed
            changed = _plot_comparison(root, runs, dpi) or changed
            if not watch or once:
                return {'output': str(root), 'updated': changed}
            time.sleep(interval)
    except KeyboardInterrupt:
        return {'output': str(root), 'stopped': True}
