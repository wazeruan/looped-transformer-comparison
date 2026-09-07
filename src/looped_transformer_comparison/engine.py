import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import batch, load_data
from .model import LanguageModel, ModelConfig


def device_for(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; run scripts/check.sh')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def amp(device, precision):
    if precision == 'fp32':
        return contextlib.nullcontext()
    if precision != 'bf16' or device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('bf16 requires a supported CUDA GPU; use fp32 on CPU')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_config(path):
    c = json.loads(Path(path).read_text())
    ModelConfig(**c['model'])
    t = c['training']
    for k in ('steps', 'batch_size', 'grad_accum', 'eval_every', 'log_every'):
        if not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if not 0 <= t['warmup_steps'] < t['steps']:
        raise ValueError('warmup_steps must be between 0 and steps-1')
    if not 0 <= t['min_lr'] <= t['lr'] or t['lr'] <= 0 or t['grad_clip'] <= 0 or t['weight_decay'] < 0:
        raise ValueError('invalid optimizer settings')
    if t['precision'] not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    return c


def learning_rate(step, t):
    if step < t['warmup_steps']:
        return t['lr'] * (step + 1) / t['warmup_steps']
    progress = (step - t['warmup_steps']) / max(1, t['steps'] - t['warmup_steps'] - 1)
    return t['min_lr'] + 0.5 * (t['lr'] - t['min_lr']) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, stream, batch_size, device, precision):
    """Every next-token target exactly once; fixed non-overlapping contexts plus tail."""
    was_training = model.training
    model.eval()
    total_loss, count = 0.0, 0
    seq = model.config.seq_len
    full = (len(stream) - 1) // seq
    for first in range(0, full, batch_size):
        offsets = range(first * seq, min(first + batch_size, full) * seq, seq)
        a = torch.from_numpy(np.stack([stream[i:i + seq + 1].astype(np.int64) for i in offsets])).to(device)
        with amp(device, precision):
            logits = model(a[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1), reduction='sum')
        total_loss += loss.item()
        count += a[:, 1:].numel()
    if full * seq < len(stream) - 1:
        a = torch.from_numpy(np.array(stream[full * seq:], dtype=np.int64)).unsqueeze(0).to(device)
        with amp(device, precision):
            logits = model(a[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1), reduction='sum')
        total_loss += loss.item()
        count += a.size(1) - 1
    model.train(was_training)
    mean = total_loss / count
    if not math.isfinite(mean):
        raise RuntimeError('non-finite evaluation loss')
    return {'loss': mean, 'perplexity': math.exp(min(mean, 700)), 'tokens': count}


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    # Only load your own checkpoints. weights_only avoids arbitrary pickle execution.
    return torch.load(path, map_location='cpu', weights_only=True)


def train(config_path, data_dir, output, architecture, resume=False, stop_after=None, calibration=False):
    c = read_config(config_path)
    t = c['training']
    device = device_for(t['device'])
    with amp(device, t['precision']):
        pass
    streams, manifest = load_data(data_dir, c['model']['seq_len'])
    c['model']['vocab_size'] = manifest['vocab_size']
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError(f'{out} is not empty; use --resume or a new output directory')
    if resume and not (out / 'last.pt').exists():
        raise ValueError('resume requires last.pt')
    seed_all(t['seed'])
    model = LanguageModel(ModelConfig(**c['model']), architecture).to(device)
    seed_all(t['seed'] + 2000)  # Same stochastic-training RNG after unequal model construction.
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    generator = torch.Generator().manual_seed(t['seed'] + 1000)
    step, best_loss, train_seconds = 0, float('inf'), 0.0
    if resume:
        ck = load_checkpoint(out / 'last.pt')
        if ck['config'] != c or ck['manifest'] != manifest or ck['architecture'] != architecture or ck.get('calibration', False) != calibration:
            raise ValueError('resume config, data or architecture mismatch')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        step, best_loss, train_seconds = ck['step'], ck['best_loss'], ck['train_seconds']
        generator.set_state(ck['batch_rng'])
        torch.set_rng_state(ck['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if not (out / 'best.pt').exists():
            raise ValueError('resume requires best.pt alongside last.pt')
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'architecture': architecture, 'config': c, 'manifest': manifest,
                'parameters': sum(p.numel() for p in model.parameters()), 'effective_depth': model.config.depth,
                'unique_layers': len(model.blocks), 'device': str(device), 'torch': str(torch.__version__),
                'python': platform.python_version(), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    target = t['steps'] if stop_after is None else min(t['steps'], stop_after)
    if target < 1:
        raise ValueError('stop_after must be positive')
    model.train()
    with (out / 'metrics.jsonl').open('a', buffering=1) as metrics_file:
        while step < target:
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, t)
            for group in optimizer.param_groups:
                group['lr'] = lr
            train_loss = 0.0
            for _ in range(t['grad_accum']):
                x, y = batch(streams['train'], t['batch_size'], model.config.seq_len, generator, device)
                with amp(device, t['precision']):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite training loss at step {step + 1}')
                (loss / t['grad_accum']).backward()
                train_loss += loss.detach().item() / t['grad_accum']
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            train_seconds += time.perf_counter() - start
            step += 1
            record = {'step': step, 'train_loss': train_loss, 'lr': lr}
            if step % t['eval_every'] == 0 or step == target:
                val = evaluate(model, streams['validation'], t['batch_size'], device, t['precision'])
                record['validation'] = val
                improved = val['loss'] < best_loss
                best_loss = min(best_loss, val['loss'])
                ck = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                      'best_loss': best_loss, 'train_seconds': train_seconds, 'config': c, 'manifest': manifest,
                      'architecture': architecture, 'calibration': calibration, 'batch_rng': generator.get_state(),
                      'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
                save_checkpoint(out / 'last.pt', ck)
                if improved:
                    save_checkpoint(out / 'best.pt', ck)
            # Persist every optimizer step; stdout remains throttled by log_every.
            metrics_file.write(json.dumps(record) + '\n')
            if step % t['log_every'] == 0 or 'validation' in record:
                print(json.dumps(record), flush=True)
    if step < t['steps']:
        return {'status': 'paused', 'step': step}
    ck = load_checkpoint(out / 'best.pt')
    model.load_state_dict(ck['model'])
    test = evaluate(model, streams['validation' if calibration else 'test'], t['batch_size'], device, t['precision'])
    tokens = step * t['batch_size'] * t['grad_accum'] * model.config.seq_len
    result = {**metadata, 'status': 'calibration_complete' if calibration else 'complete', 'steps': step, 'train_tokens': tokens,
              'best_step': ck['step'], 'best_validation_loss': best_loss, ('validation_timing_pass' if calibration else 'test'): test,
              'train_seconds': train_seconds, 'train_tokens_per_second': tokens / train_seconds,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def comparison(root):
    root = Path(root)
    results = [json.loads((root / a / 'result.json').read_text()) for a in ('standard', 'looped')]
    a, b = results
    if a['config'] != b['config'] or a['manifest'] != b['manifest'] or a['train_tokens'] != b['train_tokens']:
        raise ValueError('comparison requires identical configs, data and token budgets')
    for result, arch in zip(results, ('standard', 'looped')):
        if result['status'] != 'complete' or result['architecture'] != arch:
            raise ValueError('comparison requires both completed architectures')
    report = {'note': 'Matched effective depth and token budget; unique parameter counts differ. Single-seed results are not statistical evidence.',
              'runs': results, 'looped_minus_standard_test_loss': b['test']['loss'] - a['test']['loss']}
    (root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Matched-depth comparison', '', report['note'], '',
             '| Model | Unique layers | Effective depth | Parameters | Train tokens | Test loss | Test BPE perplexity |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['architecture']} | {r['unique_layers']} | {r['effective_depth']} | {r['parameters']:,} | {r['train_tokens']:,} | {r['test']['loss']:.4f} | {r['test']['perplexity']:.4f} |")
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    # A completed comparison always gets the final, full-step plots.
    from .plotting import plot_runs
    plot_runs(root, once=True)
    return report
