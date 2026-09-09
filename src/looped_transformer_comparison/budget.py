"""Measured paired training budget with a process-level elapsed-time watchdog."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .data import digest
from .engine import read_config, comparison


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def planned_steps(measurements, remaining_seconds, reserve_seconds, eval_every, max_steps, safety=1.05,
                  compute_matched=False):
    """Overhead estimate conservatively includes init, validation, test and saves."""
    if len(measurements) != 2 or not 0 < reserve_seconds < remaining_seconds:
        raise ValueError('Insufficient budget after calibration and reserve')
    if safety < 1 or eval_every < 1 or max_steps < 2:
        raise ValueError('Invalid planner settings')
    step_seconds = sum(m['train_seconds'] / m['steps'] for m in measurements)
    overhead = sum(max(0.0, m['wall_seconds'] - m['train_seconds']) for m in measurements)
    if not math.isfinite(step_seconds + overhead) or step_seconds <= 0:
        raise ValueError('Invalid calibration timings')
    available = remaining_seconds - reserve_seconds
    def estimate(n):
        return safety * (n * step_seconds + (math.ceil(n / eval_every) + 1) * overhead)
    low, high = 0, max_steps
    while low < high:
        mid = (low + high + 1) // 2
        if estimate(mid) <= available:
            low = mid
        else:
            high = mid - 1
    if low < 2:
        raise ValueError('Not enough time for both models; increase budget or reduce configuration')
    result = {'steps': low, 'predicted_main_seconds': estimate(low),
            'pair_train_seconds_per_step': step_seconds, 'pair_overhead_seconds': overhead,
            'safety_factor': safety}
    if compute_matched:
        available = (remaining_seconds - reserve_seconds) / safety
        per_arch = []
        for measurement in measurements:
            step = measurement['train_seconds'] / measurement['steps']
            overhead_arch = max(0.0, measurement['wall_seconds'] - measurement['train_seconds'])
            budget_arch = available / 2
            def estimate_arch(n):
                return n * step + (math.ceil(n / eval_every) + 1) * overhead_arch
            low, high = 0, max_steps
            while low < high:
                mid = (low + high + 1) // 2
                if estimate_arch(mid) <= budget_arch:
                    low = mid
                else:
                    high = mid - 1
            if low < 2:
                raise ValueError('Not enough time for two compute-matched architectures')
            per_arch.append(low)
        result['steps_by_architecture'] = {'standard': per_arch[0], 'looped': per_arch[1]}
        result['predicted_main_seconds'] = safety * sum(
            per_arch[i] * measurements[i]['train_seconds'] / measurements[i]['steps']
            + (math.ceil(per_arch[i] / eval_every) + 1) * max(0.0, measurements[i]['wall_seconds'] - measurements[i]['train_seconds'])
            for i in range(2))
    return result


def worker(command, log_path, deadline):
    """Unix process group is terminated at deadline; reserve 5s for cleanup."""
    remaining = deadline - time.time() - 5
    if remaining <= 0:
        raise TimeoutError('Eight-hour/run deadline reached before worker launch')
    start = time.monotonic()
    with log_path.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=max(0.01, min(5, deadline - time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f'Worker stopped at time limit; see {log_path}')
    if code:
        raise RuntimeError(f'Worker exited with code {code}; see {log_path}')
    return time.monotonic() - start


def run_budget(config_path, data_dir, output, hours=8.0, reserve_minutes=5.0,
               calibration_steps=128, resume=False, compute_matched=False):
    if not math.isfinite(hours) or not 0 < hours <= 8:
        raise ValueError('hours must be greater than zero and at most 8 (total for both models)')
    if not math.isfinite(reserve_minutes) or not 0 < reserve_minutes * 60 < hours * 3600:
        raise ValueError('reserve-minutes must be positive and smaller than budget')
    if calibration_steps < 2:
        raise ValueError('calibration-steps must be at least 2')
    config_path, data_dir, root = Path(config_path).resolve(), Path(data_dir).resolve(), Path(output).resolve()
    config = read_config(config_path)
    identity = {'config': config, 'data_directory': str(data_dir),
                'manifest_sha256': digest(data_dir / 'manifest.json'), 'hours': hours,
                'reserve_minutes': reserve_minutes, 'calibration_steps': calibration_steps,
                'compute_matched': compute_matched}
    state_path = root / 'budget.json'
    if resume:
        if not state_path.exists():
            raise ValueError('budget resume requires budget.json')
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('Budget resume settings/data mismatch')
        if state['status'] == 'complete':
            return comparison(root, allow_unequal_tokens=state.get('compute_matched', False))
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('Output is not empty; use --resume or a new output directory')
        root.mkdir(parents=True, exist_ok=True)
        state = {'identity': identity, 'started_unix': time.time(), 'status': 'calibrating', 'measurements': []}
        state['deadline_unix'] = state['started_unix'] + hours * 3600
        write_json(state_path, state)
    deadline = state['deadline_unix']
    if time.time() >= deadline:
        raise ValueError('Original budget expired; resume does not grant another eight hours')
    def command(cfg, path, arch):
        args = [sys.executable, '-m', 'looped_transformer_comparison.cli', 'train',
                '--config', str(cfg), '--data', str(data_dir), '--output', str(path), '--architecture', arch]
        if (path / 'last.pt').exists():
            args.append('--resume')
        return args
    try:
        if 'plan' not in state:
            calibration = root / 'calibration'
            calibration.mkdir(exist_ok=True)
            cc = copy.deepcopy(config)
            cc['training'].update(steps=calibration_steps, warmup_steps=min(2, calibration_steps - 1),
                                  eval_every=calibration_steps, log_every=1)
            cp = calibration / 'config.json'
            write_json(cp, cc)
            # Calibration models are discarded; main arms restart from the same seed.
            # Avoid test-set feedback: use validation for calibration's final timing pass.
            for arch in ('standard', 'looped'):
                if any(m['architecture'] == arch for m in state['measurements']):
                    continue
                path = calibration / arch
                if path.exists() and not (path / 'last.pt').exists():
                    raise ValueError('Interrupted calibration without checkpoint: use a fresh output')
                args = command(cp, path, arch) + ['--calibration']
                print(f'Calibrating {arch}; log: {calibration / (arch + ".log")}', flush=True)
                wall = worker(args, calibration / f'{arch}.log', deadline)
                result = json.loads((path / 'result.json').read_text())
                # Resumed timing no longer represents a full calibration; reject it.
                if '--resume' in args:
                    raise ValueError('Calibration interrupted: rerun with a fresh output for valid timings')
                state['measurements'].append({'architecture': arch, 'wall_seconds': wall,
                                              'train_seconds': result['train_seconds'], 'steps': result['steps']})
                write_json(state_path, state)
            plan = planned_steps(state['measurements'], deadline - time.time(), reserve_minutes * 60,
                                 config['training']['eval_every'], config['training']['steps'],
                                 compute_matched=compute_matched)
            resolved = copy.deepcopy(config)
            resolved['training']['steps'] = plan['steps']
            resolved['training']['warmup_steps'] = min(config['training']['warmup_steps'], max(1, plan['steps'] // 20))
            tokens_per_step = config['training']['batch_size'] * config['training']['grad_accum'] * config['model']['seq_len']
            plan['tokens_per_model'] = (plan['steps'] * tokens_per_step if not compute_matched else
                                        {a: n * tokens_per_step for a, n in plan['steps_by_architecture'].items()})
            state.update(plan=plan, status='training')
            write_json(root / 'resolved-config.json', resolved)
            if compute_matched:
                for arch, steps in plan['steps_by_architecture'].items():
                    arch_config = copy.deepcopy(resolved)
                    arch_config['training']['steps'] = steps
                    arch_config['training']['warmup_steps'] = min(arch_config['training']['warmup_steps'], max(1, steps // 20))
                    write_json(root / f'resolved-config-{arch}.json', arch_config)
            write_json(state_path, state)
            print(json.dumps(plan, indent=2), flush=True)
        for arch in ('standard', 'looped'):
            path = root / arch
            if (path / 'result.json').exists():
                continue
            if path.exists() and not (path / 'last.pt').exists():
                raise ValueError(f'{arch} interrupted before first checkpoint; use a fresh output')
            print(f'Training {arch}; log: {root / (arch + ".log")}', flush=True)
            cfg = root / f'resolved-config-{arch}.json' if compute_matched else root / 'resolved-config.json'
            worker(command(cfg, path, arch), root / f'{arch}.log', deadline)
        report = comparison(root, allow_unequal_tokens=compute_matched)
        state.update(status='complete', elapsed_seconds=time.time() - state['started_unix'])
        write_json(state_path, state)
        return report
    except (TimeoutError, RuntimeError, ValueError) as error:
        state.update(status='incomplete', reason=str(error), elapsed_seconds=time.time() - state['started_unix'])
        write_json(state_path, state)
        raise
