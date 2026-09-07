import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from looped_transformer_comparison.model import ModelConfig, LanguageModel
from looped_transformer_comparison.data import prepare, load_data, batch
from looped_transformer_comparison.engine import evaluate, train, load_checkpoint, comparison

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for s in ('train', 'validation', 'test'):
        (raw / f'{s}.txt').write_text(('small cats learn words and small dogs read books.\n' * 8) if s == 'train' else ('unseen quantum zebras read books.\n' * 3))
    out = tmp_path / 'data'
    prepare(out, 280, raw)
    return out

@pytest.fixture
def config(tmp_path):
    c = json.loads((ROOT / 'configs/smoke.json').read_text())
    c['model'].update(seq_len=8, width=16, heads=2, dropout=0.2)
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(c))
    return p

@pytest.mark.parametrize('updates', [{'loops': 3}, {'width': 7}, {'depth': 0}, {'dropout': 1}])
def test_invalid_configs(updates):
    c = dict(vocab_size=20, seq_len=8, width=16, heads=2, depth=4, loop_layers=2, loops=2)
    c.update(updates)
    with pytest.raises(ValueError):
        ModelConfig(**c)


def models(loops=2):
    c = ModelConfig(vocab_size=20, seq_len=8, width=16, heads=2, depth=4, loop_layers=4//loops, loops=loops)
    torch.manual_seed(9)
    a = LanguageModel(c, 'standard')
    torch.manual_seed(9)
    b = LanguageModel(c, 'looped')
    return a, b


def test_depth_shared_initialization_and_loop_one():
    a,b = models()
    assert len(a.blocks)==4 and len(b.blocks)==2
    assert sum(p.numel() for p in b.parameters()) < sum(p.numel() for p in a.parameters())
    for key,value in b.state_dict().items():
        assert torch.equal(value, a.state_dict()[key]), key
    calls=[]
    handles=[block.register_forward_hook(lambda *args: calls.append(1)) for block in b.blocks]
    b(torch.tensor([[1,2,3]]))
    assert len(calls)==4
    for handle in handles: handle.remove()
    a,b = models(1)
    assert torch.equal(a(torch.tensor([[1,2,3]])), b(torch.tensor([[1,2,3]])))

@pytest.mark.parametrize('arch', ['standard','looped'])
def test_causal_mask(arch):
    a,b = models()
    m = a if arch=='standard' else b
    x=torch.tensor([[1,2,3,4,5]])
    y=x.clone(); y[:,3:]=9
    torch.testing.assert_close(m(x)[:,:3],m(y)[:,:3], rtol=0, atol=0)


def test_shared_gradient_matches_sum_of_unrolled_independent_blocks():
    unrolled, shared=models()
    for i, block in enumerate(unrolled.blocks):
        block.load_state_dict(shared.blocks[i%2].state_dict())
    x=torch.tensor([[1,2,3,4]])
    target=torch.tensor([[2,3,4,5]])
    for model in (unrolled,shared):
        F.cross_entropy(model(x).reshape(-1,20),target.flatten()).backward()
    for i, block in enumerate(shared.blocks):
        grads=[dict(unrolled.blocks[j].named_parameters()) for j in (i,i+2)]
        for name,p in block.named_parameters():
            torch.testing.assert_close(p.grad, grads[0][name].grad+grads[1][name].grad, rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(shared.token.weight.grad,unrolled.token.weight.grad,rtol=1e-5,atol=1e-7)


def test_train_only_tokenizer_checksums_and_occupied(dataset,tmp_path):
    before=json.loads((dataset/'tokenizer.json').read_text())
    raw=tmp_path/'raw'
    (raw/'validation.txt').write_text('totally distinct foreign validation content '*30)
    other=tmp_path/'other'
    prepare(other,280,raw)
    assert json.loads((other/'tokenizer.json').read_text()) == before
    streams,manifest=load_data(dataset,8)
    assert all(len(streams[s])==manifest['splits'][s]['tokens'] for s in streams)
    with pytest.raises(ValueError,match='not empty'): prepare(dataset,280,raw)
    with (dataset/'test.bin').open('ab') as f: f.write(b'1234')
    with pytest.raises(ValueError,match='checksum'): load_data(dataset,8)


def test_batch_deterministic_and_shifted():
    stream=np.arange(100,dtype=np.uint32)
    a=torch.Generator().manual_seed(42); b=torch.Generator().manual_seed(42)
    for _ in range(3):
        x,y=batch(stream,3,8,a,'cpu'); u,v=batch(stream,3,8,b,'cpu')
        assert torch.equal(x,u) and torch.equal(y,v) and torch.equal(x+1,y)


def test_eval_token_weighted_tail_and_training_mode():
    class Toy(torch.nn.Module):
        config=type('Config',(),{'seq_len':4})()
        def forward(self,x):
            return F.one_hot(x%3,3).float()*2
    model=Toy()
    stream=np.array([0,1,1,1,2,1,1,0,2,2,0,1],dtype=np.uint32)
    expected=F.cross_entropy(model(torch.tensor(stream[:-1].astype('int64'))),torch.tensor(stream[1:].astype('int64'))).item()
    for bs in (1,2,8):
        result=evaluate(model,stream,bs,torch.device('cpu'),'fp32')
        assert result['tokens']==len(stream)-1
        assert result['loss']==pytest.approx(expected,abs=1e-6)
        assert model.training
    model.eval(); evaluate(model,stream,2,torch.device('cpu'),'fp32'); assert not model.training

@pytest.mark.parametrize('arch',['standard','looped'])
def test_exact_dropout_resume(dataset,config,tmp_path,arch):
    full=tmp_path/'full'; paused=tmp_path/'paused'
    result=train(config,dataset,full,arch)
    assert result['status']=='complete' and math.isfinite(result['test']['loss'])
    assert train(config,dataset,paused,arch,stop_after=1)['status']=='paused'
    train(config,dataset,paused,arch,resume=True)
    a=load_checkpoint(full/'last.pt'); b=load_checkpoint(paused/'last.pt')
    assert a['step']==b['step']==4
    for key in a['model']: assert torch.equal(a['model'][key],b['model'][key]),key
    assert torch.equal(a['batch_rng'],b['batch_rng'])
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    with pytest.raises(ValueError,match='not empty'): train(config,dataset,full,arch)
    changed=json.loads(config.read_text()); changed['training']['lr']*=2; config.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='mismatch'): train(config,dataset,paused,arch,resume=True)


def cli(*args,success=True):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    result=subprocess.run([sys.executable,'-m','looped_transformer_comparison.cli',*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert (result.returncode==0)==success,result.stdout+result.stderr
    return result


def test_paired_cli_evaluate_generate_report(dataset,config,tmp_path):
    out=tmp_path/'pair'
    cli('compare','--config',config,'--data',dataset,'--output',out)
    report=comparison(out)
    assert report['runs'][0]['train_tokens']==report['runs'][1]['train_tokens']==128
    assert (out/'comparison.md').exists()
    cli('report','--output',out)
    result=cli('evaluate','--checkpoint',out/'looped/best.pt','--data',dataset,'--device','cpu','--batch-size','3')
    assert json.loads(result.stdout)['tokens']>0
    args=('generate','--checkpoint',out/'looped/best.pt','--tokenizer',dataset/'tokenizer.json','--prompt','small cats','--max-new-tokens','3','--temperature','0','--device','cpu')
    assert cli(*args).stdout==cli(*args).stdout
    cli('compare','--config',config,'--data',dataset,'--output',out,success=False)
    cli('compare','--config',config,'--data',dataset,'--output',out,'--resume')
    r=json.loads((out/'looped/result.json').read_text()); r['train_tokens']+=1
    (out/'looped/result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='token budgets'): comparison(out)


def test_shell_syntax_and_lock():
    for path in [*ROOT.glob('scripts/*.sh'),*ROOT.glob('scripts/*.sbatch')]:
        subprocess.run(['bash','-n',str(path)],check=True)
    subprocess.run(['uv','lock','--check'],cwd=ROOT,check=True)
    cli('check')
    if not torch.cuda.is_available(): cli('check','--require-cuda',success=False)


@pytest.mark.parametrize(
    'available,bf16,names,memory_gib,selected,success,error',
    [
        (False, False, [], [], 0, False, 'CUDA with bf16 support required'),
        (True, False, ['NVIDIA H100 80GB HBM3'], [80], 0, False,
         'CUDA with bf16 support required'),
        (True, True, ['NVIDIA A100-SXM4-80GB'], [80], 0, False,
         'A full NVIDIA H100 80GB is required'),
        (True, True, ['NVIDIA H100 64GB'], [74.9], 0, False,
         'A full NVIDIA H100 80GB is required'),
        (True, True, ['NVIDIA H100 80GB HBM3'], [75], 0, True, None),
        # An acceptable H100 elsewhere must not mask an unacceptable selected GPU.
        (True, True, ['NVIDIA H100 80GB HBM3', 'NVIDIA A100-SXM4-80GB'],
         [80, 80], 1, False, 'A full NVIDIA H100 80GB is required'),
    ],
)
def test_require_h100_checks_selected_device(monkeypatch, capsys, available, bf16,
                                              names, memory_gib, selected, success, error):
    from looped_transformer_comparison import cli as cli_module
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: bf16)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: len(names))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: selected)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: names[index])
    monkeypatch.setattr(
        torch.cuda, 'get_device_properties',
        lambda index: type('Properties', (), {'total_memory': memory_gib[index] * 2**30})(),
    )
    monkeypatch.setattr(sys, 'argv', ['looped-transformer-comparison', 'check', '--require-h100'])
    if success:
        cli_module.main()
    else:
        with pytest.raises(SystemExit, match=error):
            cli_module.main()
    report = json.loads(capsys.readouterr().out)
    assert report['cuda_available'] is available
    assert report['bf16'] is bf16
    if available:
        assert report['selected_device']['name'] == names[selected]
        assert report['selected_device']['memory_gib'] == pytest.approx(memory_gib[selected])
    else:
        assert report['selected_device'] is None


def test_train_wrapper_checks_h100_before_budget_and_creates_no_state_on_failure(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_uv = fake_bin / 'uv'
    fake_uv.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$FAKE_UV_LOG"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "check" && "${FAKE_GUARD_FAIL:-0}" == "1" ]]; then exit 43; fi\n'
        'done\n'
    )
    fake_uv.chmod(0o755)
    log = tmp_path / 'uv.log'
    output = tmp_path / 'run-output'
    env = {**os.environ, 'PATH': f'{fake_bin}{os.pathsep}{os.environ["PATH"]}',
           'FAKE_UV_LOG': str(log)}
    command = [str(ROOT / 'scripts/train.sh'), '--output', str(output)]

    failed = subprocess.run(command, cwd=ROOT, env={**env, 'FAKE_GUARD_FAIL': '1'},
                            capture_output=True, text=True)
    assert failed.returncode == 43
    assert log.read_text().splitlines() == [
        'run --no-sync python -m looped_transformer_comparison.cli check --require-h100'
    ]
    assert not output.exists()

    log.unlink()
    passed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert log.read_text().splitlines() == [
        'run --no-sync python -m looped_transformer_comparison.cli check --require-h100',
        f'run --no-sync python -m looped_transformer_comparison.cli budget --output {output}',
    ]


def test_console_entrypoint_and_module_execution_have_no_runpy_warning():
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'TOKENIZERS_PARALLELISM': 'false'}
    commands = [
        ['uv', 'run', '--no-sync', 'looped-transformer-comparison', 'check'],
        ['uv', 'run', '--no-sync', 'python', '-m', 'looped_transformer_comparison.cli', 'check'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)['cuda_available'] is torch.cuda.is_available()
        assert 'RuntimeWarning' not in result.stderr

@pytest.mark.parametrize('name',['auto','cuda','cuda:2'])
def test_cuda_device_selection_mocked(monkeypatch,name):
    from looped_transformer_comparison.engine import device_for
    seen=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda: True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda: 1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda d: seen.append(d))
    expected=torch.device('cuda:2' if name=='cuda:2' else 'cuda:1')
    assert device_for(name)==expected
    assert seen==[expected]


@pytest.mark.parametrize('architecture,count,unique_layers', [
    ('standard', 350_451_328, 24), ('looped', 94_508_032, 6),
])
def test_large_h100_actual_model_size_and_depth(architecture, count, unique_layers):
    config = ModelConfig(**json.loads((ROOT / 'configs/h100.json').read_text())['model'])
    # Meta tensors verify the full architecture without allocating 350M weights.
    with torch.device('meta'):
        model = LanguageModel(config, architecture)
        assert sum(p.numel() for p in model.parameters()) == count
        assert len(model.blocks) == unique_layers
        assert config.width // config.heads == 64
        calls = []
        handles = [block.register_forward_hook(lambda *_: calls.append(1)) for block in model.blocks]
        logits = model(torch.ones((1, 8), dtype=torch.long))
        assert logits.shape == (1, 8, config.vocab_size)
        assert len(calls) == 24
        for handle in handles:
            handle.remove()


def test_h100_size_change_preserves_training_token_budget_and_small_preset():
    old = json.loads((ROOT / 'configs/h100-small.json').read_text())
    large = json.loads((ROOT / 'configs/h100.json').read_text())
    assert old['model'] == dict(vocab_size=8192, seq_len=256, width=512, heads=8,
                               depth=12, loop_layers=3, loops=4, dropout=0.0)
    assert old['training']['batch_size'] == 16 and old['training']['grad_accum'] == 4
    assert large['training']['batch_size'] == 4 and large['training']['grad_accum'] == 16
    for config in (old, large):
        training = config['training']
        tokens_per_step = training['batch_size'] * training['grad_accum'] * config['model']['seq_len']
        assert tokens_per_step == 16_384
        assert tokens_per_step * training['steps'] == 32_768_000
    assert {k: v for k, v in old['training'].items() if k not in ('batch_size', 'grad_accum')} == {
        k: v for k, v in large['training'].items() if k not in ('batch_size', 'grad_accum')}


@pytest.mark.parametrize('command', ['train', 'compare', 'report'])
def test_large_h100_cli_defaults(monkeypatch, tmp_path, command):
    from looped_transformer_comparison import cli as cli_module
    calls = []
    reports = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, 'train', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(cli_module, 'comparison', lambda output: reports.append(str(output)) or {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    argv = ['looped-transformer-comparison', command]
    if command == 'train':
        argv += ['--architecture', 'standard']
    monkeypatch.setattr(sys, 'argv', argv)
    cli_module.main()
    assert len(calls) == {'train': 1, 'compare': 2, 'report': 0}[command]
    for call in calls:
        assert call[:2] == ('configs/h100.json', 'data/wikitext103')
        expected = 'runs/h100-350m' + ('/' + call[3] if command == 'compare' else '')
        assert str(call[2]) == expected
    assert reports == ([] if command == 'train' else ['runs/h100-350m'])


def test_planner_maximal_safe_common_budget_and_overhead():
    from looped_transformer_comparison.budget import planned_steps
    timings = [{'steps': 8, 'train_seconds': 16, 'wall_seconds': 19},
               {'steps': 8, 'train_seconds': 24, 'wall_seconds': 28}]
    plan = planned_steps(timings, 200, 10, 5, 100)
    def estimate(n):
        return 1.05 * (n * 5 + (math.ceil(n / 5) + 1) * 7)
    assert plan['predicted_main_seconds'] == estimate(plan['steps'])
    assert estimate(plan['steps']) <= 190 < estimate(plan['steps'] + 1)
    assert planned_steps(timings, 200, 10, 5, 2)['steps'] == 2
    assert plan['pair_train_seconds_per_step'] == 5
    with pytest.raises(ValueError, match='Not enough time'):
        planned_steps(timings, 11, 10, 5, 100)
    with pytest.raises(ValueError):
        planned_steps(timings, 10, 10, 5, 100)
    with pytest.raises(ValueError):
        planned_steps([{'steps': 8, 'train_seconds': float('nan'), 'wall_seconds': 19}] * 2, 100, 10, 5, 100)


def test_budget_true_cpu_end_to_end_and_resume(dataset, config, tmp_path):
    out = tmp_path / 'timed-pair'
    args = ('budget', '--config', config, '--data', dataset, '--output', out,
            '--hours', '0.02', '--reserve-minutes', '0.05', '--calibration-steps', '2')
    cli(*args)
    state = json.loads((out / 'budget.json').read_text())
    assert state['status'] == 'complete'
    assert state['elapsed_seconds'] < 72
    assert state['deadline_unix'] - state['started_unix'] == pytest.approx(72)
    assert len(state['measurements']) == 2
    results = [json.loads((out / a / 'result.json').read_text()) for a in ('standard', 'looped')]
    assert results[0]['steps'] == results[1]['steps'] == state['plan']['steps'] == 4
    assert results[0]['train_tokens'] == results[1]['train_tokens'] == state['plan']['tokens_per_model'] == 128
    for arch in ('standard', 'looped'):
        calibration = json.loads((out / 'calibration' / arch / 'result.json').read_text())
        assert calibration['status'] == 'calibration_complete'
        assert 'test' not in calibration and 'validation_timing_pass' in calibration
        assert load_checkpoint(out / arch / 'last.pt')['calibration'] is False
    cli(*args, '--resume')
    assert json.loads((out / 'budget.json').read_text()) == state
    state.update(status='incomplete', deadline_unix=0)
    (out / 'budget.json').write_text(json.dumps(state))
    failed = cli(*args, '--resume', success=False)
    assert 'Original budget expired' in failed.stderr
    assert json.loads((out / 'budget.json').read_text())['deadline_unix'] == 0


def test_real_watchdog_terminates_worker_process_group(tmp_path):
    import time
    from looped_transformer_comparison.budget import worker
    marker = tmp_path / 'terminated'
    script = tmp_path / 'sleep.py'
    script.write_text('import signal, time, pathlib, sys, subprocess\n'
                      'if len(sys.argv) == 2: subprocess.Popen([sys.executable, __file__, sys.argv[1] + "-child", "child"])\n'
                      'signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).write_text("terminated"), sys.exit(0)))\n'
                      'time.sleep(60)\n')
    start = time.monotonic()
    with pytest.raises(TimeoutError, match='Worker stopped'):
        worker([sys.executable, str(script), str(marker)], tmp_path / 'log', time.time() + 5.4)
    assert marker.read_text() == 'terminated'
    child_marker = marker.with_name(marker.name + '-child')
    for _ in range(100):
        if child_marker.exists(): break
        time.sleep(0.01)
    assert child_marker.read_text() == 'terminated'
    assert time.monotonic() - start < 5
    with pytest.raises(RuntimeError, match='code 3'):
        worker([sys.executable, '-c', 'raise SystemExit(3)'], tmp_path / 'fail.log', time.time() + 10)


def test_budget_failure_persists_original_deadline(dataset, config, tmp_path, monkeypatch):
    from looped_transformer_comparison import budget
    out = tmp_path / 'failed-pair'
    def fail(*args):
        raise TimeoutError('simulated overrun')
    monkeypatch.setattr(budget, 'worker', fail)
    with pytest.raises(TimeoutError):
        budget.run_budget(config, dataset, out)
    state = json.loads((out / 'budget.json').read_text())
    assert state['status'] == 'incomplete' and 'overrun' in state['reason']
    with pytest.raises(TimeoutError):
        budget.run_budget(config, dataset, out, resume=True)
    assert json.loads((out / 'budget.json').read_text())['deadline_unix'] == state['deadline_unix']


def test_prepare_line_semantics_and_streaming_flush(tmp_path, monkeypatch):
    from looped_transformer_comparison import data as module
    from tokenizers import Tokenizer
    raw = tmp_path / 'raw'; raw.mkdir()
    content = 'small cats\r\n\r\n  \nread words\nlast line'
    for split in ('train', 'validation', 'test'):
        (raw / f'{split}.txt').write_text(content)
    out = tmp_path / 'normal'
    prepare(out, 280, raw)
    tokenizer = Tokenizer.from_file(str(out / 'tokenizer.json'))
    expected = []
    for row in content.splitlines():
        if row.strip(): expected += tokenizer.encode(row).ids + [tokenizer.token_to_id('<eos>')]
    for split in ('train', 'validation', 'test'):
        assert np.fromfile(out / f'{split}.bin', dtype=np.uint32).tolist() == expected
    class FakeTokenizer:
        def __init__(self, *_): pass
        def train_from_iterator(self, rows, trainer): assert list(rows) == ['small cats', 'read words', 'last line']
        def save(self, path): Path(path).write_text('{}')
        def get_vocab_size(self): return 280
        def token_to_id(self, token): return 2
        def encode(self, row): return type('Encoded', (), {'ids': [1] * 400_000})()
    monkeypatch.setattr(module, 'Tokenizer', FakeTokenizer)
    out = tmp_path / 'chunked'
    manifest = prepare(out, 280, raw)
    for split in ('train', 'validation', 'test'):
        values = np.fromfile(out / f'{split}.bin', dtype=np.uint32)
        assert len(values) == manifest['splits'][split]['tokens'] == 1_200_003
        assert np.where(values == 2)[0].tolist() == [400_000, 800_001, 1_200_002]
        assert (values[values != 2] == 1).all()


def test_default_budget_dispatch_and_production_controls(monkeypatch):
    from looped_transformer_comparison import cli as cli_module, budget
    import inspect
    default_calibration_steps = inspect.signature(budget.run_budget).parameters['calibration_steps'].default
    calls = []
    monkeypatch.setattr(budget, 'run_budget', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(sys, 'argv', ['looped-transformer-comparison', 'budget'])
    cli_module.main()
    assert calls == [('configs/h100-8h.json', 'data/wikitext103', 'runs/h100-350m-wiki103-8h', 8.0, 5.0, 128, False)]
    calls.clear()
    monkeypatch.setattr(sys, 'argv', ['looped-transformer-comparison', 'budget', '--calibration-steps', '7'])
    cli_module.main()
    assert calls == [('configs/h100-8h.json', 'data/wikitext103', 'runs/h100-350m-wiki103-8h', 8.0, 5.0, 7, False)]
    assert default_calibration_steps == 128
    c = json.loads((ROOT / 'configs/h100-8h.json').read_text())
    old = json.loads((ROOT / 'configs/h100.json').read_text())
    assert c['model'] == old['model']
    assert c['training']['steps'] == 1_000_000 and c['training']['eval_every'] == 500
    wrapper = (ROOT / 'scripts/train.sh').read_text()
    assert wrapper.index('check --require-h100') < wrapper.index(' budget "$@"')
    assert '#SBATCH --time=08:00:00' in (ROOT / 'scripts/train.sbatch').read_text()
    monkeypatch.setattr(cli_module, 'prepare', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(sys, 'argv', ['looped-transformer-comparison', 'prepare'])
    cli_module.main()
    assert calls[-1] == ('data/wikitext103', 8192, None, 'wikitext-103-raw-v1')
