import argparse
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .data import prepare, load_data, digest
from .engine import train, comparison, load_checkpoint, device_for, evaluate, model_config_for
from .model import LanguageModel, ModelConfig


def restored(checkpoint, device):
    ck = load_checkpoint(checkpoint)
    model = LanguageModel(model_config_for(ck['config'], ck['architecture']), ck['architecture']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck


def main():
    parser = argparse.ArgumentParser(description='Matched-depth standard vs looped causal language models')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', default='data/wikitext103')
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--local-dir', help='Folder containing train.txt, validation.txt, test.txt')
    p.add_argument('--mixture-file', help='JSON source lists for a mixed pretraining corpus')
    p.add_argument('--dataset-config', default='wikitext-103-raw-v1', choices=['wikitext-2-raw-v1', 'wikitext-103-raw-v1'])
    for name in ('train', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/h100.json')
        p.add_argument('--data', default='data/wikitext103')
        p.add_argument('--output', default='runs/h100-350m')
        p.add_argument('--resume', action='store_true')
        if name == 'train':
            p.add_argument('--calibration', action='store_true', help=argparse.SUPPRESS)
            p.add_argument('--architecture', required=True, choices=['standard', 'looped'])
            p.add_argument('--stop-after', type=int, help='Checkpoint and pause after this absolute optimizer step')
    p = sub.add_parser('budget')
    p.add_argument('--config', default='configs/h100-8h.json')
    p.add_argument('--data', default='data/wikitext103')
    p.add_argument('--output', default='runs/h100-350m-wiki103-8h')
    p.add_argument('--hours', type=float, default=8.0)
    p.add_argument('--reserve-minutes', type=float, default=5.0)
    p.add_argument('--calibration-steps', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--compute-matched', action='store_true',
                   help='allocate equal measured training time per architecture; steps/tokens may differ')
    p = sub.add_parser('report')
    p.add_argument('--output', default='runs/h100-350m')
    p = sub.add_parser('plot')
    p.add_argument('--output', default='runs/h100-350m-wiki103-8h')
    p.add_argument('--watch', action='store_true', help='keep polling metrics.jsonl until interrupted')
    p.add_argument('--interval', type=float, default=30.0, help='poll interval in seconds')
    p.add_argument('--once', action='store_true', help='render once even with --watch')
    p.add_argument('--dpi', type=int, default=130)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/wikitext103')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    p = sub.add_parser('generate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', default='data/wikitext103/tokenizer.json')
    p.add_argument('--prompt', required=True)
    p.add_argument('--max-new-tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('capability-eval')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', default='data/wikitext103/tokenizer.json')
    p.add_argument('--output', required=True)
    p.add_argument('--tasks', default='hellaswag,arc_easy')
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('check')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--require-h100', action='store_true',
                   help='require a selected H100 with at least 75 GiB and BF16 support')
    a = parser.parse_args()
    if a.command == 'prepare':
        args = (a.output, a.vocab_size, a.local_dir, a.dataset_config)
        if a.mixture_file:
            args += (a.mixture_file,)
        print(json.dumps(prepare(*args), indent=2))
    elif a.command == 'train':
        kwargs = {'calibration': True} if a.calibration else {}
        print(json.dumps(train(a.config, a.data, a.output, a.architecture, a.resume, a.stop_after, **kwargs), indent=2))
    elif a.command == 'budget':
        from .budget import run_budget
        args = (a.config, a.data, a.output, a.hours, a.reserve_minutes, a.calibration_steps, a.resume)
        if a.compute_matched:
            args += (True,)
        print(json.dumps(run_budget(*args), indent=2))
    elif a.command == 'compare':
        root = Path(a.output)
        # Reject occupied destinations before starting either architecture.
        if not a.resume:
            for arch in ('standard', 'looped'):
                path = root / arch
                if path.exists() and any(path.iterdir()):
                    raise ValueError(f'{path} is occupied; use --resume or a new output')
        for arch in ('standard', 'looped'):
            path = root / arch
            resume = a.resume and (path / 'last.pt').exists()
            train(a.config, a.data, path, arch, resume)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(json.dumps(comparison(root), indent=2))
    elif a.command == 'report':
        budget = Path(a.output) / 'budget.json'
        matched = json.loads(budget.read_text()).get('compute_matched', False) if budget.exists() else False
        report = comparison(a.output, allow_unequal_tokens=True) if matched else comparison(a.output)
        print(json.dumps(report, indent=2))
    elif a.command == 'plot':
        if a.dpi < 72:
            parser.error('--dpi must be at least 72')
        from .plotting import plot_runs
        print(json.dumps(plot_runs(a.output, a.watch, a.interval, a.once, a.dpi), indent=2))
    elif a.command == 'evaluate':
        if a.batch_size < 1:
            parser.error('--batch-size must be positive')
        device = device_for(a.device)
        model, ck = restored(a.checkpoint, device)
        streams, manifest = load_data(a.data, model.config.seq_len)
        if manifest != ck['manifest']:
            raise ValueError('checkpoint/data manifest mismatch')
        print(json.dumps(evaluate(model, streams[a.split], a.batch_size, device, 'fp32'), indent=2))
    elif a.command == 'generate':
        if a.max_new_tokens < 1 or a.temperature < 0:
            parser.error('max-new-tokens must be positive; temperature must be nonnegative')
        device = device_for(a.device)
        model, ck = restored(a.checkpoint, device)
        if digest(a.tokenizer) != ck['manifest']['tokenizer_sha256']:
            raise ValueError('checkpoint/tokenizer mismatch')
        tokenizer = Tokenizer.from_file(a.tokenizer)
        ids = tokenizer.encode(a.prompt).ids
        if not ids:
            parser.error('prompt must produce at least one token')
        torch.manual_seed(a.seed)
        with torch.no_grad():
            for _ in range(a.max_new_tokens):
                logits = model(torch.tensor([ids[-model.config.seq_len:]], device=device))[:, -1].float()
                token = logits.argmax(-1) if a.temperature == 0 else torch.multinomial((logits / a.temperature).softmax(-1), 1).view(-1)
                ids.append(token.item())
                if token.item() == tokenizer.token_to_id('<eos>'):
                    break
        print(tokenizer.decode(ids))
    elif a.command == 'capability-eval':
        from .capabilities import run_capability_evaluation
        device = device_for(a.device)
        model, ck = restored(a.checkpoint, device)
        if digest(a.tokenizer) != ck['manifest']['tokenizer_sha256']:
            raise ValueError('checkpoint/tokenizer mismatch')
        tasks = [task.strip() for task in a.tasks.split(',') if task.strip()]
        print(json.dumps(run_capability_evaluation(
            model, Tokenizer.from_file(a.tokenizer), a.output, tasks, a.limit, device,
            a.max_new_tokens, a.seed), indent=2))
    elif a.command == 'check':
        available = torch.cuda.is_available()
        devices = [{'name': torch.cuda.get_device_name(i),
                    'memory_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
                   for i in range(torch.cuda.device_count())]
        bf16 = torch.cuda.is_bf16_supported() if available else False
        selected = devices[torch.cuda.current_device()] if available else None
        print(json.dumps({'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
                          'cuda_available': available, 'devices': devices,
                          'selected_device': selected, 'bf16': bf16}, indent=2))
        if (a.require_cuda or a.require_h100) and (not available or not bf16):
            raise SystemExit('CUDA with bf16 support required for configs/h100.json')
        if a.require_h100 and ('H100' not in selected['name'].upper()
                               or selected['memory_gib'] < 75):
            raise SystemExit(
                'A full NVIDIA H100 80GB is required: selected device must contain '
                '"H100" and expose at least 75 GiB')


if __name__ == '__main__':
    main()
