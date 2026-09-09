"""Train-only byte BPE and split-preserving WikiText token streams."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DATASET_REVISION = 'f776294184f13b8ff2337b3841cf9269a6216d1e'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _mixture_rows(sources):
    from datasets import load_dataset
    active = []
    for source in sources:
        required = {'dataset', 'split'}
        if missing := required - set(source):
            raise ValueError(f'mixture source missing: {", ".join(sorted(missing))}')
        rows = load_dataset(source['dataset'], source.get('config'), split=source['split'],
                            revision=source.get('revision'), streaming=source.get('streaming', True))
        limit = source.get('max_rows')
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError('mixture max_rows must be a positive integer')
        active.append((iter(rows), source.get('text_field', 'text'), int(source.get('weight', 1)), limit))
    if not active or any(weight < 1 for _, _, weight, _ in active):
        raise ValueError('mixture needs non-empty sources with positive integer weights')
    while active:
        remaining = []
        for iterator, field, weight, limit in active:
            try:
                for _ in range(weight):
                    if limit is not None and limit <= 0:
                        raise StopIteration
                    row = next(iterator)
                    text = row[field]
                    if not isinstance(text, str):
                        raise ValueError(f'mixture field {field} must contain text')
                    yield text
                    if limit is not None:
                        limit -= 1
                remaining.append((iterator, field, weight, limit))
            except StopIteration:
                continue
        active = remaining


def prepare(output, vocab_size=8192, local_dir=None, dataset_config='wikitext-103-raw-v1', mixture_file=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; use a new data directory')
    if vocab_size < 258:
        raise ValueError('vocab_size must be at least 258 for byte BPE')
    if local_dir and mixture_file:
        raise ValueError('local_dir and mixture_file are mutually exclusive')
    if local_dir:
        def local_rows(split):
            with Path(local_dir, f'{split}.txt').open() as handle:
                for line in handle:
                    yield line.rstrip('\r\n')
        texts = {s: (lambda s=s: local_rows(s)) for s in ('train', 'validation', 'test')}
        source = {'local_files': {s: digest(Path(local_dir, f'{s}.txt')) for s in texts}}
    elif mixture_file:
        mixture = json.loads(Path(mixture_file).read_text())
        if set(mixture) != {'train', 'validation', 'test'} or not all(isinstance(mixture[s], list) for s in mixture):
            raise ValueError('mixture file must contain train, validation and test source lists')
        texts = {s: (lambda s=s: _mixture_rows(mixture[s])) for s in ('train', 'validation', 'test')}
        source = {'mixture_file_sha256': digest(mixture_file), 'mixture': mixture}
    else:
        from datasets import load_dataset
        ds = load_dataset('Salesforce/wikitext', dataset_config, revision=DATASET_REVISION)
        texts = {s: (lambda s=s: (row['text'] for row in ds[s])) for s in ('train', 'validation', 'test')}
        source = {'dataset': 'Salesforce/wikitext', 'config': dataset_config, 'revision': DATASET_REVISION}
    for s, rows in texts.items():
        if not any(row.strip() for row in rows()):
            raise ValueError(f'{s} has no text')
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator((x for x in texts['train']() if x.strip()), trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=['<unk>', '<eos>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / 'tokenizer.json'))
    manifest = {'source': source, 'vocab_size': tokenizer.get_vocab_size(),
                'tokenizer_sha256': digest(output / 'tokenizer.json'), 'splits': {}}
    for split, rows in texts.items():
        ids, count = [], 0
        with (output / f'{split}.bin').open('wb') as handle:
            for row in rows():
                if row.strip():
                    ids.extend(tokenizer.encode(row.rstrip('\r\n') if local_dir else row).ids)
                    ids.append(tokenizer.token_to_id('<eos>'))
                if len(ids) >= 1_000_000:
                    np.asarray(ids, dtype=np.uint32).tofile(handle)
                    count += len(ids)
                    ids.clear()
            np.asarray(ids, dtype=np.uint32).tofile(handle)
            count += len(ids)
        manifest['splits'][split] = {'tokens': count, 'sha256': digest(output / f'{split}.bin')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_data(directory, seq_len):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('tokenizer checksum mismatch')
    streams = {}
    for s in ('train', 'validation', 'test'):
        path = directory / f'{s}.bin'
        if digest(path) != manifest['splits'][s]['sha256']:
            raise ValueError(f'{s} checksum mismatch')
        streams[s] = np.memmap(path, dtype=np.uint32, mode='r')
        if len(streams[s]) != manifest['splits'][s]['tokens'] or len(streams[s]) < seq_len + 1:
            raise ValueError(f'{s} token count invalid or too short for seq_len={seq_len}')
        if streams[s].max() >= manifest['vocab_size']:
            raise ValueError(f'{s} token ID outside vocabulary')
    return streams, manifest


def batch(stream, batch_size, seq_len, generator, device):
    offsets = torch.randint(len(stream) - seq_len, (batch_size,), generator=generator).tolist()
    a = torch.from_numpy(np.stack([stream[i:i + seq_len + 1].astype(np.int64) for i in offsets])).to(device)
    return a[:, :-1], a[:, 1:]
