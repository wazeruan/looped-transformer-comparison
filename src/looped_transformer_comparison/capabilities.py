"""Reproducible zero-shot capability checks for trained causal language models."""
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F


WRITING_PROMPTS = [
    'Write a clear paragraph explaining why leaves change color in autumn.\n\n',
    'Write a short answer to this question: What causes a solar eclipse?\n\nAnswer:',
    'Write a concise argument for and against building a new public library.\n\n',
    'Continue this story in a coherent style:\nThe train arrived at the empty station just before dawn.\n\n',
    'Explain step by step how to estimate the area of an irregular garden.\n\n',
    'Write a helpful response to: How can I study more effectively for an exam?\n\nResponse:',
]


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


@torch.no_grad()
def choice_logprob(model, tokenizer, prompt, choice, device):
    """Mean conditional log probability of a completion, excluding prompt tokens."""
    prefix = tokenizer.encode(prompt).ids
    full = tokenizer.encode(prompt + choice).ids
    if len(prefix) < 1 or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError('prompt must end at a tokenizer-stable boundary')
    tokens = torch.tensor([full], device=device)
    logits = model(tokens[:, :-1]).float()
    log_probs = F.log_softmax(logits, dim=-1)
    targets = tokens[:, 1:]
    first = len(prefix) - 1
    values = log_probs[0, first:, :].gather(-1, targets[0, first:].unsqueeze(-1)).squeeze(-1)
    return values.mean().item()


def _hellaswag(limit):
    from datasets import load_dataset
    rows = load_dataset('Rowan/hellaswag', split='validation')
    for row in rows.select(range(min(limit, len(rows)))):
        prompt = row['ctx_a'] + ' ' + row['ctx_b'].capitalize()
        yield {'id': row['ind'], 'prompt': prompt, 'choices': row['endings'], 'answer': int(row['label'])}


def _arc_easy(limit):
    from datasets import load_dataset
    rows = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
    for row in rows.select(range(min(limit, len(rows)))):
        labels, choices = row['choices']['label'], row['choices']['text']
        answer = labels.index(row['answerKey'])
        prompt = 'Question: ' + row['question'] + '\nAnswer:'
        yield {'id': row['id'], 'prompt': prompt, 'choices': choices, 'answer': answer}


TASKS = {'hellaswag': _hellaswag, 'arc_easy': _arc_easy}


def evaluate_choices(model, tokenizer, task_names, limit, device):
    results = {}
    for task in task_names:
        if task not in TASKS:
            raise ValueError(f'unknown task: {task}')
        correct, scored = 0, []
        for item in TASKS[task](limit):
            scores = [choice_logprob(model, tokenizer, item['prompt'], ' ' + choice, device)
                      for choice in item['choices']]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            correct += prediction == item['answer']
            scored.append({'id': item['id'], 'prediction': prediction, 'answer': item['answer'], 'scores': scores})
        results[task] = {'examples': len(scored), 'correct': correct,
                         'accuracy': correct / len(scored) if scored else math.nan, 'records': scored}
    return results


@torch.no_grad()
def writing_samples(model, tokenizer, device, max_new_tokens=128, seed=42):
    samples = []
    for number, prompt in enumerate(WRITING_PROMPTS):
        torch.manual_seed(seed + number)
        ids = tokenizer.encode(prompt).ids
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids[-model.config.seq_len:]], device=device)
            logits = model(input_ids)[:, -1].float()
            next_id = torch.multinomial((logits / 0.8).softmax(-1), 1).item()
            ids.append(next_id)
            if next_id == tokenizer.token_to_id('<eos>'):
                break
        samples.append({'prompt': prompt, 'continuation': tokenizer.decode(ids[len(tokenizer.encode(prompt).ids):])})
    return samples


def run_capability_evaluation(model, tokenizer, output, tasks, limit, device, max_new_tokens, seed):
    if limit < 1 or max_new_tokens < 1:
        raise ValueError('limit and max_new_tokens must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    report = {'tasks': evaluate_choices(model, tokenizer, tasks, limit, device),
              'writing_samples': writing_samples(model, tokenizer, device, max_new_tokens, seed)}
    _atomic_json(output / 'capabilities.json', report)
    return report
