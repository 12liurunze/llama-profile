from pathlib import Path
from .common import ROOT, digest, load, save, sha


def prepare(model, output, context=4096, max_new_tokens=1024):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to replace prepared data: ' + str(output))
    from transformers import AutoTokenizer
    import json
    source = ROOT/'data/redhat8_4000.jsonl'
    manifest = load(ROOT/'data/redhat8_4000.manifest.json')
    assert sha(source) == manifest['output']['sha256'], 'Raw data hash mismatch'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rows = []
    cache = {}
    for line in source.read_text(encoding='utf-8').splitlines():
        raw = json.loads(line)
        prompt = raw['prompt']
        if prompt not in cache:
            ids = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                tokenize=True, add_generation_prompt=True, return_dict=False)
            if hasattr(ids, 'keys'):
                ids = ids['input_ids']
            if not isinstance(ids, list) or not ids or any(type(t) is not int for t in ids):
                raise ValueError('Tokenizer did not return list[int]')
            cache[prompt] = ids
        ids = cache[prompt]
        if len(ids) + max_new_tokens + 16 > context:
            raise ValueError(f'{raw["id"]}: {len(ids)} input tokens + cap do not fit context {context}; no silent truncation')
        rows.append(dict(id=raw['id'], subset=raw['subset'], source_id=raw['source_id'],
                         prompt_sha256=raw['prompt_sha256'], input_ids=ids))
    assert len(rows) == 4000 and len({r['id'] for r in rows}) == 4000
    model_root = Path(model).resolve()
    token_files = {p.name:sha(p) for p in model_root.iterdir() if p.is_file()
                   and (p.name.startswith('tokenizer') or p.name in ('chat_template.jinja', 'special_tokens_map.json', 'config.json'))}
    save(output, dict(format_version=1, model_path=str(model_root), context=context,
        max_new_tokens=max_new_tokens, raw_sha256=sha(source), tokenizer_files=token_files,
        chat_template_sha256=digest(tokenizer.chat_template), count=4000,
        min_input_tokens=min(len(r['input_ids']) for r in rows),
        max_input_tokens=max(len(r['input_ids']) for r in rows),
        mean_input_tokens=sum(len(r['input_ids']) for r in rows)/len(rows),
        inputs_sha256=digest([r['input_ids'] for r in rows]), rows=rows))
    print('PREPARED', output, '4000 requests; source prompts are repeated')
    return dict(output=str(output.resolve()),requests=4000,distinct_prompts=len(cache),sha256=sha(output))
